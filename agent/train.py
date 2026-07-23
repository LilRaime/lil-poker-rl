import os
import shutil
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from adapters.simulator.sim_env import SimulatorEnv
from adapters.simulator.self_play_env import SelfPlayEnv
from adapters.simulator.league_env import LeagueEnv, RandomBotEnv


def main():
    parser = argparse.ArgumentParser(description="Train PPO RL Agent for Texas Hold'em")
    parser.add_argument("--timesteps", type=int, default=250000)
    parser.add_argument("--envs", type=int, default=4,
                        help="Number of parallel environments. "
                             "Set to your physical CPU core count for maximum speed.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-players", type=int, default=2,
                        help="Minimum table size per episode (default: 2)")
    parser.add_argument("--max-players", type=int, default=6,
                        help="Maximum table size per episode (default: 6).")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Steps collected per env before each PPO update. "
                             "Larger → fewer but better-informed updates.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Mini-batch size for PPO gradient update. "
                             "Larger → faster updates, slightly noisier gradients.")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO gradient epochs per rollout batch (default: 10). "
                             "Lower → faster updates (higher FPS), fewer gradient steps.")
    parser.add_argument("--ent-coef", type=float, default=0.015,
                        help="Entropy coefficient for PPO (default: 0.015). "
                             "Higher → more exploration.")
    parser.add_argument("--no-subproc", action="store_true",
                        help="Use DummyVecEnv (serial) instead of SubprocVecEnv. "
                             "Useful for debugging or single-env runs.")
    parser.add_argument("--self-play", action="store_true",
                        help="Use SelfPlayEnv: bots play using a frozen copy of "
                             "the current model instead of the rule-based heuristic.")
    parser.add_argument("--league", action="store_true",
                        help="Use LeagueEnv: bots are RandomBot / RuleBot / "
                             "historical PPO snapshots (implies --self-play).")
    parser.add_argument("--opponent-update-interval", type=int, default=100_000,
                        help="Steps between opponent model snapshots (default: 100 000). "
                             "Lower → faster adaptation; higher → more stable opponent.")
    parser.add_argument("--eval-games", type=int, default=1000,
                        help="Games for Elo evaluation after each snapshot (default: 1000).")
    parser.add_argument("--no-elo", action="store_true",
                        help="Skip Elo evaluation after saving each snapshot.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to train on: cpu, cuda, or auto (default: auto)")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="Number of CPU threads for PyTorch to use (default: 1, "
                             "strongly recommended when envs > 1 to avoid thread contention)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to pre-trained model checkpoint to continue training from.")
    args = parser.parse_args()

    if args.league:
        args.self_play = True

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
        os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.num_threads)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} (PyTorch threads: {torch.get_num_threads()})")

    mode = "league" if args.league else ("self-play" if args.self_play else "standard")
    print(f"Training mode: {mode}")

    os.makedirs("models", exist_ok=True)

    use_subproc = not args.no_subproc and args.envs > 1
    vec_cls = SubprocVecEnv if use_subproc else None
    env_mode = "SubprocVecEnv" if use_subproc else "DummyVecEnv"
    print(f"Vector env: {env_mode} ({args.envs} envs)")
    if args.no_subproc and args.envs > 1:
        print("WARNING: --no-subproc runs multiple envs serially; remove it for CPU-parallel rollout.")

    if args.league:
        env_cls = LeagueEnv
    elif args.self_play:
        env_cls = SelfPlayEnv
    else:
        env_cls = SimulatorEnv

    env_kwargs = dict(min_players=args.min_players, max_players=args.max_players)
    save_path = "models/ppo_mlp_agent"

    if args.self_play:
        pool_dir = "models/opponent_pool"
        os.makedirs(pool_dir, exist_ok=True)
        src = "models/ppo_mlp_agent.zip"
        dst = os.path.join(pool_dir, "snapshot_0k.zip")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"[self-play] Initialised opponent pool with {src} as snapshot_0k.zip")

        dst_fallback = "models/opponent_model.zip"
        if os.path.exists(src) and not os.path.exists(dst_fallback):
            shutil.copy(src, dst_fallback)
            print(f"[self-play] Initialised default fallback opponent from {src}")

    env = make_vec_env(
        lambda: env_cls(**env_kwargs),
        n_envs=args.envs,
        vec_env_cls=vec_cls,
    )

    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading pre-trained model from {args.load_model}...")
        model = PPO.load(
            args.load_model,
            env=env,
            device=device,
        )
        model.learning_rate = args.lr
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.n_epochs = args.n_epochs
        model.ent_coef = args.ent_coef
        model.tensorboard_log = "./ppo_poker_tensorboard/"
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            verbose=1,
            device=device,
            tensorboard_log="./ppo_poker_tensorboard/",
        )

    callbacks = []
    if args.self_play:
        class LeagueCallback(BaseCallback):
            """Saves snapshots and (optionally) runs Elo evaluation every N steps.

            Snapshot naming: ``snapshot_{k}k.zip`` (e.g. ``snapshot_100k.zip``).
            The pool is pruned to at most 15 most-recent snapshots after each save.
            Elo evaluation runs model_a vs RuleBot and vs RandomBot and writes
            results to ``models/elo.json``.
            """

            def __init__(self, interval: int, eval_games: int, run_elo: bool):
                super().__init__(verbose=1)
                self._interval = interval
                self._eval_games = eval_games
                self._run_elo = run_elo
                self._last_update = 0

            def _on_step(self) -> bool:
                if self.num_timesteps - self._last_update < self._interval:
                    return True

                pool_dir = "models/opponent_pool"
                os.makedirs(pool_dir, exist_ok=True)

                k = self.num_timesteps // 1000
                snap_name = f"snapshot_{k}k.zip"
                snap_path = os.path.join(pool_dir, snap_name)
                self.model.save(snap_path)
                self.model.save("models/opponent_model")
                self._last_update = self.num_timesteps
                print(f"[league] Saved snapshot: {snap_path}")

                try:
                    def _snap_step(name: str) -> int:
                        try:
                            return int(name.replace("snapshot_", "").replace("k.zip", ""))
                        except ValueError:
                            return -1

                    snaps = sorted(
                        [f for f in os.listdir(pool_dir)
                         if f.startswith("snapshot_") and f.endswith(".zip")],
                        key=_snap_step,
                    )
                    if len(snaps) > 15:
                        for old in snaps[:-15]:
                            os.remove(os.path.join(pool_dir, old))
                            print(f"[league] Pruned old snapshot: {old}")
                except Exception as exc:
                    print(f"[league] Error pruning pool: {exc}")

                if not self._run_elo:
                    return True

                try:
                    from agent.league import EloRegistry, evaluate_matchup
                    from agent.league import AGENT_RULE, AGENT_RANDOM

                    eval_model = PPO.load(snap_path, device="cpu")
                    elo = EloRegistry(path="models/elo.json")

                    w, l, d = evaluate_matchup(
                        eval_model,
                        lambda: SimulatorEnv(num_players=2),
                        n_games=self._eval_games,
                    )
                    elo.record_result(snap_name, AGENT_RULE, w, l, d)
                    total = w + l + d
                    print(
                        f"[elo] {snap_name} vs RuleBot   "
                        f"W:{w} L:{l} D:{d}  WR:{w/total:.1%}  "
                        f"Elo:{elo.get_rating(snap_name):.0f}"
                    )

                    w2, l2, d2 = evaluate_matchup(
                        eval_model,
                        lambda: RandomBotEnv(num_players=2),
                        n_games=max(200, self._eval_games // 5),
                    )
                    elo.record_result(snap_name, AGENT_RANDOM, w2, l2, d2)
                    total2 = w2 + l2 + d2
                    print(
                        f"[elo] {snap_name} vs RandomBot "
                        f"W:{w2} L:{l2} D:{d2}  WR:{w2/total2:.1%}  "
                        f"Elo:{elo.get_rating(snap_name):.0f}"
                    )

                    print(elo.summary())
                except Exception as exc:
                    print(f"[elo] Evaluation error: {exc}")

                return True

        callbacks.append(LeagueCallback(
            interval=args.opponent_update_interval,
            eval_games=args.eval_games,
            run_elo=not args.no_elo,
        ))

    model.learn(total_timesteps=args.timesteps, callback=callbacks or None)
    model.save(save_path)
    env.save("models/vec_normalize.pkl")
    print(f"Saved model to {save_path}.zip")


if __name__ == "__main__":
    main()

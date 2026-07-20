import os
import argparse
import numpy as np
from adapters.lil_poker.adapter import LilPokerEnv
from poker_env.action_space import PokerAction
from agent.onnx_agent import ONNXAgent

try:
    from sb3_contrib import RecurrentPPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    class DummyVecEnv:
        def __init__(self, env_fns):
            self.envs = [fn() for fn in env_fns]
            self.num_envs = len(self.envs)

        def reset(self):
            return np.array([self.envs[0].reset()[0]])

        def step(self, actions):
            obs, rew, done, truncated, info = self.envs[0].step(actions[0])
            terminal = done or truncated
            if terminal:
                info["terminal_observation"] = obs
                obs, _ = self.envs[0].reset()
            return np.array([obs]), np.array([rew]), np.array([terminal]), [info]

        def close(self):
            for env in self.envs:
                env.close()


def main():
    parser = argparse.ArgumentParser(description="Run trained RL Agent inside a live lil-poker room")
    parser.add_argument("--algo", choices=["recurrent_ppo", "ppo"], default="ppo",
                        help="Model algorithm to load (default: ppo).")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--url", type=str, default="http://localhost:8090")
    parser.add_argument("--room", type=str, required=True)
    parser.add_argument("--name", type=str, default="PPO_Bot")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference on (default: cpu). Use 'cuda' or 'auto' if desired.")
    parser.add_argument("--onnx", action="store_true",
                        help="Run inference using ONNX Runtime instead of PyTorch.")
    parser.add_argument("--onnx-model", type=str, default=None,
                        help="Path to exported ONNX model (.onnx file).")
    args = parser.parse_args()

    print(f"Connecting to room '{args.room}' at {args.url} as '{args.name}'...", flush=True)

    vec_env = DummyVecEnv([lambda: LilPokerEnv(base_url=args.url, room_id=args.room, username=args.name)])
    raw_env = vec_env.envs[0]

    default_onnx_path = "models/ppo_mlp_agent.onnx" if args.algo == "ppo" else "models/ppo_poker_agent.onnx"
    use_onnx = (
        args.onnx
        or not HAS_PYTORCH
        or (args.onnx_model is not None)
        or (args.model is not None and args.model.endswith(".onnx"))
        or (args.model is None and os.path.exists(default_onnx_path))
    )

    if use_onnx:
        onnx_path = args.onnx_model or (args.model if args.model and args.model.endswith(".onnx") else default_onnx_path)
        print(f"Loading ONNX model from: {onnx_path}")
        model = ONNXAgent(onnx_path)
        env = vec_env
        print("ONNX Bot is ready and listening for game updates...", flush=True)
    else:
        if args.model is None:
            args.model = "models/ppo_mlp_agent" if args.algo == "ppo" else "models/ppo_poker_agent"

        vec_normalize_path = "models/vec_normalize.pkl"
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, vec_env)
            env.training = False
            env.norm_reward = False
            print("VecNormalize stats loaded successfully.")
        else:
            env = vec_env
            print("WARNING: No VecNormalize stats found — using raw observations.")

        print(f"Loading PyTorch model: {args.model} on device: {args.device}")
        model_cls = PPO if args.algo == "ppo" else RecurrentPPO
        model = model_cls.load(args.model, device=args.device)
        print("Bot is ready and listening for game updates...", flush=True)

    try:
        obs = env.reset()

        while True:
            done = False

            state = raw_env.game_state
            start_chips = 0
            for p in state.get("players", []):
                if p["id"] == raw_env.player_id:
                    start_chips = p["chips"]
                    break

            lstm_states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)

            print(f"\n--- New Hand Started (Starting Stack: {start_chips}) ---")

            while not done:
                if use_onnx:
                    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                    if not isinstance(action, np.ndarray):
                        action = np.array([action])
                elif args.algo == "recurrent_ppo":
                    action, lstm_states = model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True
                    )
                else:
                    action, _ = model.predict(obs, deterministic=True)

                episode_starts = np.zeros((env.num_envs,), dtype=bool)
                action_name = PokerAction(int(action[0])).name

                state = raw_env.game_state
                my_hole = []
                my_chips = 0
                for p in state.get("players", []):
                    if p["id"] == raw_env.player_id:
                        my_hole = p.get("hole", [])
                        my_chips = p["chips"]
                        break
                print(f"My Cards: {my_hole} | Board: {state.get('board', [])} | Action: {action_name} | My Chips: {my_chips}")

                obs, rewards, dones, infos = env.step(action)
                done = bool(dones[0])

            end_state = infos[0].get("game_state", {})
            end_chips = 0
            for p in end_state.get("players", []):
                if p["id"] == raw_env.player_id:
                    end_chips = p["chips"]
                    break

            diff = end_chips - start_chips
            if diff > 0:
                print(f"🎉 Hand finished. WON {diff} chips! (Current Stack: {end_chips})")
            elif diff < 0:
                print(f"💸 Hand finished. LOST {abs(diff)} chips! (Current Stack: {end_chips})")
            else:
                print(f"🤝 Hand finished. Break-even (0 chips change). (Current Stack: {end_chips})")

    except KeyboardInterrupt:
        print("\nStopping bot and closing connections...")
    finally:
        env.close()

if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from adapters.simulator.sim_env import SimulatorEnv
from poker_env.action_space import PokerAction
from agent.onnx_agent import ONNXAgent


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO RL Agent in Simulator")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to the saved zip model (default: models/ppo_mlp_agent).")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--players", type=int, default=2,
                        help="Number of players at the table (default: 2).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to run inference on (default: cpu). Use 'cuda' or 'auto' if desired.")
    parser.add_argument("--onnx", action="store_true",
                        help="Run inference using ONNX Runtime instead of PyTorch.")
    parser.add_argument("--onnx-model", type=str, default=None,
                        help="Path to exported ONNX model (.onnx file).")
    args = parser.parse_args()

    if args.onnx:
        onnx_path = args.onnx_model or "models/ppo_mlp_agent.onnx"
        print(f"Loading ONNX model from: {onnx_path}")
        try:
            model = ONNXAgent(onnx_path)
            print("ONNX Runtime engine loaded successfully.")
        except Exception as e:
            print(f"Could not load ONNX model: {e}. Running with random actions.")
            model = None
        vec_env = DummyVecEnv([lambda: SimulatorEnv(num_players=args.players)])
        env = vec_env
    else:
        if args.model is None:
            args.model = "models/ppo_mlp_agent"

        vec_env = DummyVecEnv([lambda: SimulatorEnv(num_players=args.players)])

        vec_normalize_path = "models/vec_normalize.pkl"
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, vec_env)
            env.training = False
            env.norm_reward = False
            print("VecNormalize stats loaded.")
        else:
            env = vec_env
            print("No VecNormalize stats found — using raw observations.")

        try:
            model = PPO.load(args.model, device=args.device)
            print(f"Model loaded successfully on device: {args.device}.")
        except Exception as e:
            print(f"Could not load model: {e}. Running with random actions.")
            model = None

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        print(f"\n--- Episode {ep + 1} ---")
        while not done:
            steps += 1
            if model:
                if args.onnx:
                    action, _ = model.predict(obs, state=None, episode_start=None, deterministic=True)
                    if not isinstance(action, np.ndarray):
                        action = np.array([action])
                else:
                    action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.array([env.action_space.sample()])

            obs, rewards, dones, infos = env.step(action)
            reward = float(rewards[0])
            done = bool(dones[0])
            info = infos[0]
            total_reward += reward

            state = info.get("game_state", {})
            action_name = PokerAction(int(np.asarray(action).flat[0])).name
            pot = state.get("pot", 0)
            phase = state.get("phase", "Unknown")

            my_chips = 0
            my_hole = []
            for p in state.get("players", []):
                if p["id"] == "player_0":
                    my_chips = p["chips"]
                    my_hole = p.get("hole", [])
                    break

            print(
                f"Step {steps} | Phase: {phase} | Cards: {my_hole} | "
                f"Board: {state.get('board', [])} | Action: {action_name} | "
                f"Pot: {pot} | My Chips: {my_chips} | Reward: {reward:.4f}"
            )

        print(f"Finished Episode {ep + 1} | Total Steps: {steps} | "
              f"Cumulative Reward: {total_reward:.4f}")


if __name__ == "__main__":
    main()

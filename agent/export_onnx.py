import os
import argparse
import pickle
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO


class OnnxableMLPPolicy(nn.Module):
    def __init__(self, policy, obs_mean=None, obs_var=None, clip_obs=10.0, epsilon=1e-8):
        super().__init__()
        self.policy = policy
        if obs_mean is not None and obs_var is not None:
            self.use_norm = True
            self.register_buffer("obs_mean", torch.tensor(obs_mean, dtype=torch.float32))
            self.register_buffer("obs_var", torch.tensor(obs_var, dtype=torch.float32))
            self.clip_obs = float(clip_obs)
            self.epsilon = float(epsilon)
        else:
            self.use_norm = False

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            observation = torch.clamp(
                (observation - self.obs_mean) / torch.sqrt(self.obs_var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
        distribution = self.policy.get_distribution(observation)
        action = distribution.mode()
        return action


def export_onnx(model_path: str, output_path: str, vec_norm_path: str = None, algo: str = "ppo", obs_dim: int = 168):
    if not os.path.exists(model_path):
        if not model_path.endswith(".zip") and os.path.exists(model_path + ".zip"):
            model_path = model_path + ".zip"
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    model_cls = PPO if algo == "ppo" else RecurrentPPO
    print(f"Loading SB3 model ({algo.upper()}) from '{model_path}'...")
    model = model_cls.load(model_path, device="cpu")
    policy = model.policy.eval()

    obs_mean, obs_var = None, None
    if vec_norm_path and os.path.exists(vec_norm_path):
        try:
            with open(vec_norm_path, "rb") as f:
                vec_data = pickle.load(f)
                obs_mean = vec_data.obs_rms.mean
                obs_var = vec_data.obs_rms.var
                print(f"Loaded VecNormalize statistics from '{vec_norm_path}'.")
        except Exception as e:
            print(f"WARNING: Could not parse VecNormalize pickle: {e}")
    else:
        print("No VecNormalize path provided or file missing — exporting model without observation scaling.")

    onnx_module = OnnxableMLPPolicy(policy, obs_mean, obs_var).eval()

    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Exporting ONNX model to '{output_path}'...")
    torch.onnx.export(
        onnx_module,
        dummy_input,
        output_path,
        opset_version=14,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"ONNX model successfully exported: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Trained SB3 Model to ONNX Graph")
    parser.add_argument("--model", type=str, default="models/ppo_mlp_agent",
                        help="Path to trained SB3 model (.zip)")
    parser.add_argument("--vec-norm", type=str, default="models/vec_normalize.pkl",
                        help="Path to VecNormalize statistics (.pkl)")
    parser.add_argument("--output", type=str, default="models/ppo_mlp_agent.onnx",
                        help="Output path for exported ONNX model (.onnx)")
    parser.add_argument("--algo", choices=["ppo", "recurrent_ppo"], default="ppo",
                        help="Algorithm type (default: ppo)")
    parser.add_argument("--obs-dim", type=int, default=168,
                        help="Observation vector dimension (default: 168)")
    args = parser.parse_args()

    export_onnx(args.model, args.output, args.vec_norm, args.algo, args.obs_dim)


if __name__ == "__main__":
    main()

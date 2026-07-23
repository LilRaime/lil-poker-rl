"""Unit tests for ONNX model export and ONNXAgent inference parity."""
import os
import tempfile
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from adapters.simulator.sim_env import SimulatorEnv
from agent.export_onnx import export_onnx
from agent.onnx_agent import ONNXAgent


def test_onnx_export_and_inference_parity():
    """Test exporting PPO model to ONNX and verifying action output parity."""
    vec_env = DummyVecEnv([lambda: SimulatorEnv(num_players=2)])
    model = PPO("MlpPolicy", vec_env, n_steps=64, verbose=0)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test_ppo.zip")
        onnx_path = os.path.join(tmp_dir, "test_ppo.onnx")
        
        model.save(zip_path)
        assert os.path.exists(zip_path)
        
        export_onnx(zip_path, onnx_path, vec_norm_path=None, obs_dim=168)
        assert os.path.exists(onnx_path)
        
        agent = ONNXAgent(onnx_path)
        
        dummy_obs = np.random.randn(10, 168).astype(np.float32)
        
        for i in range(len(dummy_obs)):
            single_obs = dummy_obs[i]
            
            pt_action, _ = model.predict(single_obs, deterministic=True)
            
            onnx_action, _ = agent.predict(single_obs, deterministic=True)
            
            assert int(pt_action) == int(onnx_action), (
                f"Action mismatch at index {i}: PyTorch={pt_action}, ONNX={onnx_action}"
            )
            assert 0 <= int(onnx_action) <= 4, f"Invalid action value: {onnx_action}"


def test_onnx_batch_prediction():
    """Test batch observation prediction with ONNXAgent."""
    vec_env = DummyVecEnv([lambda: SimulatorEnv(num_players=2)])
    model = PPO("MlpPolicy", vec_env, n_steps=64, verbose=0)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "test_ppo_batch.zip")
        onnx_path = os.path.join(tmp_dir, "test_ppo_batch.onnx")
        
        model.save(zip_path)
        export_onnx(zip_path, onnx_path, vec_norm_path=None, obs_dim=168)
        
        agent = ONNXAgent(onnx_path)
        batch_obs = np.random.randn(5, 168).astype(np.float32)
        actions, _ = agent.predict(batch_obs, deterministic=True)
        
        assert isinstance(actions, np.ndarray)
        assert len(actions) == 5

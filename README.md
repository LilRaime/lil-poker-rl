# 🃏 Lil Poker RL Agent

**Python:** 3.10+ &nbsp;|&nbsp; **License:** [GPL v3](LICENSE) &nbsp;|&nbsp; **RL Library:** [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (PPO) &nbsp;|&nbsp; **Env:** [Gymnasium](https://gymnasium.farama.org/) 0.29+

A portable Reinforcement Learning agent for Texas Hold'em built using **Gymnasium** and **Stable-Baselines3 (PPO)**.
It trains offline inside a high-speed simulator and plays online on a live
**[lil-poker](https://github.com/LilRaime/lil-poker)** Go backend via WebSockets.

---

## 📂 Project Structure

- `poker_env/` - Core Gymnasium interface:
  - `base_env.py`: Abstract Gymnasium base environment.
  - `action_space.py`: Action mapping (Fold, Check/Call, Raise Min, Raise Pot, All-in).
  - `observation.py`: Converts raw game state dictionaries into a normalized float32 vector of shape `(168,)` (including hole cards, board cards, stacks, blinds, relative positions, hand strength, board texture, action history grids, and opponent profiles).
- `adapters/` - Integration adapters:
  - `simulator/`: Offline simulator utilizing the `treys` library for showdown evaluation.
  - `lil_poker/`: WebSocket + HTTP API client to connect the bot to the live Go backend.
- `agent/` - CLI scripts:
  - `train.py`: Trains the PPO model inside vectorized simulators.
  - `evaluate.py`: Evaluates the trained agent against simulator bots.
  - `play_live.py`: Runs the trained agent inside a live poker room (supports ONNX auto-detection).
  - `export_onnx.py`: Exports a trained SB3 `.zip` model to ONNX format.
  - `onnx_agent.py`: Lightweight ONNX Runtime inference wrapper (no PyTorch required).

---

## 🚀 Installation & Setup

1. **Create and activate a virtual environment:**
   - **Unix-like:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows PowerShell:**
     ```powershell
     python -m venv .venv_win
     .\.venv_win\Scripts\activate
     ```
     > **If you get "running scripts is disabled"** run this once (no admin needed):
     > ```powershell
     > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
     > ```
     > Or bypass without changing policy: `& .\.venv_win\Scripts\Activate.ps1`


2. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```
   > **Note:** For CPU-only training (recommended for small MLP networks — avoids PCIe bottleneck):
   > ```bash
   > pip install torch --index-url https://download.pytorch.org/whl/cpu
   > pip install -e .
   > ```


---

## 💻 Command Reference & Usage

### 1. Training the Agent (`agent.train`)
Train the PPO model in the offline simulator.
```bash
python -m agent.train [arguments]
```
- **Arguments:**
  - `--envs <int>` (Default: `4`): Number of parallel environments (simulated tables) to run. Set this to your CPU physical core count (e.g., 6 or 8) for maximum speed.
  - `--timesteps <int>` (Default: `250000`): Total timesteps (steps) to train. Recommended: `1000000` to `3000000` for a solid self-play policy.
  - `--lr <float>` (Default: `3e-4`): Learning rate for PPO.
  - `--self-play` (Action flag): Trains against frozen PPO snapshots from `models/opponent_pool/`. Snapshots are saved as `snapshot_Xk.zip` every `--opponent-update-interval` steps.
  - `--league` (Action flag): Enables full **League Training** mode — bots are drawn from RandomBot, RuleBot, and historical PPO snapshots, weighted by Elo rating. Implies `--self-play`.
  - `--opponent-update-interval <int>` (Default: `100000`): Steps between opponent snapshots.
  - `--eval-games <int>` (Default: `1000`): Games played for Elo evaluation after each snapshot.
  - `--no-elo` (Action flag): Skip Elo evaluation after each snapshot (faster training).
  - `--device <str>` (Default: `auto`): Hardware device to run training (`cpu` or `cuda`). Recommended `cpu` for small MLP architectures.
  - `--num-threads <int>` (Default: `1`): Number of CPU threads per environment. Setting to `1` is strongly recommended for parallel environments to avoid thread contention.
  - `--n-steps <int>` (Default: `2048`): Number of steps collected per environment before updating the policy.
  - `--batch-size <int>` (Default: `256`): Mini-batch size for PPO gradient updates.
  - `--n-epochs <int>` (Default: `10`): Number of epochs to optimize the PPO loss on each update.
- **Example (Standard training against heuristics):**
  ```bash
  python -m agent.train --envs 6 --timesteps 500000
  ```
- **Example (Self-play with Elo tracking):**
  ```bash
  python -m agent.train --envs 8 --timesteps 3000000 --self-play --device cpu --num-threads 1 --n-steps 1024 --batch-size 2048 --n-epochs 4
  ```
- **Example (Full League Training — recommended):**
  ```bash
  python -m agent.train --envs 8 --timesteps 3000000 --league --device cpu --num-threads 1 --n-steps 1024 --batch-size 2048 --n-epochs 4
  ```

### 2. Evaluating the Agent (`agent.evaluate`)
Test the trained agent against the rule-based simulator bots and see decisions step-by-step.
```bash
python -m agent.evaluate [arguments]
```
- **Arguments:**
  - `--model <path>` (Default: `models/ppo_mlp_agent`): Path to the saved zip model file.
  - `--episodes <int>` (Default: `10`): Number of episodes (hands) to play.
  - `--players <int>` (Default: `2`): Number of players at the table during evaluation.
  - `--device <str>` (Default: `cpu`): Device to run inference on (`cpu`, `cuda`, or `auto`).
- **Example:**
  ```bash
  python -m agent.evaluate --model models/ppo_mlp_agent --episodes 10 --device cpu
  ```

### 3. Playing Live on the Server (`agent.play_live`)
Run the trained agent to play inside a real lobby room on the `lil-poker` Go backend.
```bash
python -m agent.play_live --room <room_id> [arguments]
```
- **Required Arguments:**
  - `--room <str>`: The 6-character room ID (e.g. `HF4YL8`) created in the Web UI.
- **Optional Arguments:**
  - `--model <path>` (Default: `models/ppo_mlp_agent`): Path to the trained PPO model.
  - `--url <str>` (Default: `http://localhost:8090`): Base URL of the API.
  - `--name <str>` (Default: `PPO_Bot`): Nickname of the bot (max 12 characters).
  - `--device <str>` (Default: `cpu`): Device to run inference on (`cpu`, `cuda`, or `auto`).
- **Example:**
  ```bash
  python -m agent.play_live --room HF4YL8 --device cpu
  ```

### 4. Exporting to ONNX & Running ONNX Runtime Inference (`agent.export_onnx`)
Export PyTorch PPO models to lightweight ONNX format with embedded observation normalization for ultra-fast, zero-PyTorch inference.
```bash
# Export trained PyTorch model to ONNX format
python -m agent.export_onnx --model models/ppo_mlp_agent.zip --vec-norm models/vec_normalize.pkl --output models/ppo_mlp_agent.onnx

# Evaluate using ONNX Runtime
python -m agent.evaluate --onnx --onnx-model models/ppo_mlp_agent.onnx --episodes 10

# Play live using ONNX Runtime
python -m agent.play_live --room HF4YL8 --onnx --onnx-model models/ppo_mlp_agent.onnx
```

---

## ⚡ Performance Optimizations

This repository implements advanced performance optimizations to avoid typical Python/RL overheads:
- **Fast Hand Strength Evaluator:** The real-time state encoder uses `phevaluator` for O(1) mathematical lookup of 5 to 7 card ranks, eliminating costly list manipulations.
- **Card Object Caching:** For the simulator, all `treys.Card` objects for 52 cards are pre-computed at module initialization. This completely eliminates string parsing (`Card.new`) in the hot loop.
- **Single Lookup Table:** The `treys.Evaluator` is a global module singleton, preventing the expensive reconstruction of the 7,462-hand mathematical lookup table on every step.
- **Deck Copying:** The deck array is cloned using `copy()` instead of list comprehensions on resets.
- These modifications boost the offline training speed from **160 FPS to over 2800+ FPS** on standard CPU hardware.

---

## 🐳 Docker Deployment

### 1. Main Production ONNX Bot Container (`Dockerfile` - ~360MB)
This is the default lightweight image used by the `lil-poker` Go backend when clicking **"Add Bot"** in the Web UI.
The trained `models/ppo_mlp_agent.onnx` model is **baked into the image** — no volume mount needed at runtime.
```bash
# 1. Export your trained PyTorch model to ONNX (run once on host machine)
python -m agent.export_onnx --model models/ppo_mlp_agent.zip --output models/ppo_mlp_agent.onnx

# 2. Build production image (tagged 'lil-poker-rl')
docker build -t lil-poker-rl .

# 3. (Optional) Run bot manually connecting to a live room
#    Note: no ENTRYPOINT — full command required
docker run --net=host lil-poker-rl python -m agent.play_live --room HF4YL8 --url http://localhost:8090
```

### 2. Training Container (`Dockerfile.train` - PyTorch + SB3)
For training models inside Docker:
```bash
# Build training image
docker build -f Dockerfile.train -t lil-poker-rl-train .

# Run training and mount local models directory
docker run -v $(pwd)/models:/app/models lil-poker-rl-train
```

---

## 🧪 Running Tests

The test suite uses **pytest** and covers environment correctness, chip conservation, and reward finiteness.

```bash
# Install dev dependencies first (if not already installed)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**.
See the [LICENSE](LICENSE).
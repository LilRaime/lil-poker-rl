FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir \
    "gymnasium>=0.29.0" \
    "stable-baselines3[extra]>=2.2.1" \
    "sb3-contrib>=2.2.1" \
    "numpy>=1.24.0" \
    "requests>=2.31.0" \
    "websocket-client>=1.6.0" \
    "treys>=0.1.8" \
    "phevaluator>=0.5.0"

COPY . .

RUN pip install --no-cache-dir --no-deps .

CMD ["python", "-m", "agent.train", "--envs", "8", "--timesteps", "3000000", "--self-play", "--device", "cpu", "--num-threads", "1", "--n-steps", "1024", "--batch-size", "2048", "--n-epochs", "4"]


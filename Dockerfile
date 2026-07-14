FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    binutils \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

RUN pip install --no-cache-dir --no-compile \
    "torch==2.13.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple

RUN pip install --no-cache-dir --no-compile \
    "gymnasium>=0.29.0" \
    "stable-baselines3>=2.2.1" \
    "sb3-contrib>=2.2.1" \
    "numpy>=1.24.0" \
    "requests>=2.31.0" \
    "websocket-client>=1.6.0" \
    "treys>=0.1.8" \
    "phevaluator>=0.5.0"

COPY . .
RUN pip install --no-cache-dir --no-deps --no-compile .

RUN set -eux; \
    find /usr/local/lib/python3.10/site-packages -name "*.pyc" -delete; \
    find /usr/local/lib/python3.10/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true; \
    find /usr/local/lib/python3.10/site-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true; \
    find /usr/local/lib/python3.10/site-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true; \
    rm -rf /usr/local/lib/python3.10/site-packages/torch/test; \
    rm -rf /usr/local/lib/python3.10/site-packages/torch/include; \
    rm -rf /usr/local/lib/python3.10/site-packages/torch/share; \
    rm -rf /usr/local/lib/python3.10/site-packages/torch/csrc; \
    rm -rf /usr/local/lib/python3.10/site-packages/torch/utils/data/datasets_generation; \
    find /usr/local/lib/python3.10/site-packages -name "*.a" -delete; \
    find /usr/local/lib/python3.10/site-packages -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true; \
    find /usr/local/lib/python3.10/site-packages -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null || true

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

CMD ["python", "-m", "agent.train", "--envs", "8", "--timesteps", "3000000", "--self-play", "--device", "cpu", "--num-threads", "1", "--n-steps", "1024", "--batch-size", "2048", "--n-epochs", "4"]
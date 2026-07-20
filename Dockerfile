FROM python:3.10-slim AS builder
WORKDIR /app

RUN set -e; \
    pip install --no-cache-dir --no-compile \
    "numpy>=1.24.0" \
    "onnxruntime>=1.16.0" \
    "requests>=2.31.0" \
    "websocket-client>=1.6.0" \
    "phevaluator>=0.5.0" \
    "treys>=0.1.8" \
    "gymnasium>=0.29.0"; \
    SP=/usr/local/lib/python3.10/site-packages; \
    find "$SP" -name "*.pyc" -delete; \
    find "$SP" -type d -name "__pycache__" -exec rm -rf {} +; \
    find "$SP" -type d -iname "test*" -exec rm -rf {} +; \
    find "$SP" -name "*.dist-info" -exec sh -c 'rm -f "$1"/RECORD "$1"/*.txt "$1"/direct_url.json' _ {} \; ; \
    rm -rf "$SP/pip" "$SP/setuptools" "$SP/wheel"; \
    rm -rf "$SP/sympy" "$SP/mpmath"; \
    rm -rf "$SP/onnxruntime/transformers" "$SP/onnxruntime/quantization" \
    "$SP/onnxruntime/training" "$SP/onnxruntime/tools" "$SP/onnxruntime/datasets"

FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

RUN rm -rf /usr/local/lib/python3.10/site-packages/pip \
    /usr/local/lib/python3.10/site-packages/setuptools \
    /usr/local/lib/python3.10/site-packages/wheel \
    && find /usr/local/lib/python3.10/site-packages -maxdepth 1 -name "*.dist-info" \
    \( -iname "*pip*" -o -iname "*setuptools*" -o -iname "*wheel*" \) \
    -exec rm -rf {} +

COPY poker_env/   poker_env/
COPY adapters/    adapters/
COPY agent/onnx_agent.py  agent/onnx_agent.py
COPY agent/play_live.py   agent/play_live.py
COPY agent/__init__.py    agent/__init__.py
COPY pyproject.toml .
COPY models/ppo_mlp_agent.onnx models/ppo_mlp_agent.onnx
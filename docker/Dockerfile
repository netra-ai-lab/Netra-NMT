# syntax=docker/dockerfile:1.7
#
# Netra NMT — English <-> Khmer translation web app + REST API, model baked in (runs offline).
#
#   docker build -f docker/Dockerfile -t netra-nmt .
#   docker run -p 8000:8000 netra-nmt
#   then open http://localhost:8000
#
# CPU only, and no PyTorch: the model runs on ONNX Runtime (netra_nmt/inference).

ARG PYTHON_VERSION=3.12

# ── base: shared by the build and runtime stages ─────────────────────────────
# No system packages unless a default feature needs one: keep the image small.
FROM python:${PYTHON_VERSION}-slim AS base

# ── build: virtualenv with netra-nmt[server] on ONNX Runtime (CPU), pruned ──
FROM base AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY netra_nmt ./netra_nmt
RUN pip install ".[server,cpu]"

# Prune what the app never needs (checked by docker/smoke_test.py below):
# bytecode caches and pip.
RUN find /opt/venv -depth -name __pycache__ -type d -exec rm -rf {} + && \
    pip uninstall -y pip

# ── weights: the model bundles, fetched once at build time ──────────────────
# Then run the whole pipeline once against the pruned environment, offline:
# a prune that removed something needed fails here, not on the first upload.
FROM build AS weights
COPY docker/fetch_weights.py docker/smoke_test.py /tmp/
RUN HF_HOME=/tmp/hf python /tmp/fetch_weights.py /models/netra-nmt && rm -rf /tmp/hf && \
    cd /tmp && NETRA_NMT_BUNDLES=/models/netra-nmt HF_HUB_OFFLINE=1 python /tmp/smoke_test.py

# ── runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime
LABEL org.opencontainers.image.title="Netra NMT" \
      org.opencontainers.image.description="Compact English <-> Khmer translation (77M parameters) on ONNX Runtime: web app + REST API, offline" \
      org.opencontainers.image.source="https://github.com/netra-ai-lab/Netra-NMT" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

RUN useradd --create-home --uid 10001 netra
COPY --from=build /opt/venv /opt/venv
COPY --from=weights /models /models

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NETRA_NMT_BUNDLES=/models/netra-nmt \
    HF_HUB_OFFLINE=1 \
    NETRA_NMT_HOST=0.0.0.0 \
    NETRA_NMT_PORT=8000 \
    NETRA_NMT_DEVICE=cpu

USER netra
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('NETRA_NMT_PORT','8000'), timeout=4)"
CMD ["netra-nmt", "serve"]

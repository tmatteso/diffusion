# PyTorch Dev Container
# Use PLATFORM build arg to select between arm64 and cuda

ARG PLATFORM=arm64

# For CUDA: Use PyTorch CUDA image
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel AS base-cuda

# For ARM64: Use Ubuntu with CPU PyTorch
FROM ubuntu:22.04 AS base-arm64

# Select base based on PLATFORM arg — this makes the devcontainer CLI pick the right stage
ARG PLATFORM
FROM base-${PLATFORM} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install common tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    openssh-client \
    curl \
    ca-certificates \
    build-essential \
    vim \
    wget \
    axel \
    pigz \
    pv \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Create platform info script
RUN printf '#!/bin/bash\necho "=== Platform Info ==="\necho "Arch: $(uname -m)"\npython3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")"' > /usr/local/bin/show-platform && \
    chmod +x /usr/local/bin/show-platform

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# Re-declare after FROM so it's in scope for the rest of this stage
ARG PLATFORM

# For CUDA: inherit conda's PyTorch into the venv to avoid re-downloading it
RUN if [ "${PLATFORM}" = "cuda" ]; then \
        uv venv /opt/venv --system-site-packages --python /opt/conda/bin/python3; \
    else \
        uv venv /opt/venv; \
    fi

# Install project dependencies into the image — cached as a layer, only re-runs when lockfile changes
COPY pyproject.toml uv.lock ./
# On CUDA, torch/torchvision/torchaudio are already in the base image's conda
# environment (visible via --system-site-packages), so skip re-downloading them.
# Since torch is skipped here, uv never generates a torchrun console-script in
# the venv, so regenerate it by hand, bound to the venv's own interpreter, so
# torchrun's spawned workers get torch (inherited from conda) AND the venv's
# own uv-installed deps on the same interpreter — instead of
# /opt/conda/bin/torchrun's workers, which only see conda's site-packages.
RUN if [ "${PLATFORM}" = "cuda" ]; then \
        uv sync --frozen \
            --no-install-package torch \
            --no-install-package torchvision \
            --no-install-package torchaudio && \
        printf '#!/opt/venv/bin/python3\nimport sys\nfrom torch.distributed.run import main\nif __name__ == "__main__":\n    sys.exit(main())\n' \
            > /opt/venv/bin/torchrun && \
        chmod +x /opt/venv/bin/torchrun; \
    else \
        uv sync --frozen; \
    fi

# Install hadolint
RUN if [ "${PLATFORM}" = "arm64" ]; then \
        wget -qO /usr/local/bin/hadolint \
            https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-arm64; \
    else \
        wget -qO /usr/local/bin/hadolint \
            https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64; \
    fi \
    && chmod +x /usr/local/bin/hadolint

WORKDIR /workspaces/diffusion


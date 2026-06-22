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

# Install PyTorch for ARM64 (CUDA image already has it from base image)
ARG PLATFORM
RUN if [ "${PLATFORM}" = "arm64" ]; then \
        pip3 install --no-cache-dir \
            torch==2.7.0 \
            torchvision==0.22.0 \
            torchaudio==2.7.0 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Create platform info script
RUN printf '#!/bin/bash\necho "=== Platform Info ==="\necho "Arch: $(uname -m)"\npython3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")"' > /usr/local/bin/show-platform && \
    chmod +x /usr/local/bin/show-platform

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

# For CUDA: inherit conda's PyTorch into the venv to avoid re-downloading it
ARG PLATFORM
RUN if [ "${PLATFORM}" = "cuda" ]; then \
        uv venv /opt/venv --system-site-packages --python /opt/conda/bin/python3; \
    else \
        uv venv /opt/venv; \
    fi

# Install hadolint
ARG PLATFORM
RUN if [ "${PLATFORM}" = "arm64" ]; then \
        wget -qO /usr/local/bin/hadolint \
            https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-arm64; \
    else \
        wget -qO /usr/local/bin/hadolint \
            https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64; \
    fi \
    && chmod +x /usr/local/bin/hadolint

WORKDIR /workspaces/diffusion

# Install project dependencies into the image — cached as a layer, only re-runs when lockfile changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# PyTorch Dev Container
# Use PLATFORM build arg to select between arm64 and cuda

ARG PLATFORM=arm64

# For CUDA: Use PyTorch CUDA image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel AS base-cuda

# For ARM64: Use Ubuntu with CPU PyTorch
FROM ubuntu:22.04 AS base-arm64

# Select based on PLATFORM build argument
FROM base-${PLATFORM}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install common tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    curl \
    ca-certificates \
    build-essential \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_SYSTEM_PYTHON=1

# Install PyTorch for ARM64 (CUDA image already has it from base image)
ARG PLATFORM
RUN if [ "${PLATFORM}" = "arm64" ]; then \
        pip3 install --no-cache-dir \
            torch==2.5.1 \
            torchvision==0.20.1 \
            torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Create platform info script (runs via postCreateCommand)
RUN printf '#!/bin/bash\necho "=== Platform Info ==="\necho "Arch: $(uname -m)"\npython3 -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")"' > /usr/local/bin/show-platform && \
    chmod +x /usr/local/bin/show-platform

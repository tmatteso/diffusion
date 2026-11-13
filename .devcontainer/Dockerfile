# PyTorch Dev Container
# Use PLATFORM build arg to select between arm64 and cuda

ARG PLATFORM=arm64

# For CUDA: Use PyTorch CUDA image
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel AS base-cuda

# For ARM64: Use Ubuntu with CPU PyTorch
FROM ubuntu:24.04 AS base-arm64

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

# Install GitHub CLI from its official apt repository
RUN mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO /etc/apt/keyrings/githubcli-archive-keyring.gpg \
        https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22 from the NodeSource apt repository, for the web/ frontend.
# apt reads the ASCII-armored key directly (.asc), so gnupg isn't needed.
# Both base images get Node from here, so the arm64 and cuda containers can't
# drift apart.
RUN wget -qO /etc/apt/keyrings/nodesource.asc \
        https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    && chmod go+r /etc/apt/keyrings/nodesource.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/nodesource.asc] https://deb.nodesource.com/node_22.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Activate pnpm through corepack, which ships with Node 22. pnpm fetches its
# native binary into corepack's cache on first run, so run it once here: the
# image then carries it, and the first `pnpm` call in the container needs no
# network access and shows no download prompt. web/package.json's
# `packageManager` field pins the version a project uses; keep the two in step.
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable pnpm \
    && corepack install --global pnpm@12.16.1 \
    && pnpm --version

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

# For CUDA: inherit the base image's own PyTorch into the venv (its interpreter
# must match the one torch is installed against, so use the system python3).
# For arm64/CPU: no such constraint, so let uv manage its own Python 3.12.3
# build directly instead of depending on whatever the base OS's apt-installed
# python3 happens to be.
RUN if [ "${PLATFORM}" = "cuda" ]; then \
        uv venv /opt/venv --system-site-packages --python "$(command -v python3)"; \
    else \
        uv python install 3.12.3 && \
        uv venv /opt/venv --python 3.12.3; \
    fi

# Install project dependencies into the image — cached as a layer, only re-runs when lockfile changes
COPY pyproject.toml uv.lock ./
# On CUDA, torch/torchvision/torchaudio are already in the base image's conda
# environment (visible via --system-site-packages). pyproject.toml's markers
# leave them out of the lock on x86_64 Linux, so uv never installs them here.
# Since uv does not install torch, it never generates a torchrun console-script in
# the venv, so regenerate it by hand, bound to the venv's own interpreter, so
# torchrun's spawned workers get torch (inherited from conda) AND the venv's
# own uv-installed deps on the same interpreter — instead of
# /opt/conda/bin/torchrun's workers, which only see conda's site-packages.
RUN if [ "${PLATFORM}" = "cuda" ]; then \
        uv sync --frozen && \
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


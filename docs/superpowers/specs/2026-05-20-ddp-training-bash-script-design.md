# DDP Training Bash Script Design

**Date:** 2026-05-20

## Goal

Create a bash script that launches a `torchrun` DDP training run for PallAtom and keeps running after SSH disconnects.

## Location

`/workspaces/diffusion/bash_scripts/run_ddp_train.sh`

## Requirements

- Hardcoded paths to data, splits, config, and log file under `pallatom/`
- GPU count detected dynamically via `nvidia-smi`
- Process survives SSH disconnect via `nohup`
- Stdout redirected to `bash_scripts/train_stdout.log` alongside the script

## Design

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PALLATOM_ROOT=/workspaces/diffusion/pallatom
TORCHRUN=/opt/venv/bin/torchrun
STDOUT_LOG="$SCRIPT_DIR/train_stdout.log"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPU(s). Launching DDP training under nohup..."

nohup "$TORCHRUN" \
    --nproc_per_node="$NUM_GPUS" \
    "$PALLATOM_ROOT/train/train_loop.py" \
    --data     "$PALLATOM_ROOT/data/chain_set.jsonl" \
    --splits   "$PALLATOM_ROOT/data/chain_set_splits.json" \
    --config   "$PALLATOM_ROOT/train/run_config.json" \
    --log_file "$PALLATOM_ROOT/train/train_logs_ddp.jsonl" \
    --num_workers 0 \
    > "$STDOUT_LOG" 2>&1 &

echo "Training PID: $!"
echo "Stdout → $STDOUT_LOG"
echo "Structured log → $PALLATOM_ROOT/train/train_logs_ddp.jsonl"
```

## Key Decisions

- `nohup ... &` backgrounds torchrun and makes it immune to SIGHUP on SSH disconnect
- `nvidia-smi --query-gpu=name --format=csv,noheader | wc -l` for GPU detection — fails loudly if CUDA unavailable
- `STDOUT_LOG` placed in `bash_scripts/` next to the script for easy discovery
- Script exits immediately after launch; monitoring via `tail -f` on either log file
- `--num_workers 0` matches the original Python DDP launch configuration

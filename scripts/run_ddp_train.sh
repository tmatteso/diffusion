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
    --dataset_jsonl         "$PALLATOM_ROOT/data/chain_set.jsonl" \
    --keys_for_splits_json  "$PALLATOM_ROOT/data/chain_set_splits.json" \
    --config                "$PALLATOM_ROOT/train/run_config.json" \
    --structlog_jsonl       "$PALLATOM_ROOT/train/train_logs_ddp.jsonl" \
    --shard_dir             "$PALLATOM_ROOT/data/shards" \
    --ddp \
    > "$STDOUT_LOG" 2>&1 &

echo "Training PID: $!"
echo "Stdout → $STDOUT_LOG"
echo "Structured log → $PALLATOM_ROOT/train/train_logs_ddp.jsonl"

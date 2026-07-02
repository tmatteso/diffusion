#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PALLATOM_ROOT=/workspaces/diffusion/pallatom
TORCHRUN=/opt/venv/bin/torchrun
STDOUT_LOG="$SCRIPT_DIR/train_stdout.log"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPU(s). Launching DDP training under nohup..."

# train_loop.py is run by file path, so Python only puts its own directory
# (pallatom/train/) on sys.path. Its first-party imports (architecture,
# helpers, ...) resolve relative to pallatom/ as the root, so that needs to
# be on PYTHONPATH explicitly.
export PYTHONPATH="$PALLATOM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# GPU P2P support depends on host/VM topology, which can silently change
# across a reboot (e.g. a cloud box coming back up with a different
# underlying GPU/PCIe assignment). When P2P is unsupported, NCCL's
# SHM/GPUDirect fallback can itself crash with an illegal memory access
# instead of cleanly falling back further to sockets. So probe the fastest
# transport with a trivial all_reduce before committing to it for the real
# run; if the probe hangs or crashes, force NCCL over plain sockets instead.
PREFLIGHT_SCRIPT="$(mktemp --suffix=.py)"
cat > "$PREFLIGHT_SCRIPT" <<'EOF'
import os

import torch
import torch.distributed as dist

dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
tensor = torch.ones(1, device=f"cuda:{local_rank}")
dist.all_reduce(tensor)
dist.destroy_process_group()
EOF

if timeout 30 "$TORCHRUN" --nproc_per_node="$NUM_GPUS" "$PREFLIGHT_SCRIPT" \
    > /dev/null 2>&1; then
    echo "NCCL preflight OK, using the default (fastest available) transport."
else
    echo "NCCL preflight failed; falling back to NCCL over sockets."
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
fi
rm -f "$PREFLIGHT_SCRIPT"

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

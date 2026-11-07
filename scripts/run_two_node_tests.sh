#!/usr/bin/env bash
# Run the two-node hybrid-sharding tests. Start it on both nodes at once:
#
#   node A: scripts/run_two_node_tests.sh 0 <node-A-address>
#   node B: scripts/run_two_node_tests.sh 1 <node-A-address>
#
# Both nodes need this checkout at the same commit. Each node must expose
# exactly two GPUs. Optional environment:
#   PALLATOM_DATASET_JSONL     dataset to compare across nodes
#   PALLATOM_TWO_NODE_REPORT   where rank 0 writes its measurements
#   NCCL_SOCKET_IFNAME         network interface, if NCCL picks the wrong one
set -euo pipefail

NODE_RANK="${1:?usage: run_two_node_tests.sh NODE_RANK MASTER_ADDR [PORT]}"
MASTER_ADDR="${2:?usage: run_two_node_tests.sh NODE_RANK MASTER_ADDR [PORT]}"
MASTER_PORT="${3:-29500}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIME_LIMIT="${PALLATOM_TWO_NODE_TIME_LIMIT:-3600}"

export PALLATOM_TWO_NODE=1
export PALLATOM_TWO_NODE_REPORT="${PALLATOM_TWO_NODE_REPORT:-${TMPDIR:-/tmp}/pallatom_two_node_report.json}"

cd "$REPO_ROOT"
timeout "$TIME_LIMIT" torchrun \
    --nnodes=2 \
    --node_rank="$NODE_RANK" \
    --nproc_per_node=2 \
    --rdzv-backend=static \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --module pytest pallatom/tests/fsdp/test_two_node.py \
    -p no:cacheprovider --tb=short

if [[ "$NODE_RANK" == "0" ]]; then
    echo "Measurements written to $PALLATOM_TWO_NODE_REPORT"
fi

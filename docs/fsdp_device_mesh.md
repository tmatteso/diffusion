# FSDP on a 2×2 device mesh: design notes

**Status:** draft for discussion. Stacked on the DDP lockstep fix (#5), which is
assumed correct. No training code changes yet.

**Goal:** train across 2 nodes with 2 GPUs each, using FSDP within a node and
DDP-style replication across nodes.

## Summary

- **The topology works.** PyTorch's `fully_shard` supports it directly with a
  2D mesh (hybrid sharding, HSDP). Wrapping FSDP in `DDP` does not work: it
  fails at construction.
- **It trains correctly.** On an emulated 2×2 mesh, HSDP matched DDP on weights,
  gradient norms, EMA evaluation and checkpoint resume.
- **Four code changes are required.** Three of the four failures are silent or
  would only show up later (see [Required changes](#required-changes)).
- **It does not pay off for today's model.** FSDP can only shard
  parameter-shaped state, which is about 4% of peak GPU memory here. On two real
  GPUs it saved 5.6% of peak memory and doubled step time.

**Recommendation:** ship multi-node DDP first, since it needs only launcher
changes. Land the FSDP preparation only if the model is expected to grow
substantially (see [Open questions](#open-questions)).

## Why FSDP saves so little here

FSDP shards parameters, gradients, optimizer state and (with the fix below) EMA.
It does not shard activations: each GPU still runs the full model on its own
batch.

Measured with the default config, real proteins, 128 residues, batch size 8, on
one GPU:

| Quantity | Size |
|---|---|
| Parameters (23.1M, float32) | 88 MiB |
| Params + grads + Adam (2×) + EMA | about 440 MiB |
| Peak allocated during a training step | 10.8 GiB |

So about 96% of peak memory is activations, dominated by the N²-sized pair
representation. Sharding across two GPUs can free at most half of the 440 MiB.

Measured on the 2 physical GPUs (4 steps, default model, 128 residues, batch
size 3, same batches for both runs, first step excluded):

| | DDP | FSDP (1×2 mesh) | Change |
|---|---|---|---|
| Peak allocated per GPU | 4399 MiB | 4151 MiB | −5.6% |
| Step time | about 2.1 s | about 4.2 s | about 2× |
| First step (includes FSDP lazy init) | 4.8 s | 23.1 s | |

The step-time cost is specific to this machine and may differ on a cluster. I
did not break it down, so I can't say how much is all-gather versus DTensor
overhead.

**Break-even estimate:** for 2-way sharding to free 25% of peak memory with
activations unchanged, parameters would need to reach roughly 2 GiB, about 24×
today's size. That assumes activations stay fixed, which they would not if the
model grew by widening `c_pair`.

## Proposed design

```python
mesh = init_device_mesh(
    "cuda",
    (n_nodes, gpus_per_node),
    mesh_dim_names=("replicate", "shard"),
)
for block in (
    *model.node_updates,
    *model.atom_decoders,
    *model.pair_updates,
    model.template_embedder,
    model.atom_encoder,
):
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)  # root last; remaining params form one group
```

- **Placement:** parameters become `(Replicate(), Shard(0))`. On the emulated
  run, shard groups were `[0, 1]` and replicate groups `[0, 2]`, so sharding
  stayed within a node.
- **Mesh order depends on rank order.** `torchrun` numbers ranks contiguously per
  node, but a wrong order would silently shard across the network. Assert at
  startup that every shard group is on one host.
- **Data is unchanged.** There are still 4 data-parallel ranks, so
  `DistributedSampler` and `tp.accumulated_token_budget // world_size` already
  do the right thing.
- **No wrapper.** `fully_shard` changes the module's class in place, so
  state-dict keys stay unprefixed and `.module` unwrapping is not needed.
- **Sharding happens in `main()`, not `train()`.** It replaces the parameter
  objects, so `Adam` and `EMA` must be built afterwards.

### Dependency on #5

Under DDP, `no_sync` means a window has one collective however many micro-batches
it holds. Under FSDP every forward and backward has collectives, so ranks must
run the same number of micro-batches or they hang. The `any_rank_true` flush vote
from #5 guarantees this for training.

Two new constraints follow:

- `evaluate()` gains collectives for the first time, so unequal validation
  batch counts would hang. `drop_last=True` covers this today, but
  `require_equal_batch_counts` should also run on the validation loader.
- Any model call made on rank 0 only will hang.

## Required changes

Each was hit or confirmed on the emulated mesh.

| # | Problem | What happens | Fix |
|---|---|---|---|
| 1 | Typed `__call__` overrides call `self.forward()` directly | FSDP hooks never run; crashes with mixed `Tensor`/`DTensor` | `super().__call__()` in the 6 sharded classes |
| 2 | `EMA.swap` uses `named_parameters()` and `.data` | Crash on the next forward, or a corrupted root group | Swap via `state_dict()`, in place |
| 3 | Nothing seeds model init | **Silent:** replicas never converge | Seed identically before building the model |
| 4 | `save_checkpoint` saves DTensors on rank 0 only | **Silent:** writes a partial checkpoint | Gather collectively, then save on rank 0 |

**1. `__call__` overrides.** 23 modules use this pattern. Only the 6 passed to
`fully_shard` strictly need changing: `MainTrunk`, `NodeUpdate`,
`AtomAttentionDecoder`, `PairUpdate`, `TemplateEmbedder`, `AtomFeatureEncoder`.
The change also makes module hooks work in general, and is a no-op for DDP.

**2. EMA swap.** `reshard_after_forward` defaults to false for the root group, so
after a `no_grad` forward the root's parameters are left unsharded and
`named_parameters()` returns plain tensors. `state_dict()` reshards first, which
makes it a safe view. The EMA must also be built after sharding so that its
shadow weights are shards too. The swap costs one temporary per parameter:

```python
for name, tensor in model.state_dict().items():
    shadow = self.shadow[name]
    if isinstance(tensor, DTensor):
        live, saved = tensor.to_local(), shadow.to_local()
        scratch = live.clone()
        live.copy_(saved)
        saved.copy_(scratch)
```

**3. Seeding.** `DDP.__init__` broadcasts weights and buffers from rank 0, which
hides that `main()` never seeds. FSDP has no such broadcast. In the unseeded run
all four ranks had different local shards and `freqs`/`phases` buffers after two
optimizer steps, with no error raised. Gradients are averaged, so the replicas
can never reconverge.

**4. Checkpoints.** Today's `save_checkpoint` succeeded and wrote a 77 MiB file of
one rank's DTensor shards, which then failed to load into a plain `MainTrunk`.
`full_tensor()` is collective, so the `rank != 0` early return must move below
the gather.

I recommend gathering to a single file rather than adopting
`torch.distributed.checkpoint`. State is small, `sampling.py` keeps reading
`ckpt["ema"]` unchanged, and checkpoints stay independent of the mesh. On load,
use `distribute_tensor` for Adam state.

### Gradient accumulation

`DDPNoSync` is a silent no-op under FSDP because `no_sync` doesn't exist there.
Gradients are still correct, but they sync on every micro-batch. The equivalent
is `model.set_requires_gradient_sync(is_last)`, which also matched DDP. It
holds unsharded gradients during the window, which gives back some of the memory
saving. That cost is not yet measured, so this choice is open.

## Evidence

Each 4-rank run used the repo's real `optimizer_step`, `take_step`, `EMA` and
activation checkpointing, with 2 accumulation windows of 2 micro-batches each.
`grad_clip=1.0` kept clipping active, and dropout was off for determinism.

| Check | DDP | HSDP |
|---|---|---|
| Sum of squared weights, step 1 | 44853.6334 | 44853.6334 |
| Sum of squared weights, step 2 | 44450.9836 | 44450.9836 |
| Eval loss, raw weights | 8.07397 | 8.07397 |
| Eval loss, EMA weights | 7.92995 | 7.92995 |
| Component gradient norms | | within 3.3e-7 |

Resume test: gathering a checkpoint, rebuilding on the mesh and taking one step
gave 44052.44313, against 44052.44315 for continuing without a restart.

**Limits of this evidence:**

- There was no real second node. Two `torchrun` instances shared one machine, so
  4 ranks shared 2 GPUs and "cross-node" traffic used loopback sockets. Only
  correctness was tested this way; no multi-node timing exists.
- The runs were tiny: `K_unit=3`, 64 residues, batch size 2, 2–3 steps, torch
  2.11.0.
- Dropout was off, so train-mode behaviour under FSDP is untested.
- Failures of a real network or node were not tested.

To reproduce the emulation, run these two commands at once with the same port,
and set a script to use `init_device_mesh("cuda", (2, 2), ...)`:

```bash
NCCL_HOSTID=fakenode0 NCCL_SOCKET_IFNAME=lo torchrun --nnodes=2 --node_rank=0 \
  --nproc_per_node=2 --rdzv-backend=static --master_addr=127.0.0.1 \
  --master_port=29500 script.py
NCCL_HOSTID=fakenode1 NCCL_SOCKET_IFNAME=lo torchrun --nnodes=2 --node_rank=1 \
  --nproc_per_node=2 --rdzv-backend=static --master_addr=127.0.0.1 \
  --master_port=29500 script.py
```

`NCCL_HOSTID` is needed because NCCL rejects two ranks on one GPU within a host.

## Plan

1. **Multi-node DDP (worth doing regardless).** Extend
   `scripts/run_ddp_train.sh` with `--nnodes`, `--node_rank` and
   `--master_addr`, make its preflight cross nodes, and document that dataset
   and checkpoint paths must exist on both nodes. Python changes should not be
   needed, because `DistProcessGroup` already uses `LOCAL_RANK`.
2. **FSDP preparation (no-ops for DDP).** Changes 1 and 3, plus the gather-based
   checkpoint code path.
3. **HSDP behind a flag.** Add a `--mesh NODESxGPUS` option, the EMA swap, the
   host assertion, `model: MainTrunk | DDP` union updates, and
   `require_equal_batch_counts` for validation.
4. **Tests.** Port the parity checks above to `pallatom/tests/train`, covering
   DDP vs HSDP weights, EMA evaluation and checkpoint round trip, then run on
   real hardware.

## Open questions

1. **What is the goal?** If it's more throughput or longer proteins, FSDP won't
   help. If a much larger parameter count is planned, it might.
2. **Are 2 nodes available to test on?** Network speed decides whether the
   cross-node savings matter at all.
3. **Would cutting activation memory serve better?** `node_updates`,
   `template_embedder` and `atom_encoder` aren't activation-checkpointed today,
   and mixed precision isn't used. Neither has been measured, but both target
   the 96%.

## Existing issues found along the way

These are unrelated to FSDP and are not fixed here.

- `scripts/run_ddp_train.sh` line 43: `timeout 30 NCCL_P2P_DISABLE=1 "$TORCHRUN"`
  can never run, because `timeout` treats the assignment as the command. The SHM
  tier is always skipped. It needs `timeout 30 env NCCL_P2P_DISABLE=1 ...`.
- `EMA` is built before `DDP` broadcasts weights, so with no seed each rank's
  shadow starts from different random weights. The effect fades through the EMA
  ramp, and the seeding in change 3 removes it.
- `TrainingParams.resume_checkpoint` is documented but never read. `main()` gates
  on `pretrained_weights`, then `load_checkpoint` reads
  `checkpoint.checkpoint_path`. This needs sorting out before the resume path
  changes.

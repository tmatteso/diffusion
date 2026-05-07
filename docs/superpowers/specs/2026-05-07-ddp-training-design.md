# DDP Training Design

**Date:** 2026-05-07  
**Scope:** Add `train_ddp()` and `evaluate_ddp()` to `pallatom/train/train_loop.py`, update `__main__` to support `torchrun`-based multi-node / multi-GPU training.

---

## Goal

Enable distributed data-parallel training across multiple GPUs on one or more nodes without breaking the existing single-GPU `train()` function. The new code path is launched via `torchrun` and is the only supported multi-GPU entry point.

---

## Approach

**Option A (chosen):** `torchrun`-native. Process spawning is handled by the launcher; `__main__` calls `init_process_group` then `train_ddp()` directly. No `mp.spawn`.

---

## Components

### 1. `evaluate_ddp(rank, world_size, model, loader, tcfg, distogram_res, distogram_atom, index_embedding, device)`

- Identical forward-pass logic to `evaluate()`.
- Val loader is backed by `DistributedSampler(shuffle=False)` — each rank processes a disjoint shard.
- After the local loop, packs the 8 metric sums + `n_batches` into a single `torch.tensor` on `device` and calls `dist.all_reduce(..., op=ReduceOp.SUM)` — one collective.
- Divides aggregated sums by aggregated `n_batches` to produce global means.
- Returns the result dict on all ranks; only rank 0 logs it.
- Note on padding: `DistributedSampler(drop_last=False)` may pad the last batch per rank with duplicates; the error is at most `world_size / total_val_batches` and acceptable at current scale. Switch to `drop_last=True` if exact metrics become important.

### 2. `train_ddp(rank, local_rank, world_size, model, tcfg, train_loader, test_loader, distogram_res, distogram_atom, index_embedding)`

**Signature** — replaces `device: str` with explicit process-identity args.

**DDP wrapping:**
- `ddp_model = DDP(model, device_ids=[local_rank])`
- `ddp_embedding = DDP(index_embedding, device_ids=[local_rank])`
- `distogram_res` and `distogram_atom` have no trainable parameters; they remain plain modules.

**Optimizer:** Built over `list(ddp_model.parameters()) + list(ddp_embedding.parameters())` — same interface as before.

**Epoch loop:**
- `train_loader.sampler.set_epoch(epoch)` called before iterating to reseed the `DistributedSampler` shuffle consistently across ranks.
- tqdm progress bar shown on rank 0 only.
- Per-step wandb logging guarded with `if rank == 0`.

**Checkpointing:**
- `ddp_model.module.state_dict()` and `ddp_embedding.module.state_dict()` unwrap DDP before saving.
- Checkpoint files are format-identical to those produced by `train()` and can be loaded by either function.
- Checkpoint saves guarded with `if rank == 0`.

**Evaluation:** Calls `evaluate_ddp(...)` each epoch; result dict used on rank 0 for wandb logging and best-checkpoint tracking.

### 3. `__main__` changes

New flow when invoked via `torchrun`:

```
dist.init_process_group(backend="nccl")
rank       = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
world_size = dist.get_world_size()
device     = f"cuda:{local_rank}"
torch.cuda.set_device(local_rank)
```

- Model and `index_embedding` constructed and moved to `device` **before** DDP wrapping.
- Pretrained weights loaded before DDP wrapping: all ranks call `torch.load(..., map_location=device)` independently. DDP's `__init__` then broadcasts params from rank 0 to all others to guarantee consistency.
- `DistributedSampler` created for both train and val datasets; passed into `DataLoader` with `shuffle=False` (sampler handles shuffling).
- `wandb.init(...)` called on rank 0 only.
- `train_ddp(rank, local_rank, world_size, ...)` called instead of `train(...)`.
- `dist.destroy_process_group()` called in a `finally` block.

**Launch command:**
```bash
# Single node, N GPUs (debug mode)
torchrun --nproc_per_node=N train_loop.py --data ... --splits ...

# Multi-node
torchrun --nnodes=NODES --nproc_per_node=GPUS_PER_NODE \
         --rdzv_backend=c10d --rdzv_endpoint=HOST:PORT \
         train_loop.py --data ... --splits ...
```

---

## Data Flow

```
torchrun (per node)
  └─ N processes, one per GPU
       ├─ Each: init_process_group → build model → DDP wrap
       ├─ Each: DistributedSampler splits dataset into N shards
       ├─ Each epoch:
       │    ├─ train shard → forward/backward → DDP all-reduce grads → optimizer step
       │    └─ val shard → evaluate_ddp → all_reduce metrics → rank 0 logs
       └─ rank 0: checkpoint, wandb, tqdm
```

---

## Invariants

- `train()` (single-GPU) is unchanged.
- Checkpoint file format is unchanged — `train()` and `train_ddp()` produce identical checkpoint keys.
- `jaxtyped(typechecker=beartype)` decorator kept on `train_ddp()` — beartype validates at call time, compatible with DDP.

---

## Out of Scope

- Gradient checkpointing / activation recomputation.
- Mixed-precision (`torch.cuda.amp`).
- FSDP or model parallelism.
- Changes to `TrainConfig`, `ModelParams`, or any architecture files.

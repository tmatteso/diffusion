# pallatom/train

Training loop for the `MainTrunk` EDM denoiser: optimizer setup, gradient
accumulation under a token budget, the composite diffusion loss, evaluation,
metric logging, checkpointing, and optional multi-GPU (DDP) training.

## Contents

| File | Purpose |
|------|---------|
| `train_loop.py` | Checkpoint I/O, the per-micro-batch forward/backward (`take_step`), gradient-accumulation plumbing, evaluation, epoch-level metric logging, the `train` loop, and the `main` CLI entry point. |
| `train_config.py` | Frozen Pydantic models (`TrainConfig` and its sub-configs) describing every training hyperparameter. |

## Pipeline

```
main(args, tcfg)
  │  build data loaders, MainTrunk, Adam, StepLR, Distogram heads
  │  optionally restore a checkpoint (pretrained_weights)
  ▼
train(best_val_loss, train_loader, test_loader, model_params, log)
  │  optionally wrap model in DDP
  │
  │  for epoch in 1 .. num_epochs:
  │    for batch in train_loader:
  │      accumulate into micro_buffer until per-rank token budget exceeded
  │        └─ flush_micro_buffer → optimizer_step → process_accum_window
  │                                                    └─ take_step (×N micro-batches)
  │    evaluate(test_loader)                    # full validation pass
  │    log_epoch(...)                           # structlog + optional W&B
  │    if val total_loss improved: save_checkpoint(...)
```

## `main` — entry point (`train_loop.py`)

1. Parses CLI args (`_parse_args` → `TrainArgs`): dataset JSONL, shard
   directory, train/val/test split keys, a `TrainConfig` JSON path, a
   structured-log output path, and `--ddp`/`--debug_run` flags.
2. If `--ddp`, enters `DistProcessGroup("nccl")` to initialise the
   distributed process group and resolve `rank`/`is_rank_zero`/`device`;
   otherwise runs single-device with `rank=0`.
3. Enters `StructlogConfig` (JSON log lines, rank-0-only) and `FatalOnError`
   (logs any uncaught exception via structlog, then exits) for the rest of
   the run.
4. Builds the train/val data loaders via `make_bucketed_data_loaders` —
   token-budget-aware bucketed batching over pre-sharded protein data (see
   `helpers/data.py`); out of scope for this doc.
5. Constructs `MainTrunk` (see
   [`architecture/docs/main_trunk.md`](../architecture/docs/main_trunk.md)),
   an `Adam` optimizer (`tp.lr`, `tp.weight_decay`), and a `StepLR` scheduler
   (`tp.lr_decay_steps`, `tp.lr_decay_factor`).
6. Constructs the two `Distogram` head modules used both to build ground-truth
   distance-bin labels and (for the residue distogram) as the template
   conditioning signal:
   - `distogram_res`: `n_bins - 1` real bins **plus an overflow bin**
     (`overflow_bin=True`), since `ResidueDistogramParams.n_bins` already
     includes the overflow slot in its count.
   - `distogram_atom`: exactly `n_bins` bins, no overflow bin — the local
     atom-pair window guarantees every pair distance falls within
     `[min_dist, max_dist]`.
7. Bundles everything into a `ModelSetup` (model, `TrainConfig`, both
   distogram heads, device, optimizer, scheduler).
8. If `tcfg.training.pretrained_weights` is set, restores model, optimizer,
   and scheduler state via `load_checkpoint` (and inherits its
   `best_val_loss`); otherwise starts from `best_val_loss = inf`.
9. On rank 0, if `tcfg.logging.use_wandb`, calls `wandb.init` with the full
   config dumped as run metadata.
10. Calls `train(...)`.

## `train` — the per-epoch loop

If a distributed process group is active, the model is wrapped in `DDP`
(`find_unused_parameters=False`, `gradient_as_bucket_view=True` — both
require that every parameter always receives a gradient, which holds here
since `MainTrunk`'s forward pass always exercises every sub-module). The
global token budget (`tp.accumulated_token_budget`) is divided by
`world_size` to get `per_rank_token_budget`, so the *effective* total batch
size across all ranks stays constant regardless of how many GPUs are used.

For each epoch: the model is set to `.train()`, and the loop reads batches
from `train_loader`, appending each to `micro_buffer` and tracking
`accum_tokens` (summed `f_pseudo_beta_mask` counts — real residues, not
padding). **Before** a batch would push `accum_tokens` over
`per_rank_token_budget`, the buffer is flushed (`flush_micro_buffer`) and
reset — so every optimizer step processes as many whole protein batches as
fit under the token budget, then takes exactly one optimizer step. After the
final batch of the epoch, `evaluate` runs a full pass over `test_loader`,
`log_epoch` records the epoch's metrics, and a checkpoint is saved if
validation `total_loss` improved.

### Gradient accumulation under a token budget

Protein batches vary in size (different residue counts, different numbers of
proteins packed per batch), so a fixed *batch count* budget would give each
optimizer step a wildly different amount of signal. Accumulating by *token
budget* instead keeps each optimizer step's total training signal roughly
constant.

- **`flush_micro_buffer`** calls **`optimizer_step`**, which calls
  **`process_accum_window`** to run forward+backward over every micro-batch
  in the buffer, then clips gradients, steps the optimizer and scheduler,
  and zeros gradients.
- **`process_accum_window`** scales each micro-batch's loss by
  `total_proteins / n_proteins_i` before backward, so the summed gradient
  across the whole accumulation window is equivalent to a single backward
  pass over one large batch containing every protein in the window — not a
  simple average, which would under-weight windows with more micro-batches.
  Metrics are aggregated with the same protein-count weighting.
- **`DDPNoSync`** wraps every micro-batch except the last one in the window
  in DDP's `no_sync()` context (when the model is DDP-wrapped), so gradient
  all-reduces only happen once per accumulation window instead of once per
  micro-batch — an important cost saving since a window can contain many
  micro-batches.
- **`component_grad_norms`** captures the per-submodule gradient L2 norm
  (`ComponentNorms`) *before* clipping, purely for diagnostic logging — one
  norm per named `MainTrunk` sub-module (`template_embedder`,
  `atom_encoder`, `atom_decoders`, both distogram heads, and the
  intermediate/final sequence-head projections).
- Gradients are then clipped to `tp.grad_clip` (default `10.0`; `None`
  disables clipping) via `nn.utils.clip_grad_norm_`, and the optimizer and
  LR scheduler each take one step.

## `take_step` — the EDM training step

This is the core forward/backward pass, run once per micro-batch (train or
eval):

1. **EDM loss weighting**: `lambda_sigma_loss_weight = (t̂² + sigma_data²) /
   (t̂ · sigma_data)²` — the same per-sample noise-level weighting used
   inside `atom_loss` and `med_loss`
   ([`architecture/docs/losses.md`](../architecture/docs/losses.md)), computed
   here from the batch's own `t_hat` (each training sample is noised to an
   independently-sampled noise level; see `NoiseScheduleParams.P_mean`/
   `P_std` below) and passed through to those loss functions.
2. **`StepContext`** puts the model in eval mode with gradients disabled when
   `train_mode=False` (used by `evaluate`), or leaves it in train mode
   (dropout active, gradients enabled) otherwise.
3. `model_params.model(featurized_batch)` runs the full `MainTrunk` forward
   pass, producing `PredictedOutputs` (see
   [`architecture/docs/main_trunk.md`](../architecture/docs/main_trunk.md)).
4. All six loss components are computed and combined into `total_loss`
   exactly as documented in
   [`architecture/docs/losses.md` → Combining the losses](../architecture/docs/losses.md#combining-the-losses-total_loss):
   `Kabsch_aligned_MSE_loss` (`atom_loss`), `CE_loss` (`seq_ce_loss`),
   `smooth_lddt_loss`, `residue_distogram_loss`, `atom_distogram_loss`, and
   `intermediate_loss` (`med_loss`), weighted by `LossParams`' `lam` /
   `alpha_0..4`.
5. If `train_mode`, `torch.autograd.backward([total_loss / grad_scale])` —
   `grad_scale` is the protein-count reweighting factor from
   `process_accum_window` (`1.0` when called standalone, as in tests).
6. **RMSD** (for monitoring only, not part of the loss): the ground-truth
   structure is Kabsch-aligned onto the *detached* denoised prediction (same
   convention as `atom_loss`, but unweighted by `lambda_sigma` and reported
   as a plain root-mean-square deviation rather than a squared, EDM-weighted
   loss) — a human-interpretable Å figure alongside the training-loss
   components.
7. Returns `LossMetrics` (every loss component, detached) and
   `ThroughputStatistics` (batch size, the fraction of the padded batch that
   was real residue vs. padding — `token_pack_rate` — and residues/atoms
   processed per second, timed around the whole step).

## Evaluation

`evaluate` runs `take_step` with `train_mode=False, grad_scale=0.0` (backward
is skipped entirely since `train_mode` is `False`) over every batch in the
given loader, with a `tqdm` progress bar shown on rank 0 only, and returns
the plain (unweighted-by-protein-count — every batch counts equally)
mean `LossMetrics`/`ThroughputStatistics` over the full pass.

## Metrics objects (`helpers/useful_objects.py`)

All four metrics dataclasses (`LossMetrics`, `ThroughputStatistics`,
`ComponentNorms`) mix in `TensorAccumulatorMixin`, which gives them `+=`,
`*`, and `/=` operators over every scalar-tensor field at once — this is
what lets `process_accum_window` and the per-epoch loop accumulate
protein-count-weighted running sums without hand-written per-field
bookkeeping. `EpochMetrics` bundles the epoch's train/val `LossMetrics`,
train `ThroughputStatistics`, and train `ComponentNorms` together with the
epoch number and global step count; `StepProgress` is the mutable
per-epoch accumulator (`loss_sum`, `throughput_sum`, `norms_sum`,
`n_proteins_total`, plus the tqdm bar and rank) that `flush_micro_buffer`
updates after every optimizer step.

## Logging and checkpointing

`log_epoch` writes four structlog entries per epoch (`"train"`,
`"throughput_statistics"`, `"gradient_norms"`, `"val"`) and, when
`tcfg.logging.use_wandb` is set, mirrors the same data to Weights & Biases
under `train/`, `throughput_statistics/`, `gradient_norms/`, and `val/`
prefixes. It is a no-op when `do_log=False`, which `train` passes on every
non-rank-0 worker to avoid duplicate I/O under DDP.

`save_checkpoint`/`load_checkpoint` (de)serialise a `Checkpoint` — model,
optimizer, and scheduler state dicts plus `best_val_loss` — to/from
`tcfg.checkpoint.checkpoint_path`. Saving is a no-op on non-rank-0 workers,
and the DDP wrapper (if any) is stripped via `.module` before calling
`state_dict()` so a checkpoint saved under DDP loads correctly into a plain
`MainTrunk` and vice versa.

`train` only calls `save_checkpoint` once per epoch, when the epoch's
validation `total_loss` improves on `best_val_loss`. **Note**:
`CheckpointParams.save_every` is documented on `log_epoch`'s docstring as
also gating a periodic per-epoch checkpoint, but no such call exists in the
current `train_loop.py` — only the best-validation-loss checkpoint is
actually written.

## Distributed training (DDP)

Passing `--ddp` makes `main` enter `DistProcessGroup("nccl")`, which
initialises `torch.distributed`, resolves this process's `rank`,
`world_size`, and `device`, and tears the process group down on exit
(propagating a `DistributedPeerError` across the group if a peer rank dies
mid-collective, rather than hanging). `train` then wraps the model in `DDP`
and divides the token budget by `world_size` (see
[Gradient accumulation under a token budget](#gradient-accumulation-under-a-token-budget)
above). Without `--ddp`, everything runs identically with `rank=0,
world_size=1` and no `DDP` wrapper.

## Parameters (`train_config.py`)

`TrainConfig` is the top-level frozen Pydantic model, loaded from the JSON
path passed via `--config`. It aggregates:

### `training` — `TrainingParams`

| Field | Default | Role |
|---|---|---|
| `num_epochs` | `50` | Total epochs the outer loop in `train` runs. |
| `lr` | `1e-3` | Peak learning rate passed to `Adam`. |
| `weight_decay` | `1e-4` | L2 regularisation coefficient in `Adam`. |
| `grad_clip` | `10.0` | Max gradient norm passed to `clip_grad_norm_` in `optimizer_step`; `None` disables clipping (`float("inf")` is used instead). |
| `pretrained_weights` | `None` | Checkpoint path loaded via `load_checkpoint` before training starts, when set. |
| `resume_checkpoint` | `None` | Documented as a full resume path (weights + optimizer state); not read anywhere in `train_loop.py` — `pretrained_weights` is the field actually wired up to `load_checkpoint`. |
| `accumulated_token_budget` | `4096` | Global (pre-`world_size`-division) token budget per optimizer step — see [Gradient accumulation](#gradient-accumulation-under-a-token-budget). |
| `lr_decay_steps` | `50_000` | Optimizer steps between each `StepLR` decay. |
| `lr_decay_factor` | `0.95` | Multiplicative LR decay applied every `lr_decay_steps` steps. |

### `model` — `ModelParams`

Architecture capacity/channel-dimension parameters consumed by `MainTrunk`'s
constructor — documented in
[`architecture/docs/main_trunk.md`](../architecture/docs/main_trunk.md) and
the [tensor reference table](../CLAUDE.md#maintrunk-tensor-reference) in
`pallatom/CLAUDE.md` (`c_res`, `c_pair`, `c_atom`, `c_atompair`, `K_unit`,
attention block/head counts, etc.).

### `noise` — `NoiseScheduleParams`

The same EDM noise-schedule model used for sampling (see the `noise` section
of [`pallatom/sample/README.md`](../sample/README.md)); during training,
`P_mean`/`P_std`
additionally parameterise the log-normal distribution each training sample's
noise level `t_hat` is drawn from (upstream in the data pipeline), which is
what `lambda_sigma_loss_weight` in `take_step` is computed from.

### `distogram_res` / `distogram_atom` — `ResidueDistogramParams` / `AtomDistogramParams`

Distance-bin configuration for the two `Distogram` head modules built in
`main` (`min_dist`, `max_dist`, `n_bins`, `tok_emb_dim`). `AtomDistogramParams`
subclasses `ResidueDistogramParams` with tighter defaults
(`min_dist=0.0, max_dist=10.0, n_bins=22`) suited to local atom-atom
distances rather than inter-residue Cβ distances.

### `loss` — `LossParams`

The weights and thresholds combined into `total_loss` in `take_step` — see
the [Combining the losses](../architecture/docs/losses.md#combining-the-losses-total_loss)
table in `architecture/docs/losses.md` for exactly how `lam`/`alpha_0..4`
map onto each loss term. `gamma` and `smooth_lddt_cutoff` are consumed
inside `med_loss` and `smooth_lddt_loss` respectively, not in `train_loop.py`
directly.

### `checkpoint` — `CheckpointParams`

| Field | Default | Role |
|---|---|---|
| `checkpoint_path` | `pallatom_best.pt` | Where `save_checkpoint`/`load_checkpoint` read and write. |
| `save_every` | `1` | Documented as a periodic-checkpoint cadence (see the note in [Logging and checkpointing](#logging-and-checkpointing)); not currently consumed by any code path. |

### `logging` — `LoggingParams`

| Field | Default | Role |
|---|---|---|
| `use_wandb` | `True` | Gates both `wandb.init` in `main` and the `wandb.log` call inside `log_epoch`. |
| `wandb_project` | `"pallatom-training"` | W&B project name passed to `wandb.init`. |

### `train_loader` / `test_loader` — `TrainLoaderConfig` / `LoaderConfig`

DataLoader construction parameters consumed by `make_bucketed_data_loaders`
(`helpers/data.py`), out of scope for this document beyond noting their
existence: `max_seq_length`, `batch_size`, `num_workers` (shared base
`LoaderConfig`), plus training-only fields on `TrainLoaderConfig`
(`token_budget`, `batch_prefetch_depth`, `epoch_prefetch_depth`, `seed`,
`n_threads`, `n_shards`, `noise_magnitude`) governing the sharded,
bucketed data pipeline.

### `conditioning_dropout` — `ConditioningDropoutConfig`

`p_distogram`, `p_atom`, `p_seq` (each default `0.15`) are documented as
per-conditioning-signal dropout probabilities intended to make the model
robust to missing template/sequence conditioning. As with `save_every`
above, no code path in this repository currently reads these three fields —
they are validated as part of `TrainConfig` but not yet wired into the data
or forward pipeline.

## Usage

```bash
python -m train.train_loop \
  --dataset_jsonl path/to/proteins.jsonl \
  --shard_dir path/to/shards/ \
  --keys_for_splits_json path/to/splits.json \
  --config path/to/train_config.json \
  --structlog_jsonl path/to/log.jsonl \
  [--ddp] [--debug_run]
```

`--ddp` enables `DistributedDataParallel` training (launch with
`torchrun`/`torch.distributed.launch`); `--debug_run` restricts the dataset
to 252 proteins for fast iteration.

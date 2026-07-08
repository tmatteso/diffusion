# `data.py` — dataset and data loading for training

[← back to helpers overview](README.md)

Dataset and data-loading utilities for protein structure files: lazy-loading
proteins from JSONL, the per-protein and per-batch featurization pipeline
that turns a raw `Protein` into model-ready diffusion training tensors, and
a shard-based FFD (First-Fit-Decreasing) bucketed batching system for
large-scale training.

## `FeaturizedItem` / `FeaturizedBatch`

`FeaturizedItem` holds every tensor produced by `featurize_single_item` for
**one** protein, before batching: `r_gt`/`r_gt_noised` (flat atom-indexed
ground-truth and noised positions), `atom5_mask`, `f_pseudo_beta_mask`,
`gt_res_distogram_indices` (the distogram loss target), `noised_res_distogram`
(the self-conditioning template), `aa_indices`, `ref_pos`/`ref_element`
(the tiled reference conformer), `f_residue_idx`, `t_hat` (the sampled
noise level), `t_normalized`, `ref_space_uid`, `tok_idx`, `center_uid`, and
the sparse `gt_atom_distogram_sparse`/`gt_atom_distogram_mask_sparse`.

`FeaturizedBatch` is the same set of fields with a leading batch dimension
`B` prepended — the type `featurize_batch` and `build_sampling_context`
(`sample/sampling.py`) both produce, and what `MainTrunk.forward` consumes
directly. It also carries `.to(device, non_blocking=...)` and
`.pin_memory()` methods (both returning a new `FeaturizedBatch` via
`dataclasses.replace`) so the whole batch can be moved to the GPU or pinned
as a unit. For the full per-tensor shape/meaning reference, see the
[MainTrunk tensor reference](../CLAUDE.md#maintrunk-tensor-reference) table
in `pallatom/CLAUDE.md`.

## `Distogram`

An `nn.Module` that precomputes `n_bins + 1` linearly-spaced bin edges
between `min_dist` and `max_dist` at construction, then maps per-residue
coordinates to a one-hot distogram plus a pair-validity mask. Accepts either
one atom per residue (`(..., N, 3)`, e.g. pseudo-Cβ, auto-expanded to
`(..., N, 1, 3)`) or several (`(..., N, A, 3)`, e.g. atom5/atom37). When
`overflow_bin=True`, an extra bin captures distances beyond `max_dist` and
every valid atom pair contributes to the mask regardless of distance; when
`overflow_bin=False`, the output has exactly `n_bins` classes and pairs
farther than `max_dist` are marked invalid in the mask instead. This one
module is reused for three distinct purposes across the codebase: the
residue-level Cβ distogram (both the ground-truth loss target *and* the
noised self-conditioning template), and the sparse atom-pair distogram.

## `ref_pos_for_residue` / `sinusoidal_encoding`

- **`ref_pos_for_residue(resname)`** returns the `(5, 3)` atom5 reference
  conformer for a residue name, looked up from
  `atom_utils.rigid_group_atom_positions`; atoms absent from that table
  default to the origin. `featurize_single_item` always calls this with
  `"ALA"` — every residue is given the same alanine reference geometry
  regardless of its true identity, since the model must predict structure
  *and* sequence from a sequence-agnostic starting point.
- **`sinusoidal_encoding(positions, dim)`** is a standard sin/cos
  positional encoding (log-spaced frequencies, half the output channels
  sine, half cosine) mapping per-residue integer indices to a `dim`-wide
  embedding.

## `featurize_single_item` — the core training-example builder

This is the function that turns one raw `Protein` (atom37 coordinates,
sequence, residue indices) into everything a single diffusion training
example needs. Its docstring: *"Converts atom37 coordinates to atom5, pads
to max_seq_len_in_batch, computes ground-truth and self-conditioning Cβ
distograms, builds the sparse atom-pair distogram over the local residue
window, tiles ALA reference geometry across residues, and samples a
diffusion noise level from the lognormal schedule."* Concretely, in order:

1. **Padding.** `atom37_positions`, `atom37_mask`, and `f_residue_idx` are
   zero-padded (via `F.pad`) from the protein's true length up to
   `max_seq_len_in_batch` — the longest protein in the current batch — so
   every item in a batch shares one uniform `N_res`. `aa_indices` is padded
   with `-100` (a sentinel that `seq_ce_loss` ignores, see
   [`architecture/docs/losses.md`](../architecture/docs/losses.md)) rather
   than `0`, so padding never masquerades as a valid amino acid.
2. **Reference conformer.** `ref_pos_for_residue("ALA")` and
   `ATOM5_ELEMENTS` are tiled across every one of the `N_res` residue slots
   to build `ref_pos`/`ref_elem` — the same sequence-agnostic reference
   geometry the model always starts from at both train and sample time.
3. **Noise level sampling.** `t_hat` — the per-example EDM noise level σ —
   is drawn from the log-normal schedule: `t_hat = sigma_data *
   exp(N(0,1) * P_std + P_mean)`, i.e. `ln(t_hat / sigma_data) ~
   N(P_mean, P_std²)`, using `tcfg.noise.P_mean`/`P_std`/`sigma_data`. This
   is the reparameterisation trick applied to the EDM training noise
   distribution — the same `sigma_data` that also parameterises the
   *sampling* schedule (see the `noise` section of
   [`sample/README.md`](../sample/README.md)).
4. **Template time weight.** `t_normalized` is a single `uniform(0, 1)`
   draw, broadcast to every `(N_res, N_res)` pair — the time-conditioning
   signal `TemplateEmbedder` uses to weight the template distogram (see
   `architecture/docs/template_embedder.md`).
5. **atom37 → atom5.** `atom37_to_atom5` extracts the 5 backbone+Cβ atoms;
   `residue_mask = atom5_mask.any(dim=-1)` marks which padded/real residues
   have at least one valid atom (this doubles as `f_pseudo_beta_mask`).
6. **Random rigid augmentation.** `centre_random_augment`
   ([`alignment.md`](alignment.md#centre_random_augment)) is applied to the
   flattened atom5 positions *before* noising, so `r_gt` and `r_gt_noised`
   share the same randomly-rotated/translated frame — this is AF3
   Algorithm 20's training-time augmentation, preventing the model from
   ever learning a canonical global orientation.
7. **Ground-truth Cβ distogram.** Computed from the **unaugmented**
   `atom37_positions` via `atom37_to_cb`, deliberately *not* from the
   post-augmentation `flat_pos` — a comment in the source explains why:
   `centre_random_augment` is a rigid transform, so pairwise distances (and
   therefore the distogram) are numerically identical either way, and
   recomputing from the augmented frame would just be wasted work.
   `gt_res_distogram_indices` is the `argmax` bin index of this distogram —
   the residue-distogram loss target.
8. **Noising.** Zero-mean Gaussian noise scaled by `t_hat` is added to the
   augmented `flat_pos` to produce `r_gt_noised`; the noise itself is
   re-centred (`masked_com`, valid atoms only) before scaling so it doesn't
   introduce a net translation.
9. **Self-conditioning template.** The **noised** positions are converted
   back to atom37 and through `atom37_to_cb` again, and *that* Cβ set is
   passed through the same `Distogram` module (`c_beta_distogram_fn`) to
   produce `noised_res_distogram` — the template signal actually fed into
   the model as self-conditioning (distinct from the clean-coordinate
   target in step 7).
10. **Index tensors.** `token_idx` maps each of the `N_res * 5` atoms to its
    parent residue (`0..N_res)`); `center_uid` marks each residue's
    designated "center" atom (the Cα slot, `ATOM5_CA`) repeated across all 5
    of that residue's atom entries; `ref_space_uid` is set equal to
    `f_residue_idx` repeated per atom (today's single-chain setting makes
    residue index and space/chain identifier the same value).
11. **Sparse atom distogram.** `build_sparse_pairs` (see
    `architecture/docs/atom_transformers.md`) finds each atom's local
    `K`-neighbour window; pairwise distances are computed **only** for
    those `K` neighbours (avoiding the dense `O(N²)` intermediate a full
    `Distogram.forward` would allocate) and bucketized directly into
    `gt_atom_distogram_sparse`/`gt_atom_distogram_mask_sparse` — the sparse
    atom-distogram loss target and its validity mask (valid atom **and**
    valid neighbour **and**, when `atom_distogram_fn.overflow_bin` is
    `False`, within `max_dist`).

The result is one fully-formed `FeaturizedItem`, ready to be stacked with
others from the same batch.

## `featurize_batch` / `FeaturizeCollate`

`featurize_batch` finds the longest protein in the incoming `list[Protein]`,
calls `featurize_single_item` once per protein with that shared
`max_seq_len_in_batch`, and stacks every field across the batch into a
`FeaturizedBatch`. `FeaturizeCollate` is a picklable `dataclasses.dataclass`
wrapper around `featurize_batch` (capturing `tcfg`/`distogram_res`/
`distogram_atom`) that satisfies the `DataLoader` `collate_fn` contract and
survives the pickle round-trip required by multi-worker `DataLoader`s.

## Dataset loading and splits

- **`ProteinEntry`** — the minimal Pydantic schema for one JSONL line
  (`name`, `seq`, `coords`).
- **`ProteinNamesManifest`** / **`DatasetSplitsManifest`** — a flat name
  list, and the `train`/`validation`/`test` name-list split manifest
  (plus optional CATH topology metadata) loaded from the `--keys_for_splits_json`
  file.
- **`ProteinDataset`** — a lazy-loading `torch.utils.data.Dataset` backed
  directly by a JSONL file: it scans the file once at construction to build
  a name → byte-offset index (keeping only offsets in memory, not the
  protein data itself), and each `__getitem__` seeks to and parses only the
  requested line, then centres and pads/truncates it to `max_seq_length` via
  `make_np_example`/`center_positions`/`make_fixed_size`/`truncate_to_length`
  (`atom_utils.py`). The open file handle is excluded from pickling
  (`__getstate__`) and reopened lazily per worker, so it's safe to use with
  `num_workers > 0`.

## Sharded streaming pipeline (dormant)

A second, considerably more elaborate data path is fully implemented but
**not currently wired up** — `make_bucketed_data_loaders` (below) builds all
three loaders from the plain `ProteinDataset` above; the shard-based
construction is present in the source only as commented-out code. It's worth
understanding because it's the design the training loop's token-budget
gradient accumulation (`train/README.md`) was actually built against:

- **`ShardMetadata`** — the persisted record (`shard_metadata.json`) of the
  parameters a shard directory was built with (`names_hash`, `token_budget`,
  `shard_size`, `n_shards`), used to validate an existing shard directory
  before reuse.
- **`ShardBudgetParameters`** — every scalar input needed to compute one
  epoch's batch plan: shard directory, token budget, max sequence length,
  seed, thread count, DDP `world_size`/`rank`, expected proteins-per-shard,
  length-jitter `noise_magnitude`, and `num_workers`.
- **`FFDWorkerPlan`** / **`FFDBatchPlan`** — a pre-computed, per-epoch,
  per-DataLoader-worker plan: which shards this worker streams, the
  permutation that reorders each shard's streamed proteins into sorted
  order, and the cumulative batch-end cut points.
- **`ProteinShardDataset`** — a `torch.utils.data.IterableDataset` that
  first builds shard tars if they don't exist yet (`build_sorted_shards`:
  one JSONL pass, global descending sort by sequence length, sliced into
  `shard_{id:05d}.tar` chunks), then, once a plan is injected via
  `set_plan`, streams its assigned shards (round-robin across workers) and
  cuts each shard's stream into pre-planned batches — prefetching the next
  shard from disk in a background thread while the current one is being
  yielded.
- **`ShardDataLoader.ffd_pack`** is the actual First-Fit-Decreasing bin
  packer: walking a descending-sorted, length-clamped list once, it either
  extends the currently-open batch or closes it and starts a new one,
  keeping every batch's worst-case padded cost (`batch_size *
  longest_protein_in_batch²`) within `token_budget²`. A protein whose own
  cost already exceeds the budget is emitted as a solo batch.
  `compute_ffd_plan` adds per-protein length jitter (`noise_magnitude`)
  before sorting — so batch groupings vary epoch to epoch rather than being
  perfectly deterministic — and `pack_one_shard` applies `ffd_pack`
  per-shard per-worker (valid because globally-sorted shards have disjoint
  length ranges, so FFD batches never actually need to span shards).
- **`ShardDataLoader`** wraps `ProteinShardDataset` plus a background
  `ProcessPoolExecutor`/`ThreadPoolExecutor` pair that computes (or loads
  from an on-disk cache, keyed by a hash of the budget parameters —
  `plan_cache`) each epoch's `FFDBatchPlan` one or more epochs ahead of when
  it's needed (`epoch_prefetch_depth`), so plan computation never blocks
  training.

## `make_bucketed_data_loaders` — what's actually active

Despite its name, the function currently in use builds **all three** loaders
(`train`, `val`, `test`) from the plain `ProteinDataset` above with a fixed
`batch_size` (`cfg.test_loader.batch_size`) and a standard
`torch.utils.data.DataLoader` — the `ProteinShardDataset`/`ShardDataLoader`
construction is present in the source, immediately above, but commented
out. It still auto-detects DDP (`dist.is_initialized()`): under DDP, each
loader gets a `DistributedSampler` (`shuffle=False`; val/test additionally
set `drop_last=True`); without DDP, it's a single-process loader with
`shuffle=False` on val/test. `extra_train_args.debug_run` truncates the
training name list to the first 252 proteins for fast iteration. All three
loaders share one `FeaturizeCollate` instance built from freshly-constructed
`distogram_res`/`distogram_atom` `Distogram` modules (both `.eval()`'d,
since they're deterministic binning functions with no learned parameters).

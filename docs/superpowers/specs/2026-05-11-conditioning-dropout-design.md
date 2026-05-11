# Conditioning Dropout for PallAtom

**Date:** 2026-05-11
**Status:** Approved

## Problem

The model was trained in a single regime: given the full Cβ template distogram and all atom positions as context, denoise noisy coordinates back to ground truth. There is no conditioning dropout anywhere in the pipeline, so five of the six intended inference use cases are out-of-distribution:

| Use case | At inference | Training exposure |
|---|---|---|
| 1 — Unconditional | zeros distogram, atom5_mask=all False | Never seen |
| 2 — Sequence only | zeros distogram, atom5_mask=all False | Never seen |
| 3 — Seq + partial atoms | partial atom5_mask | Never seen |
| 4 — Partial template | partial distogram | Never seen |
| 5 — Full template | full distogram, atom5_mask=all False | Closest to training |
| 6 — Atoms + partial template | partial distogram, full mask | Never seen |

The fix is classifier-free guidance style conditioning dropout: randomly zero out `gt_res_distogram`, `atom5_mask`, and `aa_indices` at the residue level during training.

## Approach

**Approach A — Standalone `apply_conditioning_dropout` function** (chosen over baking into `featurize_batch` or a model-level wrapper).

`featurize_batch` stays pure (data construction only). A separate function applies stochastic augmentation. This keeps the two concerns cleanly separated and makes each independently testable.

## Architecture

### New function

`apply_conditioning_dropout(batch: FeaturizedBatch, p_distogram: float, p_atom: float, p_seq: float, device: str) -> FeaturizedBatch`

Location: `pallatom/helpers/featurize.py`

Returns a new `FeaturizedBatch` via `dataclasses.replace` — no in-place mutation.

### Config changes

New nested config block in `TrainConfig`:

```python
@dataclass
class ConditioningDropoutConfig:
    p_distogram: float = 0.15
    p_atom: float = 0.15
    p_seq: float = 0.15
```

`TrainConfig` gains a `conditioning_dropout: ConditioningDropoutConfig` field. Defaults of 0.15 mean each residue's conditioning is present 85% of the time — a reasonable starting point.

### Model change

`"X"` is appended to `restypes` in `pallatom/helpers/atom_utils.py` (after `"V"` at position 19), making it position 20. Because `restype_order` is built from `restypes` via `enumerate`, `restype_order["X"]` automatically equals 20 and `restype_num` becomes 21. The existing `.get(r, 20)` fallbacks in `featurize.py` and `sampling.py` already handle unknown characters — adding `"X"` explicitly promotes it from a silent fallback to a first-class mask token.

The sequence embedding table in `MainTrunk` is widened from 20 → 21 entries to accommodate index 20. The new row is randomly initialized like any other embedding row.

The sequence prediction head output dimension stays at 20 (`n_amino`). The mask token is an input-side embedding only — the model is never asked to predict `"X"` as an output.

## Per-Signal Masking Logic

Three independent residue-level dropout masks are sampled at the start of `apply_conditioning_dropout`:

```python
drop_d = torch.bernoulli(torch.full((B, N_res), p_distogram)) & residue_mask
drop_a = torch.bernoulli(torch.full((B, N_res), p_atom))      & residue_mask
drop_s = torch.bernoulli(torch.full((B, N_res), p_seq))       & residue_mask
```

Masking only applies to valid residues (`& residue_mask`) — padding residues are never the source of a signal, so there is nothing to drop.

### `gt_res_distogram` (B, N_res, N_res, n_templ_bins)

For each residue `i` in `drop_d`, zero out row `i` and column `i`. Both row and column are zeroed because the distogram is symmetric — zeroing only one side would leave a spurious one-sided signal. The corresponding entry in `f_pseudo_beta_mask` is also zeroed.

```python
row_mask = ~drop_d  # True = keep
disto_mask = row_mask[:, :, None] & row_mask[:, None, :]  # (B, N_res, N_res)
new_distogram = batch.gt_res_distogram * disto_mask.unsqueeze(-1)
new_pseudo_beta_mask = batch.f_pseudo_beta_mask * row_mask.float()
```

### `atom5_mask` (B, N_atom)

The residue-level drop mask is expanded to atom level (5 atoms per residue) and ANDed with the existing atom mask:

```python
drop_a_expanded = repeat(drop_a, "b n -> b (n a)", a=5)
new_atom5_mask = batch.atom5_mask & ~drop_a_expanded
```

### `aa_indices` (B, N_res)

Dropped residue tokens are replaced with 20 (the new mask token):

```python
new_aa_indices = batch.aa_indices.masked_fill(drop_s.bool(), 20)
```

## Training Loop Integration

`apply_conditioning_dropout` is called immediately after `featurize_batch` in all three call sites in `train_loop.py`:

```python
featurized_batch = featurize_batch(...)
featurized_batch = apply_conditioning_dropout(
    featurized_batch,
    tcfg.conditioning_dropout.p_distogram,
    tcfg.conditioning_dropout.p_atom,
    tcfg.conditioning_dropout.p_seq,
    device,
)
```

## Inference / Sampling

No changes to `sampling.py`. The sampling context is already constructed with partial or zero distograms and masks per use case. `apply_conditioning_dropout` is never called outside training.

## Testing

### `test_featurize.py` — unit tests for `apply_conditioning_dropout`
- With `p=1.0` for distogram: all distogram entries and pseudo-beta mask entries are zero
- With `p=1.0` for atom: atom5_mask is all False
- With `p=1.0` for seq: all valid aa_indices equal 20
- With `p=0.0` for all: output batch is identical to input (no-op)
- Symmetry: dropped residue `i` zeroes both row `i` and column `i` of the distogram
- Drop mask respects `residue_mask`: padding residues are never changed

### `test_main_trunk.py` — embedding table width
- Forward pass with `aa_indices` containing index 20 (mask token) does not raise IndexError

### `test_train_loop.py` — integration smoke test
- One training step with `p_distogram=p_atom=p_seq=0.5` produces a finite scalar loss

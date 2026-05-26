# Sequence CE Loss Fix — Design Spec

**Date:** 2026-05-26
**Status:** Approved

---

## Problem

Two bugs in the sequence cross-entropy (CE) loss pipeline:

### Bug 1 — Padding positions get the alanine embedding instead of zero

In `embed_inputs` (`main_trunk.py:476-477`):

```python
aa_idx_clamped = aa_indices.clamp(min=0, max=20)
s_init = s_init + self.aa_embedding(aa_idx_clamped)
```

Padding residues have `aa_indices = -100`. The clamp converts -100 → 0, silently
giving padding positions the alanine (`"A"`, index 0) embedding. Padding positions
still participate in attention in `NodeUpdate` and `AtomAttentionDecoder`, so this
corrupts the representations at valid residues via cross-attention. The comment in
the code ("embedding value doesn't matter") is wrong.

### Bug 2 — Intermediate and final CE supervise opposite, non-overlapping position sets

After `apply_conditioning_dropout`, positions where dropout fired have `aa_indices = 20`
("X"). The two CE computations then diverge:

- **Intermediate CE** uses `flat_aa_targets`: original AA at X positions, -100 everywhere
  else → only masked (dropped) positions receive a gradient.
- **Final CE** uses `mask_seq_target(featurized_batch.aa_indices)`: converts 20 → -100,
  so only *visible* (non-dropped) positions receive a gradient.

They teach the sequence head opposite things with no shared logic and no shared
function. There is no test enforcing they are equivalent.

---

## Design

### Principle

- Padding and conditioning-dropped positions both contribute zero to `s_init` via the
  embedding path, but via different mechanisms (zero vector vs. null embedding).
- The CE loss is computed identically in every call site: a single jaxtyped wrapper
  that ignores padding (-100) and the null/unknown slot (index ≥ n_amino).
- `apply_conditioning_dropout` is unchanged — it correctly uses index 20 for dropped
  positions, which is also the PDB-X slot (shared null embedding, Option A).

---

### Change 1 — `embed_inputs` in `main_trunk.py`

Replace the clamp-and-lookup with a masked addition:

```python
# Padding (aa_indices = -100, residue_mask = False) → zero vector.
# PDB-X (20) and conditioning-dropped (20) → aa_embedding[20] (null embedding).
# Real AAs (0-19) → their embedding.
valid: Bool[torch.Tensor, "B N_res"] = aa_indices >= 0
emb: Float[torch.Tensor, "B N_res c_res"] = self.aa_embedding(aa_indices.clamp(0, 20))
s_init = s_init + emb * rearrange(valid.float(), "b n -> b n 1")
```

`aa_embedding` remains `nn.Embedding(21, c_res)` — no change to model parameters.

The gate `aa_indices >= 0` is False for padding (-100) in all circumstances. After
`apply_conditioning_dropout`, it is True for both real AAs (0-19) and
conditioning-dropped/PDB-X positions (20), which correctly receive the null embedding.

---

### Change 2 — New `seq_ce_loss` in `architecture/losses.py`

```python
@jaxtyped(typechecker=beartype)
def seq_ce_loss(
    logits: Float[torch.Tensor, "B N_res n_amino"],
    aa_indices: Int[torch.Tensor, "B N_res"],
) -> Float[torch.Tensor, ""]:
    """CE loss for the sequence head; ignores padding and null/unknown tokens.

    Positions are ignored when aa_indices < 0 (padding) or aa_indices >= n_amino
    (PDB-X at index 20, conditioning-dropped at index 20). Only real visible amino
    acids (indices 0 to n_amino-1) contribute to the loss.

    Args:
        logits: Sequence logits from any head (final or intermediate).
        aa_indices: Per-residue amino acid indices, post-conditioning-dropout.

    Returns:
        Scalar mean CE loss over all valid positions.
    """
    n_amino: int = logits.size(-1)
    targets: Int[torch.Tensor, "B N_res"] = aa_indices.masked_fill(
        aa_indices >= n_amino, -100
    )
    return F.cross_entropy(
        rearrange(logits, "b n c -> (b n) c"),
        rearrange(targets, "b n -> (b n)"),
    )
```

`F.cross_entropy` ignores index -100 by default. The `masked_fill` converts any
index ≥ n_amino (i.e., index 20 = X/null) to -100 before passing to CE.

---

### Change 3 — Simplify `take_step` in `train_loop.py`

**Remove:**
- `mask_seq_target` function (entire function, no longer used)
- `orig_aa_indices` clone
- `mask_positions`
- `aa_targets`
- `flat_aa_targets`
- `n_masked_aa`
- The `if train / else` branch inside the intermediate CE loop

**Replace all CE call sites with:**
```python
# Intermediate (inside decoder loop):
inter_ce = seq_ce_loss(
    pred_outputs.intermediate_pred_aa_logit_stack[k_idx],
    featurized_batch.aa_indices,
)

# Final:
CE_loss = seq_ce_loss(pred_outputs.seq_logits, featurized_batch.aa_indices)
```

Both training and validation paths call the same function with the same arguments.
The `if train:` guard around the intermediate CE block is removed.

---

## Tests — `pallatom/tests/test_seq_ce_loss.py`

Five tests, all using `@jaxtyped(typechecker=beartype)` and Google-style docstrings
per the project conventions.

| # | Name | What it checks |
|---|------|----------------|
| 1 | `test_intermediate_and_final_ce_are_identical` | Given identical logits and `aa_indices`, `seq_ce_loss` returns the same scalar whether called for an intermediate or final head. Asserts the two results are `torch.allclose`. |
| 2 | `test_conditioning_dropout_uses_index_20_not_minus_100` | With `p_seq=1.0` applied to a batch where all residues are valid (`residue_mask=True`), every `aa_indices` entry equals 20 (the null token). No entries become -100. Padding positions (already -100) are unaffected. |
| 3 | `test_seq_ce_loss_ignores_minus_100` | Manually construct logits and targets where half the positions are -100. Assert the scalar loss equals the loss computed on only the non-(-100) positions via a direct `F.cross_entropy` call. |
| 4 | `test_seq_head_has_no_x_output_slot` | Run a full forward pass of `MainTrunk` on a minimal batch. Assert `seq_logits.shape[-1] == 20` and every tensor in `intermediate_pred_aa_logit_stack` also has `shape[-1] == 20`. Index 20 cannot be the argmax output. |
| 5 | `test_pipeline_preserves_pdb_x_as_index_20` | Construct a synthetic `aa_sequence` string containing `'X'`. Run through `featurize_single_item`. Assert the corresponding position in `aa_indices` equals 20. |

---

## Files Changed

| File | Change |
|------|--------|
| `pallatom/architecture/main_trunk.py` | Fix `embed_inputs`: zero vector for padding, masked embedding for all others |
| `pallatom/architecture/losses.py` | Add `seq_ce_loss` |
| `pallatom/train/train_loop.py` | Remove `mask_seq_target`, simplify `take_step` CE logic |
| `pallatom/tests/test_seq_ce_loss.py` | New test file (5 tests) |

`pallatom/helpers/featurize.py` — **not changed**. `apply_conditioning_dropout`
already uses index 20 correctly for the null embedding design.

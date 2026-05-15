# Gradient-Flow Integration Tests — Design Spec

**Date:** 2026-05-15
**Status:** Approved

---

## Problem

The existing gradient test in `test_main_trunk.py` (`test_main_trunk_gradient_flows_to_r_input`)
only verifies that a single scalar reduction of `r_denoised` back-propagates to `r_input`. It does
not use the real composite training loss, does not check model parameters, and does not exercise the
full `train_step()` path. The file has a comment at line 425 explicitly asking for integration tests.

---

## Goal

Add two integration tests that together prove the entire computation graph — from input features
through all encoder/decoder submodules to all loss terms — produces finite, nonzero gradients at
every trainable parameter.

---

## Files Changed

| File | Change |
|---|---|
| `pallatom/tests/architecture/test_main_trunk.py` | Add `test_integration_gradient_flow_composite_loss` after line 425 |
| `pallatom/tests/train/test_train_loop.py` | Add `test_integration_gradient_flow_via_train_step` |

---

## Test 1 — `test_integration_gradient_flow_composite_loss`

**Location:** `pallatom/tests/architecture/test_main_trunk.py`, immediately after the line 425 comment.

**Purpose:** Tests the model in isolation. Builds the full 7-term composite loss inline, calls
`.backward()`, and asserts per-submodule gradient presence. Independent of `TrainConfig` and
`train_step()` internals.

### Fixtures reused

All fixtures are already defined in the file:
- `model: MainTrunk` — small model in eval mode; test calls `.train()` to enter training mode.
- `featurized_batch: FeaturizedBatch` — complete batch with `aa_indices = zeros` (valid class 0
  targets), `r_gt = zeros`, `atom5_mask = all-True`, `residue_mask = all-True`.

### Loss terms constructed inline

| Term | Function | Notes |
|---|---|---|
| Kabsch MSE | `atom_loss(r_denoised, r_gt, atom5_mask).mean()` | Main coordinate denoising loss |
| Sequence CE | `F.cross_entropy(f_seq_logits reshaped, aa_indices reshaped)` | Final seq head |
| Smooth lDDT | `smooth_lddt_loss(r_denoised, r_gt, atom5_mask, cutoff=15.0)` | Local distance test |
| Residue distogram | `distogram_loss_residue(res_logits, gt_res_bin_idx, residue_mask).mean()` | `gt_res_bin_idx = gt_res_distogram.argmax(-1).clamp(...)` |
| Atom distogram | `distogram_loss_atom(atom_logits, gt_atom_distogram_sparse, gt_atom_distogram_mask_sparse).mean()` | Sparse K-neighbour |
| Intermediate (×K_unit) | `atom_loss(inter_coords, r_gt, mask) + F.cross_entropy(inter_logits, aa_indices)` × `0.99^(K-k-1)` | One term per decoder unit |

Loss weights are all 1.0 (no `TrainConfig` dependency). The scalar `total_loss = sum(all terms)` is
the only call site for `.backward()`.

### New imports required

```python
from architecture.losses import atom_loss, smooth_lddt_loss, distogram_loss_residue
```

(`distogram_loss_atom` is already imported in the file.)

### Assertion strategy — per-submodule bucket check

After `.backward()`, walk `model.named_parameters()` and bucket by the first dot-segment of each
name. For each bucket assert: **at least one parameter has a gradient that is finite everywhere and
has a nonzero absolute maximum**.

Submodule prefixes expected to be covered:

```
proj_residue_idx, time_fourier, aa_embedding, rel_pos_enc, template_embedder,
atom_encoder, norm_s_init, proj_s_init,
node_updates, atom_decoders, pair_updates,
residue_distogram_head, atom_distogram_head,
inter_proj_seq, inter_seq_logits,
proj_seq, seq_logits
```

If any bucket has no finite nonzero gradient, the assertion message names the failing prefix.

---

## Test 2 — `test_integration_gradient_flow_via_train_step`

**Location:** `pallatom/tests/train/test_train_loop.py`.

**Purpose:** Tests the production training path end-to-end. Calls `train_step()` as a black box,
then reads `.grad` attributes from model parameters. Gradients persist on parameters after
`train_step()` returns because `optimizer.zero_grad()` is the first thing called at the *next*
`train_step()` invocation — not the last thing at the end of the current one.

### Fixtures reused

All fixtures already exist in the file:
- `model: MainTrunk` — small model with `_C_RES=32`, `_C_ATOM=32`, `_C_PAIR=32`, `_K_UNIT=1`.
- `tcfg: TrainConfig` — 1-epoch config, `grad_clip=1.0`, W&B disabled.
- `distogram_res: Distogram`, `distogram_atom: Distogram` — eval-mode Distogram helpers.
- `mini_batch: Mapping[...]` — single-item batch of `_N_KEEP=16` residues.

### Test body

1. Convert `mini_batch` to a `ProteinBatch` via `_to_protein_batch([mini_batch])`.
2. Construct `Adam(model.parameters(), lr=1e-4)`.
3. Call `train_step(batch, model, tcfg, distogram_res, distogram_atom, optimizer, device="cpu")`.
4. Apply the same per-submodule bucket check as test 1.

### Why this test is distinct from test 1

- Exercises `featurize_batch` + `apply_conditioning_dropout` → true production featurization.
- Confirms that the loss weights, gamma schedule, and masking in `train_step()` do not silently kill
  any gradient paths.
- If `train_step()` is ever refactored (e.g. loss term removed), this test will catch regressions
  that the inline test would not.

---

## Helper — `_assert_submodule_grads`

A small private helper function defined once in each test file (not shared via conftest, to keep
files self-contained):

```python
def _assert_submodule_grads(model: MainTrunk) -> None:
    buckets: dict[str, list[torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            buckets.setdefault(name.split(".")[0], []).append(param.grad)
    for prefix, grads in buckets.items():
        assert any(torch.isfinite(g).all() and g.abs().max() > 0 for g in grads), (
            f"submodule '{prefix}' has no finite nonzero gradients"
        )
```

This is intentionally not a pytest fixture (it takes no inputs, returns nothing, is a pure
assertion utility).

---

## What is NOT tested

- Gradient *magnitude* correctness (that is a numerical property of the loss, not the graph).
- Gradient clipping behavior (already tested elsewhere in `test_train_loop.py`).
- DDP gradient synchronization (separate concern).
- The exact loss value (covered by separate loss-function unit tests).

---

## Conventions followed

- Module-level `def test_*()` functions, no class grouping.
- `einops.rearrange` used for tensor reshaping (no `.view`/`.reshape`).
- `jaxtyping` annotations on the helper function.
- `torch.manual_seed` already set at module level in both files.

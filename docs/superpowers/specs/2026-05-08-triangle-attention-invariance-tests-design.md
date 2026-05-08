# Triangle Attention Invariance Tests — Design

**Date:** 2026-05-08  
**File to modify:** `pallatom/tests/architecture/test_pair_update.py`

## Goal

Add six tests that demonstrate, in a layered narrative, that Triangle Attention (via `PairUpdate`) is invariant to rigid-body transformations of the input 3D coordinates. The tests form a chain: distances are invariant → RBF encoding is invariant → full `PairUpdate` output is invariant.

## Context

`PairUpdate` takes pair embeddings `z [B, N_res, N_res, C]` and residue center positions `r_center [B, N_res, 3]`. In step 1 it computes pairwise Euclidean distances `d_ij = ||r_i - r_j||`. Because Euclidean distance is preserved under translation and rotation, `d_ij` — and everything downstream of it (RBF bias, Triangle Attention) — must be invariant to these transforms.

The Triangle Attention sub-modules (`TriangleAttentionStartingNodeWithBias`, `TriangleAttentionEndingNodeWithBias`) receive `z` and `b_ij` (not coordinates directly), so the invariance is demonstrated at the `PairUpdate` level, with intermediate layers peeled back to show why it holds.

## New Helpers

Two typed helper functions added near the top of the test file, after existing helpers:

```python
@jaxtyped(typechecker=beartype)
def compute_dij(
    r: Float[torch.Tensor, "B N_res 3"],
) -> Float[torch.Tensor, "B N_res N_res"]:
    diff = rearrange(r, "b n d -> b n 1 d") - rearrange(r, "b n d -> b 1 n d")
    return diff.norm(dim=-1)

@jaxtyped(typechecker=beartype)
def random_rotation() -> Float[torch.Tensor, "3 3"]:
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if Q.det() < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

@jaxtyped(typechecker=beartype)
def apply_rotation(
    r: Float[torch.Tensor, "B N_res 3"],
    R: Float[torch.Tensor, "3 3"],
) -> Float[torch.Tensor, "B N_res 3"]:
    return einsum(r, R, "b n d, d e -> b n e")
```

No new fixtures needed — existing `r_center`, `z`, `d`, `rbf`, and `pair_update` fixtures cover all six tests.

## Six Tests in Three Layers

### Layer 1 — Distance computation (pure geometry, no modules)

**`test_distance_translation_invariant(r_center)`**  
Shift all positions by a random `[1, 1, 3]` translation. Compute `d_ij` before and after. Assert `allclose(atol=1e-5)`. Verifies the property at the most fundamental level, before any learned parameters are involved.

**`test_distance_rotation_invariant(r_center)`**  
Apply a random SO(3) rotation (via QR decomposition). Compute `d_ij` before and after. Assert `allclose(atol=1e-5)`.

### Layer 2 — RBF encoding (verifies invariant distances → invariant bias)

**`test_rbf_translation_invariant(rbf, r_center)`**  
Compute `d_orig` and `d_shift` (from translated positions). Feed both into `rbf`. Assert outputs are `allclose(atol=1e-5)`. Confirms `TransformRBF` preserves invariance (it is a deterministic function of distance, so this should hold analytically).

**`test_rbf_rotation_invariant(rbf, r_center)`**  
Same but with rotated positions.

### Layer 3 — PairUpdate end-to-end

**`test_pair_update_translation_invariant(pair_update, z, r_center)`**  
Call `pair_update(z, r_center)` and `pair_update(z, r_center + t)` under `torch.no_grad()`. Assert `allclose(atol=1e-5)`.

**`test_pair_update_rotation_invariant(pair_update, z, r_center)`**  
Call `pair_update(z, r_center)` and `pair_update(z, apply_rotation(r_center, R))`. Assert `allclose(atol=1e-5)`.

## Tolerance and Determinism

- `atol=1e-5` throughout — appropriate for float32 accumulation through RBF projection + multi-head attention.
- `pair_update` fixture already sets `dropout=0.0`; no stochasticity in eval mode.
- `torch.manual_seed(42)` at file top covers weight initialization.
- `random_rotation()` called inside each test (not a fixture) so the random matrix is regenerated per run but the model weights are stable.

## Conventions Followed

- Flat module-level `def test_...` functions (no classes).
- `jaxtyped` + `beartype` on all new helpers.
- `einops.einsum` and `einops.rearrange` (no `@`, no `view`/`unsqueeze`).
- `torch.no_grad()` context in all forward-pass-only tests.

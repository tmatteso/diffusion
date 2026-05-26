# Design: Extract `compute_beta` from `AtomTransformer` and add hard tests

**Date:** 2026-05-26
**Status:** Approved

---

## Background

`AtomTransformer.forward` (Algorithm 7, line 1) computes a sliding-window attention
bias `beta` inline. The calculation is ~40 lines of non-trivial window arithmetic
that currently has no direct test coverage. Extracting it to a named, jaxtyped
module-level function makes it independently testable and self-documenting.

---

## What `compute_beta` does

For each atom pair `(l, m)` in the sparse neighbour set, `compute_beta` returns:

- `0.0` — if at least one sliding-window centre `c` admits both `l` as a query
  (`|l − centre_c| < n_queries / 2`) **and** `m` as a key (`|m − centre_c| < n_keys / 2`),
  **and** `valid_mask[b, l, k]` is `True`.
- `-1e10` — everywhere else (out-of-window or padding slot).

Centre positions: `c * n_queries + (n_queries / 2 − 0.5)` for `c = 0, 1, …, ⌈N / n_queries⌉ − 1`.

---

## Section 1: the extracted function

### Signature

```python
@jaxtyped(typechecker=beartype)
def compute_beta(
    neighbor_idx: Int[torch.Tensor, "N K"],
    valid_mask:   Bool[torch.Tensor, "B N K"],
    n_queries: int,
    n_keys: int,
    ref: Float[torch.Tensor, "B N c"],
) -> Float[torch.Tensor, "B N K"]:
```

- `neighbor_idx` — sparse neighbour indices `[N, K]`.
- `valid_mask` — True where slot is a real atom `[B, N, K]`.
- `n_queries` — query-window full-width in atoms.
- `n_keys` — key-window full-width in atoms.
- `ref` — any float tensor whose `.device` and `.dtype` are used for the output
  (typically `q`, the query atom embedding). The shape `"B N c"` enforces it is
  a 3-D float tensor; only device and dtype are read from it.

### Placement

Module-level in `pallatom/architecture/atom_transformers.py`, placed directly after
`build_sparse_pairs` and before the `AtomTransformer` class definition.

### Export

Add `compute_beta` to the names imported in the test file alongside the existing
`build_sparse_pairs`, `AtomTransformer`, etc.

### Caller

`AtomTransformer.forward` replaces the entire inline beta block with:

```python
beta = compute_beta(neighbor_idx, valid_mask, self.n_queries, self.n_keys, q)
```

---

## Section 2: the hard tests

All tests live in `pallatom/tests/architecture/test_atom_transformers.py`.

### `compute_beta` unit tests

These tests import and call `compute_beta` directly. They prove correctness of the
window arithmetic, not just shape and finiteness.

| Test | Assertion |
|------|-----------|
| `test_compute_beta_output_shape` | Output shape is `(B, N, K)` |
| `test_compute_beta_values_are_binary` | Every element is exactly `0.0` or `-1e10`; no intermediate values |
| `test_compute_beta_zero_for_near_atoms` | A pair of adjacent atoms sharing a window centre produces `0.0` |
| `test_compute_beta_large_neg_for_far_atoms` | Atoms many windows apart produce `-1e10` |
| `test_compute_beta_valid_mask_forces_neg` | `beta = -1e10` wherever `valid_mask = False`, even if the pair is geometrically in-window |
| `test_compute_beta_boundary_exclusive` | An atom at exactly `half_q` distance from its nearest centre is **excluded** (strict `<`, not `≤`) |
| `test_compute_beta_cross_window_centre_admits_pair` | A pair that misses one centre but falls inside an adjacent centre gets `0.0` (validates multi-centre accumulation) |
| `test_compute_beta_all_zero_when_n_leq_n_queries` | When `N ≤ n_queries` all valid pairs are in-window — no `-1e10` among valid-mask entries |
| `test_compute_beta_wrong_shape_raises` | Passing a 2-D `neighbor_idx` triggers `TypeCheckError` |

The boundary and cross-window tests are the hardest: they would catch off-by-one
errors in the centre-placement formula `c * n_queries + (half_q − 0.5)`.

### `AtomTransformer.forward` block-isolation test

**`test_atom_transformer_block_isolation`**

**Setup:** `n_queries = n_keys = 4`, `N = 8`. With equal query and key half-widths,
the two atom blocks `[0..3]` and `[4..7]` map to disjoint window centres and share
no in-window pairs. Use a dense `neighbor_idx` (`[N, N]` with `arange(N)` repeated)
and all-True `valid_mask` `[B=1, N, N]` so every pair is structurally reachable.

**Procedure:**

1. Run `AtomTransformer.forward` with random `q`, `c`, `p` → `out_original`.
2. Clone `q` and `c`; replace block-1 slices (`[:, 4:, :]`) with fresh random values.
3. Run `AtomTransformer.forward` again → `out_scrambled`.
4. Assert `out_original[:, :4, :]` equals `out_scrambled[:, :4, :]` (exact match,
   since `exp(-1e10)` underflows to `0.0` in float32).

**What it proves:** Atoms outside a window block contribute exactly zero attention
weight, making sequence-local attention equivalent to independent self-attention
within each rectangular diagonal block. This rules out silent cross-block information
leakage from buggy beta computation.

---

## Non-goals

- No changes to `DiffusionTransformer`, `AtomFeatureEncoder`, or `AtomAttentionDecoder`.
- No new public API surface beyond adding `compute_beta` to the module namespace.
- No refactoring of the window-centre formula itself — only extraction.

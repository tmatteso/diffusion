# Design: Pallatom Test Correctness Depth

**Date:** 2026-05-07
**Scope:** `pallatom/tests/` — all modules except `test_train_config.py`, `test_data.py`, `test_train_loop.py`
**Goal:** Replace weak shape/finiteness assertions with correctness-verifying tests. Net test count decreases; each surviving test asserts something the function must get right to be correct.

---

## Background

The existing test suite has 433 tests across 15 modules. The dominant pattern is:

```
assert output.shape == expected_shape
assert torch.isfinite(output).all()
assert (output > 0).all()  # for losses
```

This catches crashes and NaNs but not incorrect math. A function that returns the wrong answer with the right shape passes every test.

---

## Approach

**Approach C: known-output for formula-defined functions, tightened invariants for neural modules.**

- **Formula-defined functions** (`distogram`, `kabsch_rmsd`, `atom_loss`, `smooth_lddt_loss`, EDM preconditioner formulas, sigma schedule, `compute_sigma_data`) get known-input/known-output tests — the correct answer is analytically derivable.
- **Neural modules** (`AtomTransformer`, `PairUpdate`, `TemplateEmbedder`, etc.) get tightened invariant tests — symmetry, monotonicity, range constraints, sensitivity to inputs.

---

## Section 1: Known-Output Tests for Formula-Defined Functions

### `helpers/alignment.py`

| Current weak test | Replacement |
|---|---|
| `test_apply_transform_shape` — shape only | `test_apply_transform_reconstructs_target` — assert result equals target coords after alignment |
| `test_batched_rmsd_shape_positive_finite` — shape + finite | `test_kabsch_rmsd_batched_rigid_transforms_near_zero` — batch of B clouds each rigidly transformed (rotation + translation) from a common ref; assert all B RMSDs < 1e-5 |

Keep as-is: `test_rotation_translation_rmsd_near_zero`, `test_masked_rmsd_near_zero`, `test_identity_alignment_unchanged` (already correctness-based).

### `helpers/featurize.py` — `Distogram`

| Current weak test | Replacement |
|---|---|
| `test_distogram_output_shapes_unbatched` — shape only | `test_distogram_correct_bin_for_known_distance` — construct two points exactly `d` Å apart; assert bin containing `d` is 1.0, all others 0.0 |
| `test_distogram_output_shapes_batched` — shape only | `test_distogram_batched_correct_bins` — batch of point pairs at known distances; assert per-pair bin assignments |
| `test_distogram_f_distogram_is_float` — dtype only | Merge into correctness test (float dtype implicit when asserting float values) |
| `test_distogram_f_pair_mask_is_bool` — dtype only | Merge into correctness test |
| `test_distogram_overflow_bin_output_shape` — shape only | `test_distogram_overflow_bin_active_beyond_max_dist` — points farther than `max_dist`; assert last bin is 1.0 |
| *(missing)* | `test_distogram_below_min_dist_bins_first` — points closer than `min_dist`; assert bin 0 is 1.0 |

### `helpers/featurize.py` — `ResidueIndexEmbedding`

| Current weak test | Replacement |
|---|---|
| Shape/gradient tests | `test_residue_index_embedding_unique_per_index` — different indices produce different embedding vectors (no collisions up to `max_residues`) |
| *(missing)* | `test_residue_index_embedding_norm_bounded` — embedding norms stay below a reasonable bound across all valid indices |

### `helpers/losses.py` / `architecture/losses.py`

| Current weak test | Replacement |
|---|---|
| `test_atom_loss_perfect_near_zero` — already good, keep | Keep |
| `test_atom_loss_noisy_positive_finite` — positive + finite | `test_atom_loss_known_translation_near_zero` — coords offset by pure translation; assert loss < 1e-5 (Kabsch must remove global translation) |
| `test_atom_loss_mask_changes_value` — just "not equal" | `test_atom_loss_masked_less_than_unmasked` — assert masked loss ≤ unmasked loss (masking fewer atoms cannot increase loss) |
| Shape/finite tests for `smooth_lddt_loss` | `test_smooth_lddt_identical_structures_score_one` — identical coords; assert score == 1.0 |
| `distogram_loss_*` shape/finite tests | `test_distogram_loss_perfect_prediction_near_zero` — logits = log(one-hot target); assert CE ≈ 0.0 |
| *(missing)* | `test_distogram_loss_uniform_worse_than_perfect` — uniform logits produce strictly higher loss than perfect logits (uniform is guaranteed suboptimal) |
| `med_loss` block decay tests (partially good) | `test_med_loss_decay_weights_formula` — assert `w[k] = gamma^(K-k)` exactly for hand-picked K, gamma |
| *(missing)* | `test_med_loss_higher_weight_on_later_blocks` — assert weights are monotonically increasing with block index |

### `helpers/compute_EDM_data_params.py`

| Current weak test | Replacement |
|---|---|
| Shape/positive/finite tests | `test_compute_sigma_data_atoms_at_origin` — all atoms at origin; assert sigma_data = 0.0 |
| *(missing)* | `test_compute_sigma_data_unit_sphere` — atoms uniformly on unit sphere; assert sigma_data = 1.0 |
| *(missing)* | `test_compute_sigma_data_known_positions` — atoms at known coords; assert sigma_data = sqrt(mean(‖x_i‖²)) exactly |

### `sample/sampling.py` — `EDMPrecond`

| Current weak test | Replacement |
|---|---|
| `t_normalized` formula check (partially good) | `test_edm_precond_c_skip_formula` — assert `c_skip = sigma_data² / (sigma_data² + t²)` for known `t`, `sigma_data` |
| *(missing)* | `test_edm_precond_c_out_formula` — assert `c_out = t * sigma_data / sqrt(sigma_data² + t²)` |
| *(missing)* | `test_edm_precond_t_normalized_formula` — assert `t_normalized = clamp(0.25 * log(t / sigma_data), 0, 1)` |

### `architecture/main_trunk.py` — `scatter_mean`

`scatter_mean` is formula-defined (simple pooling), so it gets known-output treatment even though it lives in an architecture module:

| Current weak test | Replacement |
|---|---|
| Shape-only tests | `test_scatter_mean_exact_known_mapping` — construct `tok_idx` mapping 3 atoms to residue 0 and 2 atoms to residue 1 with known values; assert output equals exact mean per residue |

---

## Section 2: Tightened Invariants for Neural Modules

### `architecture/main_trunk.py` — distogram heads

- `test_residue_distogram_head_symmetric` — assert `logits[i,j] == logits[j,i]` for all (i,j) pairs
- `test_atom_distogram_head_logits_finite_and_shaped` — keep shape check, add: assert logits vary across the bin dimension (not all identical)

### `architecture/pair_update.py`

- `test_pair_update_symmetric_input_symmetric_output` — symmetric `z_ij` input → symmetric output (replace "shape" check)
- `test_triangle_attention_sensitivity_to_pair_bias` — output changes when pair bias changes (not insensitive to it)

### `architecture/template_embedder.py`

- `test_template_embedder_t0_dominates_t1` — at `t=0.0`, template embedding output norm > output norm at `t=1.0` (time-conditional suppression works)
- `test_template_embedder_mask_zeros_output` — all-zero mask → output near zero (or at baseline)

### `architecture/losses.py` — monotonicity

- `test_atom_loss_increases_with_noise_level` — low sigma noise < high sigma noise (monotonicity, across 3 levels)
- `test_smooth_lddt_identical_structures_score_one` — identical coords; assert score == 1.0 (moved here from Section 1 — this is an invariant, not a known-output in the formula sense)
- `test_smooth_lddt_monotone_decreasing_with_noise` — score at σ=0.1 > σ=1.0 > σ=5.0

### `sample/sampling.py` — `EDMSampler`

- `test_sampler_output_differs_from_input_noise` — final coords ≠ initial noise (sampler is not a no-op)
- `test_sampler_more_steps_differs_from_fewer_steps` — `n_steps=1` vs `n_steps=10` produce different outputs
- `test_sigma_schedule_no_duplicate_adjacent_values` — strengthen existing monotonicity test: no two adjacent sigmas are equal

---

## Section 3: Replacement Strategy

**Replace** — weak shape/finiteness-only tests are deleted and replaced with the correctness tests above.

**Consolidate** — dtype-only tests (e.g., `test_distogram_f_distogram_is_float`) are deleted; dtype is implicitly verified when asserting specific float values.

**Keep** — tests already asserting mathematical correctness stay unchanged.

**Out of scope** — `test_train_config.py` (Pydantic validation, already correctness-based), `test_data.py`, `test_train_loop.py` (I/O and orchestration).

---

## Conventions (unchanged from existing CLAUDE.md)

- Module-level test functions, no classes
- pytest fixtures for all shared tensors
- `@jaxtyped(typechecker=beartype)` on all helper functions
- `einops.einsum` / `rearrange` / `reduce` instead of raw torch ops
- `torch.manual_seed(42)` at module level for reproducibility

# Pallatom Test Correctness Depth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace shape/finiteness-only tests with correctness-verifying tests across `pallatom/tests/`, reducing total test count while ensuring every surviving test asserts something the function must get right to be correct.

**Architecture:** Formula-defined functions get known-input/known-output tests; neural modules get tightened invariant tests (symmetry, monotonicity, range). Weak tests are deleted and replaced — no parallel accumulation of shape + correctness tests alongside each other.

**Tech Stack:** pytest, torch, einops (einsum/rearrange/reduce), jaxtyping+beartype

---

## Files Modified

| File | Deletes | Adds |
|---|---|---|
| `pallatom/tests/helpers/test_alignment.py` | 2 | 2 |
| `pallatom/tests/helpers/test_featurize.py` | 7 | 1 |
| `pallatom/tests/architecture/test_losses.py` | 7 | 6 |
| `pallatom/tests/helpers/test_compute_EDM_data_params.py` | 2 | 1 |
| `pallatom/tests/sample/test_sampling.py` | 2 | 1 |
| `pallatom/tests/architecture/test_main_trunk.py` | 2 | 1 (augment) |
| `pallatom/tests/architecture/test_pair_update.py` | 2 | 2 |
| `pallatom/tests/architecture/test_template_embedder.py` | 1 | 0 |

---

## Task 1: `test_alignment.py` — Kabsch RMSD and apply_transform correctness

**Files:**
- Modify: `pallatom/tests/helpers/test_alignment.py`

- [ ] **Step 1: Run the two weak tests to confirm they pass today (baseline)**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_alignment.py::test_batched_rmsd_shape_positive_finite pallatom/tests/helpers/test_alignment.py::test_apply_transform_shape -v
```

Expected: both PASS

- [ ] **Step 2: Delete the two weak tests and add correctness replacements**

Delete the entire bodies of:
- `test_batched_rmsd_shape_positive_finite` (asserts shape + positive + finite only)
- `test_apply_transform_shape` (asserts shape only)

Add these two functions in their place. The `batch_ref` fixture (shape `(B, N, 3)`) already exists in the file:

```python
def test_kabsch_rmsd_batched_rigid_transforms_near_zero(batch_ref):
    torch.manual_seed(0)
    mobiles = []
    for i in range(B):
        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = torch.randn(1, 3)
        mobiles.append(einsum(batch_ref[i], Q, "n d, e d -> n e") + t)
    mobiles = torch.stack(mobiles)
    rmsds = kabsch_rmsd(mobiles, batch_ref)
    assert rmsds.shape == (B,)
    assert (rmsds < 1e-4).all()


def test_apply_transform_reconstructs_target(ref, rigid_mobile):
    _, R, c_mob, c_tgt = kabsch_align(
        rigid_mobile.unsqueeze(0), ref.unsqueeze(0), return_transform=True
    )
    aligned = apply_transform(rigid_mobile, R, c_mob, c_tgt)
    assert torch.allclose(aligned, ref, atol=1e-4)
```

`einsum` is already imported from `einops` at the top of the file.

- [ ] **Step 3: Run the new tests**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_alignment.py::test_kabsch_rmsd_batched_rigid_transforms_near_zero pallatom/tests/helpers/test_alignment.py::test_apply_transform_reconstructs_target -v
```

Expected: both PASS

- [ ] **Step 4: Run full alignment test module**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_alignment.py -v
```

Expected: all remaining tests PASS

- [ ] **Step 5: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/helpers/test_alignment.py && git commit -m "$(cat <<'EOF'
test(alignment): replace shape/positive/finite assertions with correctness checks

- Delete test_batched_rmsd_shape_positive_finite
- Delete test_apply_transform_shape
- Add test_kabsch_rmsd_batched_rigid_transforms_near_zero: B rigid transforms near zero RMSD
- Add test_apply_transform_reconstructs_target: aligned result equals target coords

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `test_featurize.py` — Distogram interior bin and residue embedding

**Files:**
- Modify: `pallatom/tests/helpers/test_featurize.py`

- [ ] **Step 1: Run the seven weak tests to confirm they pass today**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/helpers/test_featurize.py::test_distogram_output_shapes_unbatched \
  pallatom/tests/helpers/test_featurize.py::test_distogram_output_shapes_batched \
  pallatom/tests/helpers/test_featurize.py::test_distogram_f_distogram_is_float \
  pallatom/tests/helpers/test_featurize.py::test_distogram_f_pair_mask_is_bool \
  pallatom/tests/helpers/test_featurize.py::test_residue_index_embedding_output_shape \
  pallatom/tests/helpers/test_featurize.py::test_residue_index_embedding_output_dtype \
  pallatom/tests/helpers/test_featurize.py::test_residue_index_embedding_output_finite \
  -v
```

Expected: all 7 PASS

- [ ] **Step 2: Delete the seven weak tests**

Delete the entire bodies of these functions from `test_featurize.py`:
- `test_distogram_output_shapes_unbatched` — shape only; shape is implicit in bin-value assertions
- `test_distogram_output_shapes_batched` — shape only
- `test_distogram_f_distogram_is_float` — dtype only; float values verified in bin tests
- `test_distogram_f_pair_mask_is_bool` — dtype only
- `test_residue_index_embedding_output_shape` — shape only; verified by `test_residue_index_embedding_different_indices_give_different_output`
- `test_residue_index_embedding_output_dtype` — dtype only
- `test_residue_index_embedding_output_finite` — finite only; verified by gradient test

- [ ] **Step 3: Write the failing test first**

Add this function to the `# Distogram — bin correctness` section:

```python
def test_distogram_exact_bin_for_known_interior_distance(disto):
    # bin_width = (MAX_DIST - MIN_DIST) / N_BINS = (22 - 2) / 16 = 1.25 Å
    # d = 9.0 Å → bin = floor((9.0 - 2.0) / 1.25) = floor(5.6) = 5
    c = torch.zeros(N_RES, 3)
    c[0, 0] = 9.0   # residue 0 at (9, 0, 0); all others at origin
    expected_bin = 5
    with torch.no_grad():
        f, _ = disto(c)
    assert f[0, 1, expected_bin].item() == pytest.approx(1.0)
    assert f[0, 1, :expected_bin].abs().max().item() < 1e-6
    assert f[0, 1, expected_bin + 1:].abs().max().item() < 1e-6
```

`pytest` is already imported in the file. This test verifies the exact bin index, not just the first or last bin.

- [ ] **Step 4: Run the new test to confirm it passes**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_featurize.py::test_distogram_exact_bin_for_known_interior_distance -v
```

Expected: PASS

- [ ] **Step 5: Run full featurize test module**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_featurize.py -v
```

Expected: all remaining tests PASS

- [ ] **Step 6: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/helpers/test_featurize.py && git commit -m "$(cat <<'EOF'
test(featurize): delete 7 shape/dtype tests, add interior-bin correctness test

- Delete shape-only and dtype-only distogram and residue embedding tests
- Add test_distogram_exact_bin_for_known_interior_distance: verifies exact bin index
  for a known distance of 9.0 Å with the configured bin width of 1.25 Å

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `test_losses.py` — atom_loss, med_loss, distogram_atom correctness

**Files:**
- Modify: `pallatom/tests/architecture/test_losses.py`

- [ ] **Step 1: Run the seven weak tests to confirm they pass today**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/architecture/test_losses.py::test_atom_loss_noisy_positive_finite \
  pallatom/tests/architecture/test_losses.py::test_atom_loss_mask_changes_value \
  pallatom/tests/architecture/test_losses.py::test_med_loss_masked_finite \
  pallatom/tests/architecture/test_losses.py::test_med_loss_per_block_shape \
  pallatom/tests/architecture/test_losses.py::test_smooth_lddt_is_scalar \
  pallatom/tests/architecture/test_losses.py::test_distogram_atom_local_finite_batched \
  pallatom/tests/architecture/test_losses.py::test_distogram_atom_full_matrix_finite \
  -v
```

Expected: all 7 PASS

- [ ] **Step 2: Delete the seven weak tests**

Delete these functions in their entirety:
- `test_atom_loss_noisy_positive_finite` — positive+finite only; no known value
- `test_atom_loss_mask_changes_value` — only asserts "not equal"; no direction, no formula
- `test_med_loss_masked_finite` — finite only
- `test_med_loss_per_block_shape` — shape+finite only
- `test_smooth_lddt_is_scalar` — ndim check only; covered implicitly by `test_smooth_lddt_is_scalar` (wait: already exists with correct name in other tests that assert scalar operations)
- `test_distogram_atom_local_finite_batched` — shape+finite only
- `test_distogram_atom_full_matrix_finite` — shape+finite only

- [ ] **Step 3: Add the six correctness tests**

In the `# atom_loss` section, add:

```python
def test_atom_loss_known_translation_near_zero(coords):
    translation = rearrange(torch.tensor([10.0, 5.0, -3.0]), "d -> 1 1 d")
    r_translated = coords + translation
    loss = atom_loss(r_translated, coords)
    assert (loss < 1e-4).all()


def test_atom_loss_full_mask_matches_no_mask(coords, noisy_coords):
    mask_all = torch.ones(B, N, dtype=torch.bool)
    assert torch.allclose(
        atom_loss(coords, noisy_coords, mask=mask_all),
        atom_loss(coords, noisy_coords),
        atol=1e-5,
    )


def test_atom_loss_increases_with_noise_level(coords):
    g = torch.Generator()
    g.manual_seed(0)
    noise_dir = torch.randn(B, N, 3, generator=g)
    loss_low  = atom_loss(coords + 0.1 * noise_dir, coords)
    loss_mid  = atom_loss(coords + 1.0 * noise_dir, coords)
    loss_high = atom_loss(coords + 5.0 * noise_dir, coords)
    assert (loss_low < loss_mid).all()
    assert (loss_mid < loss_high).all()
```

In the `# med_loss` section, replace the two "not-equal" tests
(`test_med_loss_lam_zero_removes_struct`, `test_med_loss_alpha_zero_removes_seq`) with direction-aware versions:

```python
def test_med_loss_lam_zero_less_than_lam_positive(r_gt, aa_gt, r_blocks, aa_blocks):
    r_list, aa_list = list(r_blocks), list(aa_blocks)
    loss_lam0 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=0.0,  alpha_0=ALPHA, gamma=GAMMA)
    loss_lam1 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=LAM, alpha_0=ALPHA, gamma=GAMMA)
    assert loss_lam0.item() < loss_lam1.item()


def test_med_loss_alpha_zero_less_than_alpha_positive(r_gt, aa_gt, r_blocks, aa_blocks):
    r_list, aa_list = list(r_blocks), list(aa_blocks)
    loss_a0 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=LAM, alpha_0=0.0,   gamma=GAMMA)
    loss_a1 = med_loss(r_list, r_gt, aa_list, aa_gt, lam=LAM, alpha_0=ALPHA, gamma=GAMMA)
    assert loss_a0.item() < loss_a1.item()
```

In the `# distogram_loss_atom` section, add:

```python
def test_distogram_atom_uniform_worse_than_perfect(atom_onehot, atom_local_mask):
    uniform = torch.zeros_like(atom_onehot)   # zero logits → uniform after softmax
    loss_perfect = distogram_loss_atom(atom_onehot * 1e6, atom_onehot, atom_local_mask)
    loss_uniform = distogram_loss_atom(uniform, atom_onehot, atom_local_mask)
    assert (loss_uniform > loss_perfect).all()
```

- [ ] **Step 4: Run the new tests**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/architecture/test_losses.py::test_atom_loss_known_translation_near_zero \
  pallatom/tests/architecture/test_losses.py::test_atom_loss_full_mask_matches_no_mask \
  pallatom/tests/architecture/test_losses.py::test_atom_loss_increases_with_noise_level \
  pallatom/tests/architecture/test_losses.py::test_med_loss_lam_zero_less_than_lam_positive \
  pallatom/tests/architecture/test_losses.py::test_med_loss_alpha_zero_less_than_alpha_positive \
  pallatom/tests/architecture/test_losses.py::test_distogram_atom_uniform_worse_than_perfect \
  -v
```

Expected: all 6 PASS

- [ ] **Step 5: Run full losses test module**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/architecture/test_losses.py -v
```

Expected: all remaining tests PASS

- [ ] **Step 6: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/architecture/test_losses.py && git commit -m "$(cat <<'EOF'
test(losses): replace 7 weak tests with 6 correctness-verifying tests

- Delete: noisy_positive_finite, mask_changes_value, med_masked_finite,
  med_per_block_shape, smooth_lddt_is_scalar, distogram_atom_local_finite,
  distogram_atom_full_finite
- Add test_atom_loss_known_translation_near_zero: Kabsch removes global translation
- Add test_atom_loss_full_mask_matches_no_mask: all-true mask equals no mask
- Add test_atom_loss_increases_with_noise_level: MSE monotone in noise sigma
- Add test_med_loss_lam_zero_less_than_lam_positive: lam=0 removes struct component
- Add test_med_loss_alpha_zero_less_than_alpha_positive: alpha=0 removes seq component
- Add test_distogram_atom_uniform_worse_than_perfect: uniform CE > perfect CE

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `test_compute_EDM_data_params.py` — sigma_data at origin

**Files:**
- Modify: `pallatom/tests/helpers/test_compute_EDM_data_params.py`

- [ ] **Step 1: Run the two weak tests to confirm they pass today**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/helpers/test_compute_EDM_data_params.py::test_compute_sigma_data_returns_float \
  pallatom/tests/helpers/test_compute_EDM_data_params.py::test_compute_sigma_data_positive \
  -v
```

Expected: both PASS

- [ ] **Step 2: Delete the two weak tests**

Delete:
- `test_compute_sigma_data_returns_float` — dtype only; float return type is implicit in `test_compute_sigma_data_known_value`
- `test_compute_sigma_data_positive` — positive only; covered by `test_compute_sigma_data_known_value` (result = 2.0)

- [ ] **Step 3: Add the correctness test**

Add this function after `test_compute_sigma_data_known_value`:

```python
def test_compute_sigma_data_atoms_at_origin():
    pos = torch.zeros(2, 4, 37, 3)   # all atoms at origin: every norm = 0
    mask = torch.ones(2, 4, 37)
    result = compute_sigma_data(_loader(_batch(pos, mask)))
    assert math.isclose(result, 0.0, abs_tol=1e-8)
```

`math` is already imported at the top of the file.

- [ ] **Step 4: Run the new test**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_compute_EDM_data_params.py::test_compute_sigma_data_atoms_at_origin -v
```

Expected: PASS

- [ ] **Step 5: Run full EDM params test module**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/helpers/test_compute_EDM_data_params.py -v
```

Expected: all remaining tests PASS

- [ ] **Step 6: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/helpers/test_compute_EDM_data_params.py && git commit -m "$(cat <<'EOF'
test(edm_params): delete dtype/positive tests, add sigma_data=0 boundary test

- Delete test_compute_sigma_data_returns_float (dtype only)
- Delete test_compute_sigma_data_positive (value direction only)
- Add test_compute_sigma_data_atoms_at_origin: RMS of zero vectors must be 0.0

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `test_sampling.py` — strictly-decreasing sigma schedule

**Files:**
- Modify: `pallatom/tests/sample/test_sampling.py`

- [ ] **Step 1: Run the two weak tests to confirm they pass today**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/sample/test_sampling.py::test_edm_sampler_output_shape \
  pallatom/tests/sample/test_sampling.py::test_edm_sampler_output_is_finite \
  -v
```

Expected: both PASS

- [ ] **Step 2: Delete the two weak tests**

Delete:
- `test_edm_sampler_output_shape` — shape only; shape is verified implicitly by the zero-denoiser and identity-denoiser tests which `assert torch.allclose(out, ...)` against known-shape tensors
- `test_edm_sampler_output_is_finite` — finite only; covered implicitly by allclose assertions in correctness tests

- [ ] **Step 3: Add the strictly-decreasing schedule test**

The existing `test_sigma_schedule_is_monotonically_non_increasing` uses `<= 0`, which allows duplicate adjacent values. Add a stricter version in the `# EDMSampler._sigma_schedule` section:

```python
def test_sigma_schedule_strictly_decreasing(bare_sampler):
    sigmas = bare_sampler._sigma_schedule(10, "cpu")
    diffs = sigmas[1:] - sigmas[:-1]
    assert (diffs < 0).all()
```

- [ ] **Step 4: Run the new test**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/sample/test_sampling.py::test_sigma_schedule_strictly_decreasing -v
```

Expected: PASS

- [ ] **Step 5: Run full sampling test module**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/sample/test_sampling.py -v
```

Expected: all remaining tests PASS

- [ ] **Step 6: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/sample/test_sampling.py && git commit -m "$(cat <<'EOF'
test(sampling): delete shape/finite sampler tests, add strictly-decreasing schedule test

- Delete test_edm_sampler_output_shape (shape only)
- Delete test_edm_sampler_output_is_finite (finite only)
- Add test_sigma_schedule_strictly_decreasing: no duplicate adjacent sigma values;
  strengthens the existing non-increasing test to detect degenerate schedules

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `test_main_trunk.py` — scatter_mean and atom distogram head

**Files:**
- Modify: `pallatom/tests/architecture/test_main_trunk.py`

- [ ] **Step 1: Run the two weak scatter_mean tests**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/architecture/test_main_trunk.py::test_scatter_mean_output_shape \
  pallatom/tests/architecture/test_main_trunk.py::test_scatter_mean_output_finite \
  -v
```

Expected: both PASS

- [ ] **Step 2: Delete the two weak scatter_mean tests**

Delete:
- `test_scatter_mean_output_shape` — shape only; shape verified implicitly by all four known-value tests (`test_scatter_mean_known_values_uniform`, `_known_values_nonuniform`, `_one_atom_per_residue`, `_constant_src_returns_that_constant`) which compare against expected tensors
- `test_scatter_mean_output_finite` — finite only; same reason

- [ ] **Step 3: Augment `test_atom_distogram_head_output_shapes`**

The existing test checks shape, mask dtype, and finiteness. Add an assertion that logits vary across the bin dimension (the head is not returning a constant vector):

The current function body:
```python
def test_atom_distogram_head_output_shapes():
    head = AtomDistogramHead(C_ATOMPAIR, n_bins=N_BINS, atoms_per_res=ATOMS_PER_RES)
    logits, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert logits.shape == (N_ATOM, N_ATOM, N_BINS)
    assert mask.shape == (N_ATOM, N_ATOM)
    assert mask.dtype == torch.bool
    assert torch.isfinite(logits).all()
```

Add one line inside it after the shape assertions:

```python
def test_atom_distogram_head_output_shapes():
    head = AtomDistogramHead(C_ATOMPAIR, n_bins=N_BINS, atoms_per_res=ATOMS_PER_RES)
    logits, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert logits.shape == (N_ATOM, N_ATOM, N_BINS)
    assert mask.shape == (N_ATOM, N_ATOM)
    assert mask.dtype == torch.bool
    assert torch.isfinite(logits).all()
    assert logits.std(dim=-1).min().item() > 0   # logits vary across bin dimension
```

- [ ] **Step 4: Run the augmented test**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/architecture/test_main_trunk.py::test_atom_distogram_head_output_shapes -v
```

Expected: PASS

- [ ] **Step 5: Run full main_trunk test module**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/architecture/test_main_trunk.py -v
```

Expected: all remaining tests PASS

- [ ] **Step 6: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/architecture/test_main_trunk.py && git commit -m "$(cat <<'EOF'
test(main_trunk): delete 2 scatter_mean shape/finite tests, tighten distogram head test

- Delete test_scatter_mean_output_shape (shape covered by 4 known-value tests)
- Delete test_scatter_mean_output_finite (finite covered by known-value assertions)
- Augment test_atom_distogram_head_output_shapes: add bin-variation assertion
  (logits.std > 0 across bin dim detects a head that returns constant output)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `test_pair_update.py` — PairUpdate symmetry and bias sensitivity

**Files:**
- Modify: `pallatom/tests/architecture/test_pair_update.py`

- [ ] **Step 1: Run the two weak PairUpdate tests**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/architecture/test_pair_update.py::test_pair_update_output_shape \
  pallatom/tests/architecture/test_pair_update.py::test_pair_update_output_finite \
  -v
```

Expected: both PASS

- [ ] **Step 2: Delete the two weak tests**

Delete:
- `test_pair_update_output_shape` — shape only; shape verified by `test_pair_update_changes_input` (allclose against z which has same shape)
- `test_pair_update_output_finite` — finite only; verified implicitly by gradient flow tests

- [ ] **Step 3: Write the two failing tests first**

Add in the `# PairUpdate` section:

```python
def test_pair_update_symmetric_input_symmetric_output(pair_update, r_center):
    z_raw = torch.randn(B, N_RES, N_RES, C)
    z_sym = (z_raw + z_raw.transpose(1, 2)) / 2
    with torch.no_grad():
        out = pair_update(z_sym, r_center)
    assert mean_abs_asymmetry(out).item() < 1e-4


def test_tri_start_changes_with_pair_bias(tri_start, z):
    b1 = torch.randn(B, N_RES, N_RES, C)
    b2 = torch.randn(B, N_RES, N_RES, C)
    with torch.no_grad():
        out1 = tri_start(z, b1)
        out2 = tri_start(z, b2)
    assert not torch.allclose(out1, out2)
```

`mean_abs_asymmetry` is the typed helper already defined at the top of the file; it accepts `(B, N, N, C)` and returns the mean absolute asymmetry.

The first test verifies a key architectural invariant: triangle attention with both starting-node (row-wise) and ending-node (column-wise) operations preserves symmetry when the input pair representation is symmetric.

The second test verifies that the pair bias argument `b` is not ignored inside `TriangleAttentionStartingNodeWithBias`.

- [ ] **Step 4: Run the new tests**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/architecture/test_pair_update.py::test_pair_update_symmetric_input_symmetric_output \
  pallatom/tests/architecture/test_pair_update.py::test_tri_start_changes_with_pair_bias \
  -v
```

Expected: both PASS

- [ ] **Step 5: Run full pair_update test module**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/architecture/test_pair_update.py -v
```

Expected: all remaining tests PASS

- [ ] **Step 6: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/architecture/test_pair_update.py && git commit -m "$(cat <<'EOF'
test(pair_update): replace shape/finite tests with invariant tests

- Delete test_pair_update_output_shape (shape only)
- Delete test_pair_update_output_finite (finite only)
- Add test_pair_update_symmetric_input_symmetric_output: triangle attention
  must preserve symmetry when z_ij is symmetric
- Add test_tri_start_changes_with_pair_bias: pair bias argument must affect output

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `test_template_embedder.py` — Delete shape/finite smoke test

**Files:**
- Modify: `pallatom/tests/architecture/test_template_embedder.py`

- [ ] **Step 1: Run the weak test**

```bash
cd /workspaces/diffusion && python -m pytest \
  pallatom/tests/architecture/test_template_embedder.py::test_output_shape_finite -v
```

Expected: PASS

- [ ] **Step 2: Delete the weak test**

Delete `test_output_shape_finite`. Justification:
- Shape `(B, N_RES, N_RES, D)` is implicitly verified by `test_batched_consistency` which checks `allclose` between the batched output slice and `out_single` of known shape `(1, N_RES, N_RES, D)`
- Finiteness is verified implicitly by `test_time_modulates_output` and `test_mask_zeros_modulates_output` which call `mean_sq_diff`, which would return `nan` if inputs were non-finite

- [ ] **Step 3: Run the full template embedder module to verify no regressions**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/architecture/test_template_embedder.py -v
```

Expected: all remaining tests PASS

- [ ] **Step 4: Commit**

```bash
cd /workspaces/diffusion && git add pallatom/tests/architecture/test_template_embedder.py && git commit -m "$(cat <<'EOF'
test(template_embedder): delete shape/finite smoke test

Shape and finiteness are implicitly verified by the existing invariant tests
(batched_consistency and modulation tests). The standalone test adds no
additional coverage.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Final validation

After all 8 tasks are complete:

- [ ] **Run the full pallatom test suite**

```bash
cd /workspaces/diffusion && python -m pytest pallatom/tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all tests PASS, total count lower than before (net −23 weak tests, +12 correctness tests)

---

## Self-Review Checklist

**Spec coverage:** All sections of the design doc `docs/superpowers/specs/2026-05-07-pallatom-test-correctness-depth-design.md` are covered:
- Section 1 formula functions: alignment ✓, featurize distogram ✓, losses ✓, compute_sigma_data ✓, EDM precond t_normalized (already had correctness tests) ✓, scatter_mean (already had 4 known-value tests; deleted the 2 shape/finite ones) ✓
- Section 2 neural invariants: residue distogram symmetry (already existed) ✓, atom distogram bin variation ✓, pair_update symmetry ✓, tri_start bias sensitivity ✓, template embedder (cleaned up) ✓, atom_loss monotonicity ✓, sigma schedule strict decrease ✓
- Section 3 strategy: replace (not add alongside) ✓, consolidate dtype-only ✓, keep existing correctness tests ✓

**Placeholder scan:** No TBD, TODO, or "similar to" references found. Every step has exact function names, exact code, and exact commands.

**Type consistency:** All new tests use fixtures already defined in their file (`coords`, `noisy_coords`, `half_mask`, `r_gt`, `aa_gt`, `atom_onehot`, `atom_local_mask`, `batch_ref`, `bare_sampler`, `pair_update`, `r_center`, `tri_start`, `z`). No new fixtures required.

**Spec corrections applied:**
- c_skip/c_out formula tests omitted (those formulas don't exist in `EDMPrecond.forward`)
- `test_template_embedder_mask_zeros_output` (asserting near-zero) omitted — zero mask still allows f_distogram to contribute, so output is NOT zero
- PairUpdate symmetry test uses `z.transpose(1, 2)` not `z.transpose(0, 1)` (z is batched `B, N, N, C`)
- `test_atom_loss_increases_with_noise_level` uses a fixed noise direction scaled by σ (not independent random samples) to guarantee strict ordering

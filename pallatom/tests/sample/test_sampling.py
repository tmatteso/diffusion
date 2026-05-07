import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from einops import rearrange, reduce

from helpers.featurize import Distogram, FeaturizedBatch
from helpers.atom_utils import Protein
from sample.sampling import (
    ATOM5_TO_ATOM37,
    NATOM,
    AllAtomContext,
    TemplateContext,
    EDMPrecond,
    EDMSampler,
    atom5_to_atom37,
    build_AA_context,
    build_sampling_context,
    build_template_context,
)

torch.manual_seed(0)
np.random.seed(0)

N_RES     = 4
N_ATOM    = N_RES * NATOM   # 20
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def _make_trunk_mock() -> MagicMock:
    """MagicMock with identity denoiser behaviour: returns r_input unchanged."""
    mock = MagicMock()

    def _forward(batch):
        B_local = batch.r_input.shape[0]
        n_atom  = batch.r_input.shape[1]
        n_res   = int(batch.tok_idx.max().item()) + 1
        return (
            batch.r_input.clone(),
            torch.zeros(B_local, n_res, 20),
            torch.zeros(B_local, n_res, n_res, 38),
            torch.zeros(B_local, n_atom, 1, 38),
            [], [],
        )

    mock.side_effect = _forward
    return mock


def _make_zero_denoiser_mock() -> MagicMock:
    """MagicMock denoiser that always returns zeros."""
    mock = MagicMock()
    mock.side_effect = lambda r, sigma: torch.zeros_like(r)
    return mock


def _make_identity_denoiser_mock() -> MagicMock:
    """MagicMock denoiser that returns its input unchanged."""
    mock = MagicMock()
    mock.side_effect = lambda r, sigma: r.clone()
    return mock


def _make_half_denoiser_mock() -> MagicMock:
    """MagicMock denoiser that returns 0.5 × input (non-trivial, trajectory-dependent)."""
    mock = MagicMock()
    mock.side_effect = lambda r, sigma: r * 0.5
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def coords5() -> np.ndarray:
    return np.random.RandomState(1).randn(N_RES, 5, 3).astype(np.float32)


@pytest.fixture
def index_embedding() -> nn.Embedding:
    return nn.Embedding(256, 32)


@pytest.fixture
def context(index_embedding) -> FeaturizedBatch:
    return build_sampling_context(N_RES, index_embedding)


@pytest.fixture
def trunk_mock() -> MagicMock:
    return _make_trunk_mock()


@pytest.fixture
def edm_precond(trunk_mock, context) -> EDMPrecond:
    return EDMPrecond(trunk_mock, context, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)


@pytest.fixture
def edm_sampler(edm_precond) -> EDMSampler:
    return EDMSampler(edm_precond, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, S_churn=0.0)


@pytest.fixture
def bare_sampler() -> EDMSampler:
    """EDMSampler with a zero denoiser; used for _sigma_schedule tests only."""
    return EDMSampler(
        _make_zero_denoiser_mock(),
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        rho=7.0,
    )  # type: ignore[arg-type]


@pytest.fixture
def context_c_res_64() -> FeaturizedBatch:
    return build_sampling_context(N_RES, nn.Embedding(256, 64), c_res_embed=64)


@pytest.fixture
def edm_precond_custom_sigmas(context) -> EDMPrecond:
    from unittest.mock import MagicMock as _MM
    return EDMPrecond(_MM(), context, sigma_min=0.01, sigma_max=50.0)


@pytest.fixture
def identity_det_sampler() -> EDMSampler:
    return EDMSampler(  # type: ignore[arg-type]
        _make_identity_denoiser_mock(),
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        S_churn=0.0,
    )


@pytest.fixture
def identity_stoch_sampler_tmin_high() -> EDMSampler:
    return EDMSampler(  # type: ignore[arg-type]
        _make_identity_denoiser_mock(),
        sigma_min=SIGMA_MIN,
        sigma_max=SIGMA_MAX,
        S_churn=2.0,
        S_tmin=SIGMA_MAX * 10,
    )


# ---------------------------------------------------------------------------
# atom5_to_atom37 — output shapes
# ---------------------------------------------------------------------------

def test_atom5_to_atom37_x37_shape(coords5):
    x_37, _ = atom5_to_atom37(coords5)
    assert x_37.shape == (N_RES, 37, 3)


def test_atom5_to_atom37_mask37_shape(coords5):
    _, mask_37 = atom5_to_atom37(coords5)
    assert mask_37.shape == (N_RES, 37)


# ---------------------------------------------------------------------------
# atom5_to_atom37 — coordinate placement
# ---------------------------------------------------------------------------

def test_atom5_to_atom37_each_slot_lands_at_correct_atom37_index():
    # Give each atom5 slot a unique sentinel value so placement is unambiguous.
    coords_5 = np.zeros((N_RES, 5, 3), dtype=np.float32)
    for slot in range(5):
        coords_5[:, slot, :] = float(slot + 1)
    x_37, _ = atom5_to_atom37(coords_5)
    for slot, atom37_idx in enumerate(ATOM5_TO_ATOM37):
        assert np.allclose(x_37[:, atom37_idx, :], float(slot + 1)), (
            f"atom5 slot {slot} → atom37 slot {atom37_idx}: wrong coords"
        )


def test_atom5_to_atom37_unoccupied_atom37_slots_are_zero(coords5):
    x_37, _ = atom5_to_atom37(coords5)
    occupied = set(ATOM5_TO_ATOM37)
    for idx in range(37):
        if idx not in occupied:
            assert np.allclose(x_37[:, idx, :], 0.0), (
                f"atom37 slot {idx} should be zero (unoccupied)"
            )


# ---------------------------------------------------------------------------
# atom5_to_atom37 — mask handling
# ---------------------------------------------------------------------------

def test_atom5_to_atom37_mask_none_sets_occupied_slots_to_one(coords5):
    _, mask_37 = atom5_to_atom37(coords5, mask_5=None)
    for atom37_idx in ATOM5_TO_ATOM37:
        assert np.allclose(mask_37[:, atom37_idx], 1.0)


def test_atom5_to_atom37_explicit_mask_placed_at_correct_atom37_positions():
    rng      = np.random.RandomState(2)
    coords_5 = rng.randn(N_RES, 5, 3).astype(np.float32)
    mask_5   = rng.rand(N_RES, 5).astype(np.float32)
    _, mask_37 = atom5_to_atom37(coords_5, mask_5)
    for slot, atom37_idx in enumerate(ATOM5_TO_ATOM37):
        assert np.allclose(mask_37[:, atom37_idx], mask_5[:, slot])


def test_atom5_to_atom37_unoccupied_mask_slots_are_zero():
    coords_5 = np.ones((N_RES, 5, 3), dtype=np.float32)
    mask_5   = np.ones((N_RES, 5),    dtype=np.float32)
    _, mask_37 = atom5_to_atom37(coords_5, mask_5)
    occupied = set(ATOM5_TO_ATOM37)
    for idx in range(37):
        if idx not in occupied:
            assert np.allclose(mask_37[:, idx], 0.0)


# ---------------------------------------------------------------------------
# atom5_to_atom37 — type enforcement and edge cases
# ---------------------------------------------------------------------------

def test_atom5_to_atom37_rejects_wrong_second_dimension():
    with pytest.raises(Exception):
        atom5_to_atom37(np.zeros((N_RES, 4, 3), dtype=np.float32))   # 4 ≠ 5


def test_atom5_to_atom37_single_residue():
    coords_5 = np.random.randn(1, 5, 3).astype(np.float32)
    x_37, mask_37 = atom5_to_atom37(coords_5)
    assert x_37.shape == (1, 37, 3)
    assert mask_37.shape == (1, 37)


# ---------------------------------------------------------------------------
# build_sampling_context — return type
# ---------------------------------------------------------------------------

def test_build_sampling_context_returns_featurized_batch(context):
    assert isinstance(context, FeaturizedBatch)


# ---------------------------------------------------------------------------
# build_sampling_context — tensor shapes  (default batch_size=1)
# ---------------------------------------------------------------------------

def test_build_sampling_context_ref_pos_shape(context):
    assert context.ref_pos.shape == (1, N_ATOM, 3)


def test_build_sampling_context_ref_element_shape(context):
    assert context.ref_element.shape == (1, N_ATOM, 4)


def test_build_sampling_context_ref_space_uid_shape(context):
    assert context.ref_space_uid.shape == (1, N_ATOM)


def test_build_sampling_context_tok_idx_shape(context):
    assert context.tok_idx.shape == (1, N_ATOM)


def test_build_sampling_context_center_uid_shape(context):
    assert context.center_uid.shape == (1, N_RES)


def test_build_sampling_context_r_input_shape(context):
    assert context.r_input.shape == (1, N_ATOM, 3)


def test_build_sampling_context_f_residue_idx_shape(context):
    assert context.f_residue_idx.ndim == 3
    assert context.f_residue_idx.shape[1] == N_RES


# ---------------------------------------------------------------------------
# build_sampling_context — tensor values and invariants
# ---------------------------------------------------------------------------

def test_build_sampling_context_ref_pos_is_finite(context):
    assert torch.isfinite(context.ref_pos).all()


def test_build_sampling_context_ref_element_rows_are_one_hot(context):
    row_sums = reduce(context.ref_element, "b n_atom e -> b n_atom", "sum")
    assert torch.allclose(row_sums, torch.ones(1, N_ATOM))


def test_build_sampling_context_ref_space_uid_all_zeros(context):
    assert (context.ref_space_uid == 0).all()


def test_build_sampling_context_tok_idx_maps_atoms_to_parent_residue(context):
    for i in range(N_RES):
        assert (context.tok_idx[0, i * NATOM : (i + 1) * NATOM] == i).all()


def test_build_sampling_context_center_uid_points_to_ca_slot(context):
    # atom5 slot 1 is Cα; center_uid[0, i] = i * NATOM + 1
    expected = torch.arange(N_RES) * NATOM + 1
    assert torch.equal(context.center_uid[0], expected)


def test_build_sampling_context_gt_res_distogram_all_zeros(context):
    # Unconditional generation: no template conditioning
    assert (context.gt_res_distogram == 0).all()


def test_build_sampling_context_f_pseudo_beta_mask_all_zeros(context):
    assert (context.f_pseudo_beta_mask == 0).all()


def test_build_sampling_context_placeholder_r_input_all_zeros(context):
    assert (context.r_input == 0).all()


def test_build_sampling_context_f_residue_idx_is_finite(context):
    assert torch.isfinite(context.f_residue_idx).all()


def test_build_sampling_context_atom5_mask_all_true(context):
    assert context.atom5_mask.all()


def test_build_sampling_context_residue_mask_all_true(context):
    assert context.residue_mask.all()


def test_build_sampling_context_placeholder_scalars(context):
    assert context.t_hat == 1.0
    assert context.t_normalized == 0.5


def test_build_sampling_context_n_atom_scales_linearly_with_n_res():
    emb = nn.Embedding(256, 32)
    small = build_sampling_context(N_RES, emb)
    large = build_sampling_context(N_RES * 2, emb)
    assert large.ref_pos.shape[1]    == 2 * small.ref_pos.shape[1]
    assert large.tok_idx.shape[1]    == 2 * small.tok_idx.shape[1]
    assert large.center_uid.shape[1] == 2 * small.center_uid.shape[1]


def test_build_sampling_context_custom_n_templ_bins():
    ctx = build_sampling_context(N_RES, nn.Embedding(256, 32), n_templ_bins=20)
    assert ctx.gt_res_distogram.shape == (1, N_RES, N_RES, 20)


def test_build_sampling_context_custom_n_atom_bins():
    ctx = build_sampling_context(N_RES, nn.Embedding(256, 32), n_atom_bins=10)
    assert ctx.gt_atom_distogram_sparse.shape[3] == 10  # (B, N_atom, K, n_atom_bins)


# ---------------------------------------------------------------------------
# EDMPrecond.forward — output shape and finiteness
# ---------------------------------------------------------------------------

def test_edm_precond_forward_output_shape(edm_precond):
    out = edm_precond(torch.randn(1, N_ATOM, 3), t_hat=1.0)
    assert out.shape == (1, N_ATOM, 3)


def test_edm_precond_forward_output_is_finite(edm_precond):
    out = edm_precond(torch.randn(1, N_ATOM, 3), t_hat=1.0)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# EDMPrecond.forward — context immutability
# ---------------------------------------------------------------------------

def test_edm_precond_context_r_input_not_mutated(edm_precond, context):
    r_before = context.r_input.clone()
    edm_precond(torch.randn(1, N_ATOM, 3), t_hat=5.0)
    assert torch.equal(context.r_input, r_before)


def test_edm_precond_context_t_hat_not_mutated(edm_precond, context):
    t_before = context.t_hat
    edm_precond(torch.randn(1, N_ATOM, 3), t_hat=5.0)
    assert context.t_hat == t_before


# ---------------------------------------------------------------------------
# EDMPrecond.forward — values forwarded to trunk
# ---------------------------------------------------------------------------

def test_edm_precond_passes_r_input_to_trunk(context):
    mock = _make_trunk_mock()
    precond = EDMPrecond(mock, context, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    r = torch.randn(1, N_ATOM, 3)
    precond(r, t_hat=1.0)
    batch_seen = mock.call_args.args[0]
    assert torch.equal(batch_seen.r_input, r)


def test_edm_precond_passes_t_hat_to_trunk(context):
    mock = _make_trunk_mock()
    precond = EDMPrecond(mock, context, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    precond(torch.randn(1, N_ATOM, 3), t_hat=3.14)
    batch_seen = mock.call_args.args[0]
    assert batch_seen.t_hat == pytest.approx(3.14)


def test_edm_precond_t_normalized_is_zero_at_sigma_min(context):
    mock = _make_trunk_mock()
    precond = EDMPrecond(mock, context, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    precond(torch.randn(1, N_ATOM, 3), t_hat=SIGMA_MIN)
    batch_seen = mock.call_args.args[0]
    assert batch_seen.t_normalized == pytest.approx(0.0, abs=1e-5)


def test_edm_precond_t_normalized_is_one_at_sigma_max(context):
    mock = _make_trunk_mock()
    precond = EDMPrecond(mock, context, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    precond(torch.randn(1, N_ATOM, 3), t_hat=SIGMA_MAX)
    batch_seen = mock.call_args.args[0]
    assert batch_seen.t_normalized == pytest.approx(1.0, abs=1e-5)


def test_edm_precond_t_normalized_at_geometric_midpoint_is_half(context):
    # Geometric midpoint of [sigma_min, sigma_max] on log scale → t_normalized = 0.5
    t_mid = math.sqrt(SIGMA_MIN * SIGMA_MAX)
    mock = _make_trunk_mock()
    precond = EDMPrecond(mock, context, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    precond(torch.randn(1, N_ATOM, 3), t_hat=t_mid)
    batch_seen = mock.call_args.args[0]
    assert batch_seen.t_normalized == pytest.approx(0.5, abs=1e-5)


def test_edm_precond_trunk_called_exactly_once_per_forward(edm_precond, trunk_mock):
    edm_precond(torch.randn(1, N_ATOM, 3), t_hat=1.0)
    assert trunk_mock.call_count == 1


# ---------------------------------------------------------------------------
# EDMPrecond.forward — type enforcement
# ---------------------------------------------------------------------------

def test_edm_precond_rejects_r_input_with_wrong_last_dim(edm_precond):
    with pytest.raises(Exception):
        edm_precond(torch.randn(1, N_ATOM, 4), t_hat=1.0)   # 4 instead of 3


def test_edm_precond_rejects_r_input_with_wrong_ndim(edm_precond):
    with pytest.raises(Exception):
        edm_precond(torch.randn(N_ATOM, 3), t_hat=1.0)   # 2D instead of 3D


# ---------------------------------------------------------------------------
# EDMSampler._sigma_schedule — length and boundary values
# ---------------------------------------------------------------------------

def test_sigma_schedule_length_is_steps_plus_one(bare_sampler):
    sigmas = bare_sampler._sigma_schedule(10, "cpu")
    assert sigmas.shape == (11,)


def test_sigma_schedule_first_value_equals_sigma_max(bare_sampler):
    sigmas = bare_sampler._sigma_schedule(20, "cpu")
    assert sigmas[0].item() == pytest.approx(SIGMA_MAX, rel=1e-4)


def test_sigma_schedule_last_value_is_zero(bare_sampler):
    sigmas = bare_sampler._sigma_schedule(20, "cpu")
    assert sigmas[-1].item() == pytest.approx(0.0, abs=1e-8)


def test_sigma_schedule_is_monotonically_non_increasing(bare_sampler):
    sigmas = bare_sampler._sigma_schedule(20, "cpu")
    diffs  = sigmas[1:] - sigmas[:-1]
    assert (diffs <= 0).all()


def test_sigma_schedule_all_values_nonnegative(bare_sampler):
    sigmas = bare_sampler._sigma_schedule(20, "cpu")
    assert (sigmas >= 0).all()


def test_sigma_schedule_different_rho_gives_different_intermediate_values():
    s1 = EDMSampler(_make_zero_denoiser_mock(), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=7.0)  # type: ignore[arg-type]
    s2 = EDMSampler(_make_zero_denoiser_mock(), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=3.0)  # type: ignore[arg-type]
    sig1 = s1._sigma_schedule(10, "cpu")
    sig2 = s2._sigma_schedule(10, "cpu")
    # Endpoints are the same; interior values must differ with different ρ
    assert not torch.allclose(sig1[1:-1], sig2[1:-1])


# ---------------------------------------------------------------------------
# EDMSampler.sample — output shape and finiteness
# ---------------------------------------------------------------------------

def test_edm_sampler_output_shape(edm_sampler):
    out = edm_sampler.sample((1, N_ATOM, 3), steps=3)
    assert out.shape == (1, N_ATOM, 3)


def test_edm_sampler_output_is_finite(edm_sampler):
    out = edm_sampler.sample((1, N_ATOM, 3), steps=3)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# EDMSampler.sample — mathematical properties of special denoisers
# ---------------------------------------------------------------------------

def test_edm_sampler_identity_denoiser_z_unchanged_across_step_counts():
    # D_θ(z, σ) = z  ⟹  d = (z - z)/σ = 0  ⟹  z_next = z for every step.
    # The output equals the initial noise regardless of how many steps are run.
    # (steps=1 is excluded: _sigma_schedule divides by steps-1, causing 0/0.)
    sampler = EDMSampler(_make_identity_denoiser_mock(), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, S_churn=0.0)  # type: ignore[arg-type]
    torch.manual_seed(7)
    z2 = sampler.sample((1, N_ATOM, 3), steps=2)
    torch.manual_seed(7)
    z6 = sampler.sample((1, N_ATOM, 3), steps=6)
    assert torch.allclose(z2, z6, atol=1e-5)


def test_edm_sampler_zero_denoiser_output_is_zero():
    # D_θ(z, σ) = 0  ⟹  d = z/σ  ⟹  z scales by σ_next/σ_hat each step;
    # at the final step σ_next = 0, so the trajectory converges exactly to 0.
    sampler = EDMSampler(_make_zero_denoiser_mock(), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, S_churn=0.0)  # type: ignore[arg-type]
    out = sampler.sample((1, N_ATOM, 3), steps=5)
    assert torch.allclose(out, torch.zeros(1, N_ATOM, 3), atol=1e-5)


# ---------------------------------------------------------------------------
# EDMSampler.sample — determinism and stochasticity
# ---------------------------------------------------------------------------

def test_edm_sampler_deterministic_without_s_churn(edm_sampler):
    torch.manual_seed(3)
    out1 = edm_sampler.sample((1, N_ATOM, 3), steps=3)
    torch.manual_seed(3)
    out2 = edm_sampler.sample((1, N_ATOM, 3), steps=3)
    assert torch.equal(out1, out2)


def test_edm_sampler_s_churn_produces_different_result_than_deterministic():
    # S_churn > 0 injects extra noise per step before the predictor.
    # With the identity denoiser (d = 0), z never moves during predictor/corrector,
    # so injected noise accumulates unfiltered → output diverges from the ODE run.
    # (Zero denoiser is unsuitable here: it drives z → 0 regardless of S_churn.)
    det   = EDMSampler(_make_identity_denoiser_mock(), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, S_churn=0.0)  # type: ignore[arg-type]
    stoch = EDMSampler(_make_identity_denoiser_mock(), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, S_churn=2.0)  # type: ignore[arg-type]
    torch.manual_seed(5)
    out_det = det.sample((1, N_ATOM, 3), steps=5)
    torch.manual_seed(5)
    out_stoch = stoch.sample((1, N_ATOM, 3), steps=5)
    assert not torch.allclose(out_det, out_stoch)


# ---------------------------------------------------------------------------
# EDMSampler.sample — Heun corrector call count
# ---------------------------------------------------------------------------

def test_edm_sampler_heun_corrector_call_count():
    # steps predictor calls + (steps - 1) corrector calls (skipped when σ_next = 0)
    # Total = 2·steps - 1
    counter = _make_zero_denoiser_mock()
    steps   = 5
    sampler = EDMSampler(counter, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, S_churn=0.0)  # type: ignore[arg-type]
    sampler.sample((1, N_ATOM, 3), steps=steps)
    assert counter.call_count == 2 * steps - 1


# ---------------------------------------------------------------------------
# EDMSampler.sample — step count affects trajectory
# ---------------------------------------------------------------------------

def test_edm_sampler_step_count_changes_output_for_nontrivial_denoiser():
    # D_θ(z, σ) = 0.5·z: trajectory depends on the σ grid.
    # Coarser grid (fewer steps) integrates differently → different final output.
    coarse = EDMSampler(_make_half_denoiser_mock(), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, S_churn=0.0)  # type: ignore[arg-type]
    fine   = EDMSampler(_make_half_denoiser_mock(), sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, S_churn=0.0)  # type: ignore[arg-type]
    torch.manual_seed(9)
    out_coarse = coarse.sample((1, N_ATOM, 3), steps=2)
    torch.manual_seed(9)
    out_fine   = fine.sample((1, N_ATOM, 3), steps=10)
    assert not torch.allclose(out_coarse, out_fine)


# ---------------------------------------------------------------------------
# atom5_to_atom37 — dtype
# ---------------------------------------------------------------------------

def test_atom5_to_atom37_output_dtype_is_float32(coords5):
    x_37, mask_37 = atom5_to_atom37(coords5)
    assert x_37.dtype == np.float32
    assert mask_37.dtype == np.float32


# ---------------------------------------------------------------------------
# build_sampling_context — additional fields
# ---------------------------------------------------------------------------

def test_build_sampling_context_aa_indices_all_zeros(context):
    assert (context.aa_indices == 0).all()


def test_build_sampling_context_r_gt_all_zeros(context):
    assert (context.r_gt == 0).all()


def test_build_sampling_context_gt_atom_distogram_sparse_all_zeros(context):
    assert (context.gt_atom_distogram_sparse == 0).all()


def test_build_sampling_context_gt_atom_distogram_mask_sparse_all_false(context):
    assert not context.gt_atom_distogram_mask_sparse.any()


def test_build_sampling_context_gt_atom_distogram_sparse_leading_dim_is_n_atom(context):
    assert context.gt_atom_distogram_sparse.shape[1] == N_ATOM   # (B, N_atom, K, n_atom_bins)
    assert context.gt_atom_distogram_sparse.shape[3] == 22       # default n_atom_bins


def test_build_sampling_context_c_res_embed_sets_f_residue_idx_dim(context_c_res_64):
    assert context_c_res_64.f_residue_idx.shape == (1, N_RES, 64)


def test_build_sampling_context_tensors_on_cpu_by_default(context):
    assert context.ref_pos.device.type == "cpu"
    assert context.tok_idx.device.type == "cpu"
    assert context.r_input.device.type == "cpu"


# ---------------------------------------------------------------------------
# EDMPrecond — attribute storage and formula correctness
# ---------------------------------------------------------------------------

def test_edm_precond_stores_sigma_min_and_sigma_max(edm_precond_custom_sigmas):
    assert edm_precond_custom_sigmas.sigma_min == 0.01
    assert edm_precond_custom_sigmas.sigma_max == 50.0


def test_edm_precond_t_normalized_formula_at_arbitrary_sigma(edm_precond, trunk_mock):
    t_hat = 1.0
    edm_precond(torch.randn(1, N_ATOM, 3), t_hat=t_hat)
    batch_seen = trunk_mock.call_args.args[0]
    expected   = (math.log(t_hat) - math.log(SIGMA_MIN)) / (math.log(SIGMA_MAX) - math.log(SIGMA_MIN))
    assert batch_seen.t_normalized == pytest.approx(expected, abs=1e-5)


def test_edm_precond_multiple_calls_are_independent(edm_precond):
    # Identity trunk mock returns r_input unchanged; verify no cross-call state leak.
    r1   = torch.randn(1, N_ATOM, 3)
    r2   = torch.randn(1, N_ATOM, 3)
    out1 = edm_precond(r1, t_hat=1.0)
    out2 = edm_precond(r2, t_hat=2.0)
    assert torch.equal(out1, r1)
    assert torch.equal(out2, r2)


# ---------------------------------------------------------------------------
# EDMSampler._sigma_schedule — penultimate value
# ---------------------------------------------------------------------------

def test_sigma_schedule_penultimate_value_is_sigma_min(bare_sampler):
    # At i = steps-1 the Karras schedule formula collapses to sigma_min.
    sigmas = bare_sampler._sigma_schedule(20, "cpu")
    assert sigmas[-2].item() == pytest.approx(SIGMA_MIN, rel=1e-4)


# ---------------------------------------------------------------------------
# EDMSampler.sample — edge cases and S_churn windowing
# ---------------------------------------------------------------------------

def test_edm_sampler_sample_steps_2_runs_without_error(edm_sampler):
    out = edm_sampler.sample((1, N_ATOM, 3), steps=2)
    assert out.shape == (1, N_ATOM, 3)
    assert torch.isfinite(out).all()


def test_edm_sampler_s_tmin_above_sigma_max_disables_injection(
    identity_det_sampler, identity_stoch_sampler_tmin_high
):
    # S_tmin > sigma_max ⟹ S_tmin ≤ sigma_cur is never met ⟹ same as S_churn=0.
    torch.manual_seed(5)
    out_det = identity_det_sampler.sample((1, N_ATOM, 3), steps=5)
    torch.manual_seed(5)
    out_stoch = identity_stoch_sampler_tmin_high.sample((1, N_ATOM, 3), steps=5)
    assert torch.allclose(out_det, out_stoch)


# ---------------------------------------------------------------------------
# build_AA_context — constants
# ---------------------------------------------------------------------------

N_RES_AA    = 6
N_ATOM_AA   = N_RES_AA * NATOM    # 30
AA_SEQ_AA   = "ACDEFG"            # len == N_RES_AA
C_RES_AA    = 32
N_ATOM_BINS = 22


# ---------------------------------------------------------------------------
# build_AA_context — fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def atom37_pos() -> torch.Tensor:
    return torch.randn(N_RES_AA, 37, 3)


@pytest.fixture
def atom37_mask_all() -> torch.Tensor:
    return torch.ones(N_RES_AA, 37)


@pytest.fixture
def residue_idx_aa() -> torch.Tensor:
    return torch.arange(N_RES_AA, dtype=torch.float)


@pytest.fixture
def aa_emb() -> nn.Embedding:
    return nn.Embedding(256, C_RES_AA)


@pytest.fixture
def atom_disto_fn() -> Distogram:
    return Distogram(n_bins=N_ATOM_BINS, min_dist=2.0, max_dist=22.0)


@pytest.fixture
def aa_ctx(
    atom37_pos, atom37_mask_all, residue_idx_aa, aa_emb, atom_disto_fn
) -> AllAtomContext:
    with torch.no_grad():
        return build_AA_context(
            atom_37_coordinate_tensor=atom37_pos,
            atom_37_mask=atom37_mask_all,
            residue_index=residue_idx_aa,
            index_embedding=aa_emb,
            aa_sequence=AA_SEQ_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=2,
            device="cpu",
        )


# ---------------------------------------------------------------------------
# build_AA_context — return type
# ---------------------------------------------------------------------------

def test_build_aa_context_returns_all_atom_context(aa_ctx):
    assert isinstance(aa_ctx, AllAtomContext)


# ---------------------------------------------------------------------------
# build_AA_context — tensor shapes
# ---------------------------------------------------------------------------

def test_build_aa_context_r_gt_shape(aa_ctx):
    assert aa_ctx.r_gt.shape == (2, N_ATOM_AA, 3)


def test_build_aa_context_atom5_mask_shape(aa_ctx):
    assert aa_ctx.atom5_mask.shape == (2, N_ATOM_AA)


def test_build_aa_context_residue_mask_shape(aa_ctx):
    assert aa_ctx.residue_mask.shape == (2, N_RES_AA)


def test_build_aa_context_aa_indices_shape(aa_ctx):
    assert aa_ctx.aa_indices.shape == (2, N_RES_AA)


def test_build_aa_context_f_residue_idx_shape(aa_ctx):
    assert aa_ctx.f_residue_idx.shape == (2, N_RES_AA, C_RES_AA)


def test_build_aa_context_gt_atom_distogram_sparse_atom_dim(aa_ctx):
    assert aa_ctx.gt_atom_distogram_sparse.shape[1] == N_ATOM_AA


def test_build_aa_context_gt_atom_distogram_sparse_n_bins(aa_ctx):
    assert aa_ctx.gt_atom_distogram_sparse.shape[3] == N_ATOM_BINS


def test_build_aa_context_gt_atom_distogram_mask_consistent_with_sparse(aa_ctx):
    B_dim, N_a, K, _ = aa_ctx.gt_atom_distogram_sparse.shape
    assert aa_ctx.gt_atom_distogram_mask_sparse.shape == (B_dim, N_a, K)


# ---------------------------------------------------------------------------
# build_AA_context — tensor values
# ---------------------------------------------------------------------------

def test_build_aa_context_r_gt_is_finite(aa_ctx):
    assert torch.isfinite(aa_ctx.r_gt).all()


def test_build_aa_context_f_residue_idx_is_finite(aa_ctx):
    assert torch.isfinite(aa_ctx.f_residue_idx).all()


def test_build_aa_context_gt_atom_distogram_sparse_is_finite(aa_ctx):
    assert torch.isfinite(aa_ctx.gt_atom_distogram_sparse).all()


def test_build_aa_context_full_mask_gives_all_true_residue_mask(aa_ctx):
    assert aa_ctx.residue_mask.all()


def test_build_aa_context_full_mask_gives_all_true_atom5_mask(aa_ctx):
    assert aa_ctx.atom5_mask.all()


def test_build_aa_context_aa_indices_match_restype_order(aa_ctx):
    from helpers.atom_utils import restype_order as _ro
    expected = torch.tensor([_ro[c] for c in AA_SEQ_AA], dtype=torch.long)
    assert torch.equal(aa_ctx.aa_indices[0], expected)


def test_build_aa_context_all_batch_slices_of_r_gt_are_identical(aa_ctx):
    assert torch.equal(aa_ctx.r_gt[0], aa_ctx.r_gt[1])


def test_build_aa_context_all_batch_slices_of_aa_indices_are_identical(aa_ctx):
    assert torch.equal(aa_ctx.aa_indices[0], aa_ctx.aa_indices[1])


def test_build_aa_context_batch_size_controls_leading_dim(
    atom37_pos, atom37_mask_all, residue_idx_aa, aa_emb, atom_disto_fn
):
    with torch.no_grad():
        ctx = build_AA_context(
            atom37_pos, atom37_mask_all, residue_idx_aa, aa_emb,
            AA_SEQ_AA, atom_disto_fn, batch_size=3, device="cpu",
        )
    assert ctx.r_gt.shape[0] == 3
    assert ctx.aa_indices.shape[0] == 3


# ---------------------------------------------------------------------------
# build_template_context — constants and helpers
# ---------------------------------------------------------------------------

N_TEMPL_BINS = 39    # 38 distance bins + 1 overflow bin


def _make_protein(n_res: int, mask_value: float = 1.0) -> Protein:
    rng = np.random.RandomState(42)
    return Protein(
        atom_positions=rng.randn(n_res, 37, 3).astype(np.float32),
        aatype=np.zeros(n_res, dtype=np.int32),
        atom_mask=np.full((n_res, 37), mask_value, dtype=np.float32),
        residue_index=np.arange(n_res, dtype=np.int32),
        chain_index=np.zeros(n_res, dtype=np.int32),
        b_factors=np.ones((n_res, 37), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# build_template_context — fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def np_protein() -> Protein:
    return _make_protein(N_RES)


@pytest.fixture
def np_protein_short() -> Protein:
    return _make_protein(N_RES - 2)    # shorter, for padding test


@pytest.fixture
def templ_disto() -> Distogram:
    return Distogram(n_bins=N_TEMPL_BINS - 1, min_dist=3.25, max_dist=50.75, overflow_bin=True)


@pytest.fixture
def template_ctx(np_protein, templ_disto) -> TemplateContext:
    return build_template_context([np_protein], templ_disto)


@pytest.fixture
def template_ctx_batch2(np_protein, templ_disto) -> TemplateContext:
    return build_template_context([np_protein, np_protein], templ_disto)


# ---------------------------------------------------------------------------
# build_template_context — return type
# ---------------------------------------------------------------------------

def test_build_template_context_returns_template_context(template_ctx):
    assert isinstance(template_ctx, TemplateContext)


# ---------------------------------------------------------------------------
# build_template_context — tensor shapes
# ---------------------------------------------------------------------------

def test_build_template_context_f_template_distogram_shape(template_ctx):
    assert template_ctx.f_template_distogram.shape == (1, N_RES, N_RES, N_TEMPL_BINS)


def test_build_template_context_f_pseudo_beta_mask_shape(template_ctx):
    assert template_ctx.f_pseudo_beta_mask.shape == (1, N_RES)


def test_build_template_context_batch_size_is_len_proteins(template_ctx_batch2):
    assert template_ctx_batch2.f_template_distogram.shape[0] == 2
    assert template_ctx_batch2.f_pseudo_beta_mask.shape[0] == 2


# ---------------------------------------------------------------------------
# build_template_context — tensor dtypes
# ---------------------------------------------------------------------------

def test_build_template_context_f_template_distogram_dtype_is_long(template_ctx):
    assert template_ctx.f_template_distogram.dtype == torch.long


def test_build_template_context_f_pseudo_beta_mask_dtype_is_long(template_ctx):
    assert template_ctx.f_pseudo_beta_mask.dtype == torch.long


# ---------------------------------------------------------------------------
# build_template_context — distogram properties
# ---------------------------------------------------------------------------

def test_build_template_context_distogram_is_valid_one_hot(template_ctx):
    bin_sums = reduce(
        template_ctx.f_template_distogram.float(), "b i j bins -> b i j", "sum"
    )
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums))


def test_build_template_context_distogram_is_symmetric(template_ctx):
    disto = template_ctx.f_template_distogram
    assert torch.equal(disto, rearrange(disto, "b i j k -> b j i k"))


def test_build_template_context_different_proteins_give_different_distograms(templ_disto):
    rng_a = np.random.RandomState(1)
    rng_b = np.random.RandomState(2)
    prot_a = _make_protein(N_RES)
    prot_b = Protein(
        atom_positions=(rng_b.randn(N_RES, 37, 3) * 20).astype(np.float32),
        aatype=np.zeros(N_RES, dtype=np.int32),
        atom_mask=np.ones((N_RES, 37), dtype=np.float32),
        residue_index=np.arange(N_RES, dtype=np.int32),
        chain_index=np.zeros(N_RES, dtype=np.int32),
        b_factors=np.ones((N_RES, 37), dtype=np.float32),
    )
    ctx_a = build_template_context([prot_a], templ_disto)
    ctx_b = build_template_context([prot_b], templ_disto)
    assert not torch.equal(ctx_a.f_template_distogram, ctx_b.f_template_distogram)


# ---------------------------------------------------------------------------
# build_template_context — mask values
# ---------------------------------------------------------------------------

def test_build_template_context_full_ca_mask_gives_all_ones_pseudo_beta_mask(template_ctx):
    assert (template_ctx.f_pseudo_beta_mask == 1).all()


def test_build_template_context_padded_residues_have_zero_pseudo_beta_mask(
    np_protein_short, templ_disto
):
    # np_protein_short has N_RES-2 residues; batched with full-length protein it is
    # padded to N_RES. The 2 trailing padded residues must have mask 0.
    prot_full = _make_protein(N_RES)
    ctx = build_template_context([prot_full, np_protein_short], templ_disto)
    assert (ctx.f_pseudo_beta_mask[1, N_RES - 2 :] == 0).all()


def test_build_template_context_non_padded_residues_have_nonzero_pseudo_beta_mask(
    np_protein_short, templ_disto
):
    prot_full = _make_protein(N_RES)
    ctx = build_template_context([prot_full, np_protein_short], templ_disto)
    assert (ctx.f_pseudo_beta_mask[1, : N_RES - 2] == 1).all()

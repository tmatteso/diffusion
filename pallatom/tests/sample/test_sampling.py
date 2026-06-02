"""Tests for the conditional sampling loop."""

import math
import pathlib
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from architecture.main_trunk import PredictedOutputs
from einops import rearrange, reduce
from helpers.atom_utils import Protein, restype_order, to_pdb
from helpers.featurize import Distogram, FeaturizedBatch
from helpers.useful_objects import manual_seed
from jaxtyping import Float, TypeCheckError
from sample.sample_config import SamplerParams
from sample.sampling import (
    ATOM5_TO_ATOM37,
    NATOM,
    AllAtomContext,
    EDMSampler,
    TemplateContext,
    atom5_to_atom37,
    build_AA_context,
    build_sampling_context,
    build_template_context,
)
from train.train_config import NoiseScheduleParams

manual_seed(0)

N_RES = 4
N_ATOM = N_RES * NATOM  # 20
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_trunk_mock() -> MagicMock:
    """MagicMock with identity denoiser behaviour: returns r_input unchanged."""
    mock = MagicMock()

    def _forward(batch: FeaturizedBatch) -> PredictedOutputs:
        B_local = batch.r_gt.shape[0]
        n_atom = batch.r_gt.shape[1]
        n_res = int(batch.tok_idx.max().item()) + 1
        return PredictedOutputs(
            r_denoised=batch.r_gt.clone(),
            seq_logits=torch.zeros(B_local, n_res, 20),
            residue_distogram_logits=torch.zeros(B_local, n_res, n_res, 38),
            atom_distogram_logits=torch.zeros(B_local, n_atom, 1, 38),
            intermediate_denoised_coord_stack=[],
            intermediate_pred_aa_logit_stack=[],
        )

    mock.side_effect = _forward
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def coords5() -> Float[torch.Tensor, "N_RES 5 3"]:
    """Provide random atom5 coordinates (N_RES, 5, 3) with fixed seed."""
    return torch.tensor(np.random.RandomState(1).randn(N_RES, 5, 3).astype(np.float64))


@pytest.fixture
def context(atom_disto_fn: Distogram, templ_disto: Distogram) -> FeaturizedBatch:
    """Provide an unconditional sampling context (all-alanine, zero atom positions)."""
    with torch.no_grad():
        return build_sampling_context(
            atom_positions=torch.zeros(N_RES, 37, 3),
            atom_mask=torch.ones(N_RES, 37),
            residue_index=torch.arange(N_RES, dtype=torch.float),
            seq="A" * N_RES,
            pdb_file_path=None,
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def trunk_mock() -> MagicMock:
    """Provide an identity trunk mock that echoes r_input as the denoised output."""
    return _make_trunk_mock()


@pytest.fixture
def edm_sampler(
    trunk_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
) -> EDMSampler:
    """Provide a deterministic EDMSampler (S_churn=0) built on the identity trunk mock."""
    return EDMSampler(
        trunk_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )


@pytest.fixture
def zero_denoiser_mock() -> MagicMock:
    """MagicMock trunk that always returns zero denoised coordinates."""
    mock = MagicMock()

    def zero_side_effect(batch: FeaturizedBatch) -> PredictedOutputs:
        """Return zero coordinates and zero sequence logits."""
        B_local = batch.r_gt.shape[0]
        n_atom = batch.r_gt.shape[1]
        n_res = int(batch.tok_idx.max().item()) + 1
        return PredictedOutputs(
            r_denoised=torch.zeros_like(batch.r_gt),
            seq_logits=torch.zeros(B_local, n_res, 20),
            residue_distogram_logits=torch.zeros(B_local, n_res, n_res, 38),
            atom_distogram_logits=torch.zeros(B_local, n_atom, 1, 38),
            intermediate_denoised_coord_stack=[],
            intermediate_pred_aa_logit_stack=[],
        )

    mock.side_effect = zero_side_effect
    return mock


@pytest.fixture
def identity_denoiser_mock() -> MagicMock:
    """MagicMock trunk that returns r_gt unchanged as the denoised output."""
    return _make_trunk_mock()


@pytest.fixture
def half_denoiser_mock() -> MagicMock:
    """MagicMock trunk that scales r_gt by 0.5 (non-trivial, trajectory-dependent)."""
    mock = MagicMock()

    def half_side_effect(batch: FeaturizedBatch) -> PredictedOutputs:
        """Return 0.5 * r_gt as denoised coordinates and zero sequence logits."""
        B_local = batch.r_gt.shape[0]
        n_atom = batch.r_gt.shape[1]
        n_res = int(batch.tok_idx.max().item()) + 1
        return PredictedOutputs(
            r_denoised=batch.r_gt.clone() * 0.5,
            seq_logits=torch.zeros(B_local, n_res, 20),
            residue_distogram_logits=torch.zeros(B_local, n_res, n_res, 38),
            atom_distogram_logits=torch.zeros(B_local, n_atom, 1, 38),
            intermediate_denoised_coord_stack=[],
            intermediate_pred_aa_logit_stack=[],
        )

    mock.side_effect = half_side_effect
    return mock


@pytest.fixture
def bare_sampler(
    zero_denoiser_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
) -> EDMSampler:
    """EDMSampler with a zero denoiser; used for _sigma_schedule tests only."""
    return EDMSampler(
        zero_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(rho=7.0),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )


@pytest.fixture
def identity_det_sampler(
    identity_denoiser_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
) -> EDMSampler:
    """Provide a deterministic EDMSampler (S_churn=0) using the identity denoiser."""
    return EDMSampler(
        identity_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )


@pytest.fixture
def identity_stoch_sampler_tmin_high(
    identity_denoiser_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
) -> EDMSampler:
    """Provide a stochastic EDMSampler with S_tmin set far above sigma_max to suppress injection."""
    return EDMSampler(
        identity_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=2.0, S_tmin=SIGMA_MAX * 10),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )


# ---------------------------------------------------------------------------
# atom5_to_atom37 — output shapes
# ---------------------------------------------------------------------------


def test_atom5_to_atom37_x37_shape(coords5: Float[torch.Tensor, "N_RES 5 3"]):
    """Verify that the coordinate output expands the second axis from 5 to 37 slots."""
    x_37, _ = atom5_to_atom37(coords5)
    assert x_37.shape == (N_RES, 37, 3)


def test_atom5_to_atom37_mask37_shape(coords5: Float[torch.Tensor, "N_RES 5 3"]):
    """Verify that the mask output has the atom37 width, one binary entry per residue-atom slot."""
    _, mask_37 = atom5_to_atom37(coords5)
    assert mask_37.shape == (N_RES, 37)


# ---------------------------------------------------------------------------
# atom5_to_atom37 — coordinate placement
# ---------------------------------------------------------------------------


def test_atom5_to_atom37_each_slot_lands_at_correct_atom37_index():
    """Confirm that each atom5 slot maps coordinates to expected atom37 index in ATOM5_TO_ATOM37."""
    # Give each atom5 slot a unique sentinel value so placement is unambiguous.
    coords_5 = torch.zeros((N_RES, 5, 3), dtype=torch.float64)
    for slot in range(5):
        coords_5[:, slot, :] = float(slot + 1)
    x_37, _ = atom5_to_atom37(coords_5)
    for slot, atom37_idx in enumerate(ATOM5_TO_ATOM37):
        assert torch.allclose(
            x_37[:, atom37_idx, :], torch.tensor(float(slot + 1), dtype=torch.float64)
        ), f"atom5 slot {slot} → atom37 slot {atom37_idx}: wrong coords"


def test_atom5_to_atom37_unoccupied_atom37_slots_are_zero(
    coords5: Float[torch.Tensor, "N_RES 5 3"],
):
    """Assert that atom37 positions not covered by atom5 remain exactly zero after mapping."""
    x_37, _ = atom5_to_atom37(coords5)
    occupied = set(ATOM5_TO_ATOM37)
    for idx in range(37):
        if idx not in occupied:
            assert torch.allclose(
                x_37[:, idx, :], torch.zeros(1, dtype=torch.float64)
            ), f"atom37 slot {idx} should be zero (unoccupied)"


# ---------------------------------------------------------------------------
# atom5_to_atom37 — mask handling
# ---------------------------------------------------------------------------


def test_atom5_to_atom37_mask_none_sets_occupied_slots_to_one(
    coords5: Float[torch.Tensor, "N_RES 5 3"],
):
    """When no explicit mask is given, all atom5-occupied slots in the atom37 mask must be 1."""
    _, mask_37 = atom5_to_atom37(coords5, mask_5=None)
    for atom37_idx in ATOM5_TO_ATOM37:
        assert torch.allclose(mask_37[:, atom37_idx], torch.ones(N_RES, dtype=torch.float64))


def test_atom5_to_atom37_explicit_mask_placed_at_correct_atom37_positions():
    """An explicit atom5 mask must be faithfully scatter-copied into correct atom37 positions."""
    rng = np.random.RandomState(2)
    coords_5 = torch.tensor(rng.randn(N_RES, 5, 3).astype(np.float64))
    mask_5 = torch.tensor(rng.rand(N_RES, 5).astype(np.float64))
    _, mask_37 = atom5_to_atom37(coords_5, mask_5)
    for slot, atom37_idx in enumerate(ATOM5_TO_ATOM37):
        assert torch.allclose(mask_37[:, atom37_idx], mask_5[:, slot])


def test_atom5_to_atom37_unoccupied_mask_slots_are_zero():
    """Atom37 mask entries with no corresponding atom5 must remain zero when atom5 mask is ones."""
    coords_5 = torch.ones((N_RES, 5, 3), dtype=torch.float64)
    mask_5 = torch.ones((N_RES, 5), dtype=torch.float64)
    _, mask_37 = atom5_to_atom37(coords_5, mask_5)
    occupied = set(ATOM5_TO_ATOM37)
    for idx in range(37):
        if idx not in occupied:
            assert torch.allclose(mask_37[:, idx], torch.zeros(N_RES, dtype=torch.float64))


# ---------------------------------------------------------------------------
# atom5_to_atom37 — type enforcement and edge cases
# ---------------------------------------------------------------------------


def test_atom5_to_atom37_rejects_wrong_second_dimension():
    """A jaxtyping TypeCheckError must be raised when the second dimension is not 5."""
    with pytest.raises(TypeCheckError):
        atom5_to_atom37(torch.zeros((N_RES, 4, 3), dtype=torch.float64))  # 4 ≠ 5


def test_atom5_to_atom37_single_residue():
    """The function must handle a single-residue input without broadcasting errors."""
    coords_5 = torch.randn(1, 5, 3)
    x_37, mask_37 = atom5_to_atom37(coords_5)
    assert x_37.shape == (1, 37, 3)
    assert mask_37.shape == (1, 37)


# ---------------------------------------------------------------------------
# build_sampling_context — return type
# ---------------------------------------------------------------------------


def test_build_sampling_context_returns_featurized_batch(context: FeaturizedBatch):
    """Verify build_sampling_context returns a FeaturizedBatch, not a plain dict or namedtuple."""
    assert isinstance(context, FeaturizedBatch)


# ---------------------------------------------------------------------------
# build_sampling_context — tensor shapes  (default batch_size=1)
# ---------------------------------------------------------------------------


def test_build_sampling_context_ref_pos_shape(context: FeaturizedBatch):
    """Reference positions must be batched to (1, N_atom, 3) where N_atom = N_RES * NATOM."""
    assert context.ref_pos.shape == (1, N_ATOM, 3)


def test_build_sampling_context_ref_element_shape(context: FeaturizedBatch):
    """Element one-hot features cover every atom in the packed sequence with 4 element classes."""
    assert context.ref_element.shape == (1, N_ATOM, 4)


def test_build_sampling_context_ref_space_uid_shape(context: FeaturizedBatch):
    """Each atom must receive a chain/space identifier, giving a flat (1, N_atom) index tensor."""
    assert context.ref_space_uid.shape == (1, N_ATOM)


def test_build_sampling_context_tok_idx_shape(context: FeaturizedBatch):
    """tok_idx must assign every atom to a residue, yielding shape (1, N_atom)."""
    assert context.tok_idx.shape == (1, N_ATOM)


def test_build_sampling_context_center_uid_shape(context: FeaturizedBatch):
    """center_uid maps every atom to its residue's CA index, shape (1, N_ATOM)."""
    assert context.center_uid.shape == (1, N_ATOM)


def test_build_sampling_context_f_residue_idx_shape(context: FeaturizedBatch):
    """The residue index must have shape (B, N_RES)."""
    assert context.f_residue_idx.ndim == 2
    assert context.f_residue_idx.shape[1] == N_RES


# ---------------------------------------------------------------------------
# build_sampling_context — tensor values and invariants
# ---------------------------------------------------------------------------


def test_build_sampling_context_ref_pos_is_finite(context: FeaturizedBatch):
    """All reference positions must be finite (no NaN/Inf from coordinate preprocessing)."""
    assert torch.isfinite(context.ref_pos).all()


def test_build_sampling_context_ref_element_rows_are_one_hot(context: FeaturizedBatch):
    """Each atom's element feature must sum to exactly 1.0, confirming valid one-hot encoding."""
    row_sums = reduce(context.ref_element, "b n_atom e -> b n_atom", "sum")
    assert torch.allclose(row_sums, torch.ones(1, N_ATOM))


def test_build_sampling_context_ref_space_uid_equals_residue_index(context: FeaturizedBatch):
    """ref_space_uid must assign each atom the index of its parent residue."""
    expected = torch.arange(N_RES).repeat_interleave(NATOM)
    assert torch.equal(context.ref_space_uid[0], expected)


def test_build_sampling_context_tok_idx_maps_atoms_to_parent_residue(context: FeaturizedBatch):
    """tok_idx must assign each contiguous block of NATOM atoms to the correct residue index."""
    for i in range(N_RES):
        assert (context.tok_idx[0, i * NATOM : (i + 1) * NATOM] == i).all()


def test_build_sampling_context_center_uid_points_to_ca_slot(context: FeaturizedBatch):
    """center_uid maps every atom to the CA index of its parent residue."""
    # Each residue i broadcasts its CA index (i * NATOM + 1) to all NATOM atoms.
    expected = torch.arange(N_RES).repeat_interleave(NATOM) * NATOM + 1
    assert torch.equal(context.center_uid[0], expected)


def test_build_sampling_context_gt_res_distogram_all_zeros(context: FeaturizedBatch):
    """Without template PDB files, residue distogram conditioning signal must be entirely zero."""
    # Unconditional generation: no template conditioning
    assert (context.gt_res_distogram == 0).all()


def test_build_sampling_context_f_pseudo_beta_mask_all_zeros(context: FeaturizedBatch):
    """Without a template, the pseudo-β mask must be all-zero (no residue has template coverage)."""
    assert (context.f_pseudo_beta_mask == 0).all()


def test_build_sampling_context_f_residue_idx_is_finite(context: FeaturizedBatch):
    """Residue indices must be finite integers."""
    assert torch.isfinite(context.f_residue_idx.float()).all()


def test_build_sampling_context_atom5_mask_all_true(context: FeaturizedBatch):
    """When atom_mask is all-ones, the compressed atom5 mask must also be all-true."""
    assert context.atom5_mask.all()


def test_build_sampling_context_placeholder_scalars(context: FeaturizedBatch):
    """The placeholder diffusion scalars must be initialised to t_hat=1.0 and t_normalized=0.5."""
    assert (context.t_hat == 1.0).all()
    assert (context.t_normalized == 0.5).all()


def test_build_sampling_context_n_atom_scales_linearly_with_n_res():
    """Doubling sequence length must double atom-axis size of ref_pos, tok_idx, and center_uid."""
    a_fn = Distogram(n_bins=22, min_dist=2.0, max_dist=22.0, overflow_bin=False)
    t_fn = Distogram(n_bins=38, min_dist=3.25, max_dist=50.75, overflow_bin=True)
    with torch.no_grad():
        small = build_sampling_context(
            torch.zeros(N_RES, 37, 3),
            torch.ones(N_RES, 37),
            torch.arange(N_RES, dtype=torch.float),
            "A" * N_RES,
            None,
            a_fn,
            t_fn,
            batch_size=1,
            device="cpu",
        )
        large = build_sampling_context(
            torch.zeros(N_RES * 2, 37, 3),
            torch.ones(N_RES * 2, 37),
            torch.arange(N_RES * 2, dtype=torch.float),
            "A" * (N_RES * 2),
            None,
            a_fn,
            t_fn,
            batch_size=1,
            device="cpu",
        )
    assert large.ref_pos.shape[1] == 2 * small.ref_pos.shape[1]
    assert large.tok_idx.shape[1] == 2 * small.tok_idx.shape[1]
    assert large.center_uid.shape[1] == 2 * small.center_uid.shape[1]


def test_build_sampling_context_custom_n_templ_bins():
    """A distogram with n_bins=19 + overflow_bin must produce (B, N_res, N_res, 20) tensor."""
    a_fn = Distogram(n_bins=22, min_dist=2.0, max_dist=22.0, overflow_bin=False)
    t_fn = Distogram(n_bins=19, min_dist=3.25, max_dist=50.75, overflow_bin=True)
    with torch.no_grad():
        ctx = build_sampling_context(
            torch.zeros(N_RES, 37, 3),
            torch.ones(N_RES, 37),
            torch.arange(N_RES, dtype=torch.float),
            "A" * N_RES,
            None,
            a_fn,
            t_fn,
            batch_size=1,
            device="cpu",
        )
    assert ctx.gt_res_distogram.shape == (1, N_RES, N_RES, 20)  # 19 bins + 1 overflow


def test_build_sampling_context_custom_n_atom_bins():
    """A custom atom distogram with n_bins=10 must propagate that bin count to the sparse tensor."""
    a_fn = Distogram(n_bins=10, min_dist=2.0, max_dist=22.0, overflow_bin=False)
    t_fn = Distogram(n_bins=38, min_dist=3.25, max_dist=50.75, overflow_bin=True)
    with torch.no_grad():
        ctx = build_sampling_context(
            torch.zeros(N_RES, 37, 3),
            torch.ones(N_RES, 37),
            torch.arange(N_RES, dtype=torch.float),
            "A" * N_RES,
            None,
            a_fn,
            t_fn,
            batch_size=1,
            device="cpu",
        )
    assert ctx.gt_atom_distogram_sparse.shape[3] == 10  # (B, N_atom, K, n_atom_bins)


# ---------------------------------------------------------------------------
# EDMSampler.noise_schedule — boundary values and properties
# ---------------------------------------------------------------------------


def test_noise_schedule_at_t0_equals_sigma_data_times_sigma_max(bare_sampler: EDMSampler):
    """At t=0 the AF3 formula collapses to sigma_data * sigma_max."""
    result = bare_sampler.noise_schedule(torch.tensor(0.0))
    expected = bare_sampler.sigma_data * bare_sampler.sigma_max
    assert math.isclose(result.item(), expected, rel_tol=1e-5)


def test_noise_schedule_at_t1_equals_sigma_data_times_sigma_min(bare_sampler: EDMSampler):
    """At t=1 the AF3 formula collapses to sigma_data * sigma_min."""
    result = bare_sampler.noise_schedule(torch.tensor(1.0))
    expected = bare_sampler.sigma_data * bare_sampler.sigma_min
    assert math.isclose(result.item(), expected, rel_tol=1e-5)


def test_noise_schedule_is_monotonically_decreasing(bare_sampler: EDMSampler):
    """Noise level must never increase as t increases from 0 to 1."""
    ts = torch.linspace(0.0, 1.0, 20)
    vals = torch.stack([bare_sampler.noise_schedule(t) for t in ts])
    diffs = vals[1:] - vals[:-1]
    assert (diffs <= 0).all()


def test_noise_schedule_all_values_positive(bare_sampler: EDMSampler):
    """Noise level must be strictly positive for all t in [0, 1]."""
    ts = torch.linspace(0.0, 1.0, 20)
    vals = torch.stack([bare_sampler.noise_schedule(t) for t in ts])
    assert (vals > 0).all()


def test_noise_schedule_different_rho_gives_different_intermediate_values(
    zero_denoiser_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
):
    """Changing rho must alter intermediate noise levels while preserving boundary values."""
    noise = NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    s1 = EDMSampler(zero_denoiser_mock, context, templ_disto, SamplerParams(rho=7.0), noise)
    s2 = EDMSampler(zero_denoiser_mock, context, templ_disto, SamplerParams(rho=3.0), noise)
    ts = torch.linspace(0.1, 0.9, 10)
    vals1 = torch.stack([s1.noise_schedule(t) for t in ts])
    vals2 = torch.stack([s2.noise_schedule(t) for t in ts])
    assert not torch.allclose(vals1, vals2)


# ---------------------------------------------------------------------------
# EDMSampler.sample — mathematical properties of special denoisers
# ---------------------------------------------------------------------------


def test_edm_sampler_identity_denoiser_produces_finite_output(
    identity_denoiser_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
):
    """Identity denoiser must produce finite coordinates and sequence logits."""
    sampler = EDMSampler(
        identity_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0, ddim_steps=4),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )
    manual_seed(7)
    z, seq = sampler.sample((1, N_ATOM, 3))
    assert z.shape == (1, N_ATOM, 3)
    assert torch.isfinite(z).all()
    assert torch.isfinite(seq).all()


def test_edm_sampler_zero_denoiser_output_is_zero(
    zero_denoiser_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
):
    """With a zero denoiser the trajectory drives all coordinates to zero by the final step."""
    # D_θ(z, sigma) = 0  ⟹  d = z/sigma  ⟹  z scales by sigma_next/sigma_hat each step;
    # at the final step sigma_next = 0, so the trajectory converges exactly to 0.
    sampler = EDMSampler(
        zero_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )
    out, seq_out = sampler.sample((1, N_ATOM, 3))
    assert torch.allclose(out, torch.zeros(1, N_ATOM, 3), atol=1e-5)
    assert torch.allclose(seq_out, torch.zeros_like(seq_out))


# ---------------------------------------------------------------------------
# EDMSampler.sample — determinism and stochasticity
# ---------------------------------------------------------------------------


def test_edm_sampler_deterministic_without_s_churn(edm_sampler: EDMSampler):
    """With S_churn=0, the same random seed must always produce the exact same output tensors."""
    manual_seed(3)
    out1, seq_out1 = edm_sampler.sample((1, N_ATOM, 3))
    manual_seed(3)
    out2, seq_out2 = edm_sampler.sample((1, N_ATOM, 3))
    assert torch.equal(out1, out2)
    assert torch.equal(seq_out1, seq_out2)


def test_edm_sampler_s_churn_produces_different_result_than_deterministic(
    identity_denoiser_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
):
    """Enabling S_churn must inject per-step noise that changes output relative to the ODE run."""
    # S_churn > 0 injects extra noise per step before the predictor.
    # With the identity denoiser (d = 0), z never moves during predictor/corrector,
    # so injected noise accumulates unfiltered → output diverges from the ODE run.
    # (Zero denoiser is unsuitable here: it drives z → 0 regardless of S_churn.)
    noise = NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    manual_seed(5)
    det = EDMSampler(
        identity_denoiser_mock, context, templ_disto, SamplerParams(S_churn=0.0), noise
    )
    stoch = EDMSampler(
        identity_denoiser_mock, context, templ_disto, SamplerParams(S_churn=2.0), noise
    )
    out_det, seq_det = det.sample((1, N_ATOM, 3))
    out_stoch, seq_stoch = stoch.sample((1, N_ATOM, 3))
    assert not torch.allclose(out_det, out_stoch)
    assert torch.isfinite(seq_det).all()
    assert torch.isfinite(seq_stoch).all()


# ---------------------------------------------------------------------------
# EDMSampler.sample — Heun corrector call count
# ---------------------------------------------------------------------------


def test_edm_sampler_denoiser_called_twice_per_step(
    zero_denoiser_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
):
    """Each loop step must call denoise twice: once for self-conditioning and once for update."""
    # Loop runs over range(1, ddim_steps - 1) = ddim_steps - 2 iterations.
    # Each iteration calls denoise twice → total = 2 * (ddim_steps - 2).
    ddim_steps = 5
    sampler = EDMSampler(
        zero_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0, ddim_steps=ddim_steps),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )
    sampler.sample((1, N_ATOM, 3))
    assert zero_denoiser_mock.call_count == 2 * (ddim_steps - 2)


# ---------------------------------------------------------------------------
# EDMSampler.sample — step count affects trajectory
# ---------------------------------------------------------------------------


def test_edm_sampler_step_count_changes_output_for_nontrivial_denoiser(
    half_denoiser_mock: MagicMock, context: FeaturizedBatch, templ_disto: Distogram
):
    """Coarser sigma grid (fewer steps) must integrate differently and yield a different output."""
    # D_θ(z, sigma) = 0.5·z: trajectory depends on the sigma grid.
    # Coarser grid (fewer steps) integrates differently → different final output.
    noise = NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    manual_seed(9)
    coarse = EDMSampler(
        half_denoiser_mock, context, templ_disto, SamplerParams(S_churn=0.0, ddim_steps=4), noise
    )
    fine = EDMSampler(
        half_denoiser_mock, context, templ_disto, SamplerParams(S_churn=0.0, ddim_steps=10), noise
    )
    out_coarse, seq_coarse = coarse.sample((1, N_ATOM, 3))
    out_fine, seq_fine = fine.sample((1, N_ATOM, 3))
    assert not torch.allclose(out_coarse, out_fine)
    assert torch.isfinite(seq_coarse).all()
    assert torch.isfinite(seq_fine).all()


# ---------------------------------------------------------------------------
# atom5_to_atom37 — dtype
# ---------------------------------------------------------------------------


def test_atom5_to_atom37_output_dtype_is_float64(coords5: Float[torch.Tensor, "N_RES 5 3"]):
    """Both the coordinate and mask outputs must be float64 to match the model's expected dtype."""
    x_37, mask_37 = atom5_to_atom37(coords5)
    assert x_37.dtype == torch.float64
    assert mask_37.dtype == torch.float64


# ---------------------------------------------------------------------------
# build_sampling_context — additional fields
# ---------------------------------------------------------------------------


def test_build_sampling_context_aa_indices_all_zeros(context: FeaturizedBatch):
    """An all-alanine sequence must map every residue to index 0 in the amino-acid vocabulary."""
    assert (context.aa_indices == 0).all()


def test_build_sampling_context_r_gt_all_zeros(context: FeaturizedBatch):
    """When atom positions are all-zero inputs, ground-truth coordinates must also be all-zero."""
    assert (context.r_gt == 0).all()


def test_build_sampling_context_gt_atom_distogram_sparse_is_finite(context: FeaturizedBatch):
    """Sparse atom distogram contains no NaN or Inf from distance binning of zero coordinates."""
    assert torch.isfinite(context.gt_atom_distogram_sparse).all()


def test_build_sampling_context_gt_atom_distogram_mask_sparse_all_true(context: FeaturizedBatch):
    """With all atoms present and zero positions, every sparse pair must be valid."""
    # All atoms present (ones mask) and zero positions (dist 0 ≤ max_dist) → all pairs valid.
    assert context.gt_atom_distogram_mask_sparse.all()


def test_build_sampling_context_gt_atom_distogram_sparse_leading_dim_is_n_atom(
    context: FeaturizedBatch,
):
    """Sparse distogram atom axis must equal N_ATOM and bin axis must equal the default 22 bins."""
    assert context.gt_atom_distogram_sparse.shape[1] == N_ATOM  # (B, N_atom, K, n_atom_bins)
    assert context.gt_atom_distogram_sparse.shape[3] == 22  # default n_atom_bins


def test_build_sampling_context_tensors_on_cpu_by_default(context: FeaturizedBatch):
    """Without an explicit device argument, all output tensors must reside on CPU."""
    assert context.ref_pos.device.type == "cpu"
    assert context.tok_idx.device.type == "cpu"
    assert context.r_gt.device.type == "cpu"


# ---------------------------------------------------------------------------
# EDMPrecond — attribute storage and formula correctness
# ---------------------------------------------------------------------------
# """Successive forward calls must not share state: outputs depend only on their own inputs."""


# ---------------------------------------------------------------------------
# EDMSampler._sigma_schedule — penultimate value
# ---------------------------------------------------------------------------


def test_noise_schedule_midpoint_is_between_boundaries(bare_sampler: EDMSampler):
    """At t=0.5, noise level must be between sigma_data*sigma_min and sigma_data*sigma_max."""
    mid = bare_sampler.noise_schedule(torch.tensor(0.5))
    lo = bare_sampler.sigma_data * bare_sampler.sigma_min
    hi = bare_sampler.sigma_data * bare_sampler.sigma_max
    assert lo < mid.item() < hi


# ---------------------------------------------------------------------------
# EDMSampler.sample — edge cases and S_churn windowing
# ---------------------------------------------------------------------------


def test_edm_sampler_sample_runs_without_error(edm_sampler: EDMSampler):
    """sample() runs without error and returns finite coordinate and seq tensors."""
    out, seq_out = edm_sampler.sample((1, N_ATOM, 3))
    assert out.shape == (1, N_ATOM, 3)
    assert torch.isfinite(out).all()
    assert torch.isfinite(seq_out).all()


def test_edm_sampler_s_tmin_above_sigma_max_disables_injection(
    identity_det_sampler: EDMSampler, identity_stoch_sampler_tmin_high: EDMSampler
):
    """When S_tmin exceeds sigma_max no step satisfies condition, result identical to S_churn=0."""
    # S_tmin > sigma_max ⟹ S_tmin ≤ sigma_cur is never met ⟹ same as S_churn=0.
    manual_seed(5)
    out_det, seq_det = identity_det_sampler.sample((1, N_ATOM, 3))
    manual_seed(5)
    out_stoch, seq_stoch = identity_stoch_sampler_tmin_high.sample((1, N_ATOM, 3))
    assert torch.allclose(out_det, out_stoch)
    assert torch.equal(seq_det, seq_stoch)


# ---------------------------------------------------------------------------
# build_AA_context — constants
# ---------------------------------------------------------------------------

N_RES_AA = 6
N_ATOM_AA = N_RES_AA * NATOM  # 30
AA_SEQ_AA = "ACDEFG"  # len == N_RES_AA
N_ATOM_BINS = 22


# ---------------------------------------------------------------------------
# build_AA_context — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def atom37_pos() -> Float[torch.Tensor, "N_RES_AA 37 3"]:
    """Provide random atom37 coordinates of shape (N_RES_AA, 37, 3) for AA context tests."""
    return torch.randn(N_RES_AA, 37, 3)


@pytest.fixture
def atom37_mask_all() -> Float[torch.Tensor, "N_RES_AA 37"]:
    """Provide an all-ones atom37 mask indicating every atom is present."""
    return torch.ones(N_RES_AA, 37)


@pytest.fixture
def residue_idx_aa() -> Float[torch.Tensor, "N_RES_AA"]:
    """Provide a sequential residue index [0, 1, ..., N_RES_AA-1] for AA context tests."""
    return torch.arange(N_RES_AA, dtype=torch.float)


@pytest.fixture
def atom_disto_fn() -> Distogram:
    """Provide a standard atom distogram function with 22 bins spanning 2-22 Å."""
    return Distogram(n_bins=N_ATOM_BINS, min_dist=2.0, max_dist=22.0, overflow_bin=False)


@pytest.fixture
def aa_ctx(
    atom37_pos: Float[torch.Tensor, "N_RES_AA 37 3"],
    atom37_mask_all: Float[torch.Tensor, "N_RES_AA 37"],
    residue_idx_aa: Float[torch.Tensor, "N_RES_AA"],
    atom_disto_fn: Distogram,
) -> AllAtomContext:
    """Provide fully-populated AllAtomContext for a 6-residue diverse sequence with batch_size=2."""
    with torch.no_grad():
        return build_AA_context(
            atom_37_coordinate_tensor=atom37_pos,
            atom_37_mask=atom37_mask_all,
            residue_index=residue_idx_aa,
            aa_sequence=AA_SEQ_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=2,
            device="cpu",
        )


# ---------------------------------------------------------------------------
# build_AA_context — return type
# ---------------------------------------------------------------------------


def test_build_aa_context_returns_all_atom_context(aa_ctx: AllAtomContext):
    """Verify that build_AA_context returns an AllAtomContext, not a plain FeaturizedBatch."""
    assert isinstance(aa_ctx, AllAtomContext)


# ---------------------------------------------------------------------------
# build_AA_context — tensor shapes
# ---------------------------------------------------------------------------


def test_build_aa_context_r_gt_shape(aa_ctx: AllAtomContext):
    """Ground-truth atom coordinates must be batched to (batch_size, N_atom_AA, 3)."""
    assert aa_ctx.r_gt.shape == (2, N_ATOM_AA, 3)


def test_build_aa_context_atom5_mask_shape(aa_ctx: AllAtomContext):
    """The atom5 mask must cover all packed atoms across the batch: (batch_size, N_atom_AA)."""
    assert aa_ctx.atom5_mask.shape == (2, N_ATOM_AA)


def test_build_aa_context_residue_mask_shape(aa_ctx: AllAtomContext):
    """Residue mask must have one entry per residue per batch element: (batch_size, N_RES_AA)."""
    assert aa_ctx.residue_mask.shape == (2, N_RES_AA)


def test_build_aa_context_aa_indices_shape(aa_ctx: AllAtomContext):
    """Amino-acid indices encode one class per residue per batch element: (batch_size, N_RES_AA)."""
    assert aa_ctx.aa_indices.shape == (2, N_RES_AA)


def test_build_aa_context_f_residue_idx_shape(aa_ctx: AllAtomContext):
    """The residue index must have shape (batch_size, N_RES_AA)."""
    assert aa_ctx.f_residue_idx.shape == (2, N_RES_AA)


def test_build_aa_context_gt_atom_distogram_sparse_atom_dim(aa_ctx: AllAtomContext):
    """The sparse distogram atom axis must equal N_ATOM_AA (one row of neighbours per atom)."""
    assert aa_ctx.gt_atom_distogram_sparse.shape[1] == N_ATOM_AA


def test_build_aa_context_gt_atom_distogram_sparse_n_bins(aa_ctx: AllAtomContext):
    """Sparse distogram bin axis must match the N_ATOM_BINS provided to the distogram function."""
    assert aa_ctx.gt_atom_distogram_sparse.shape[3] == N_ATOM_BINS


def test_build_aa_context_gt_atom_distogram_mask_consistent_with_sparse(aa_ctx: AllAtomContext):
    """Sparse distogram mask must share the (B, N_atom, K) prefix shape of the distogram tensor."""
    B_dim, N_a, K, _ = aa_ctx.gt_atom_distogram_sparse.shape
    assert aa_ctx.gt_atom_distogram_mask_sparse.shape == (B_dim, N_a, K)


# ---------------------------------------------------------------------------
# build_AA_context — tensor values
# ---------------------------------------------------------------------------


def test_build_aa_context_r_gt_is_finite(aa_ctx: AllAtomContext):
    """Ground-truth coordinates must be finite; NaN would indicate coordinate preprocessing bug."""
    assert torch.isfinite(aa_ctx.r_gt).all()


def test_build_aa_context_f_residue_idx_is_finite(aa_ctx: AllAtomContext):
    """Residue indices must be finite integers for all positions in the sequence."""
    assert torch.isfinite(aa_ctx.f_residue_idx.float()).all()


def test_build_aa_context_gt_atom_distogram_sparse_is_finite(aa_ctx: AllAtomContext):
    """Binned atom distances must not produce NaN or Inf in the sparse distogram."""
    assert torch.isfinite(aa_ctx.gt_atom_distogram_sparse).all()


def test_build_aa_context_full_mask_gives_all_true_residue_mask(aa_ctx: AllAtomContext):
    """An all-ones atom37 mask must result in every residue being marked valid."""
    assert aa_ctx.residue_mask.all()


def test_build_aa_context_full_mask_gives_all_true_atom5_mask(aa_ctx: AllAtomContext):
    """An all-ones atom37 mask must produce an all-true atom5 mask after the 37→5 compression."""
    assert aa_ctx.atom5_mask.all()


def test_build_aa_context_aa_indices_match_restype_order(aa_ctx: AllAtomContext):
    """Each residue's index must match canonical restype_order lookup for its one-letter code."""
    expected = torch.tensor([restype_order[c] for c in AA_SEQ_AA], dtype=torch.long)
    assert torch.equal(aa_ctx.aa_indices[0], expected)


def test_build_aa_context_all_batch_slices_of_r_gt_are_identical(aa_ctx: AllAtomContext):
    """With batch_size=2 each batch ele is a copy of same input, so r_gt[0] must equal r_gt[1]."""
    assert torch.equal(aa_ctx.r_gt[0], aa_ctx.r_gt[1])


def test_build_aa_context_all_batch_slices_of_aa_indices_are_identical(aa_ctx: AllAtomContext):
    """With batch_size=2 batch slices encode the same sequence, so aa_indices identical across."""
    assert torch.equal(aa_ctx.aa_indices[0], aa_ctx.aa_indices[1])


def test_build_aa_context_batch_size_controls_leading_dim(
    atom37_pos: Float[torch.Tensor, "N_RES_AA 37 3"],
    atom37_mask_all: Float[torch.Tensor, "N_RES_AA 37"],
    residue_idx_aa: Float[torch.Tensor, "N_RES_AA"],
    atom_disto_fn: Distogram,
):
    """Passing batch_size=3 must set the leading dimension of r_gt and aa_indices to 3."""
    with torch.no_grad():
        ctx = build_AA_context(
            atom37_pos,
            atom37_mask_all,
            residue_idx_aa,
            AA_SEQ_AA,
            atom_disto_fn,
            batch_size=3,
            device="cpu",
        )
    assert ctx.r_gt.shape[0] == 3
    assert ctx.aa_indices.shape[0] == 3


# ---------------------------------------------------------------------------
# build_template_context — constants and helpers
# ---------------------------------------------------------------------------

N_TEMPL_BINS = 39  # 38 distance bins + 1 overflow bin


def _make_protein(n_res: int, mask_value: float = 1.0) -> Protein:
    """Build a minimal Protein with random coordinates and a uniform atom mask."""
    rng = np.random.RandomState(42)
    return Protein(
        atom_positions=rng.randn(n_res, 37, 3).astype(np.float64),
        aatype=np.zeros(n_res, dtype=np.intp),
        atom_mask=np.full((n_res, 37), mask_value, dtype=np.float64),
        residue_index=np.arange(n_res, dtype=np.intp),
        chain_index=np.zeros(n_res, dtype=np.intp),
        b_factors=np.ones((n_res, 37), dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# build_template_context — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def np_protein() -> Protein:
    """Provide full-length (N_RES residues) Protein with random coordinates and all-present mask."""
    return _make_protein(N_RES)


@pytest.fixture
def np_protein_short() -> Protein:
    """Shorter protein fixture to exercise padding in batched template contexts."""
    return _make_protein(N_RES - 2)  # shorter, for padding test


@pytest.fixture
def templ_disto() -> Distogram:
    """Provide a template distogram function: 38 bins + overflow, spanning 3.25-50.75 Å."""
    return Distogram(n_bins=N_TEMPL_BINS - 1, min_dist=3.25, max_dist=50.75, overflow_bin=True)


@pytest.fixture
def template_ctx(np_protein: Protein, templ_disto: Distogram) -> TemplateContext:
    """Provide single-template TemplateContext built from np_protein and template distogram."""
    return build_template_context(np_protein, batch_size=1, distogram_fn=templ_disto, device="cpu")


# ---------------------------------------------------------------------------
# build_template_context — return type
# ---------------------------------------------------------------------------


def test_build_template_context_returns_template_context(template_ctx: TemplateContext):
    """Verify that build_template_context returns a TemplateContext dataclass, not a plain dict."""
    assert isinstance(template_ctx, TemplateContext)


# ---------------------------------------------------------------------------
# build_template_context — tensor shapes
# ---------------------------------------------------------------------------


def test_build_template_context_f_template_distogram_shape(template_ctx: TemplateContext):
    """A single-protein input must yield a distogram of shape (1, N_RES, N_RES, N_TEMPL_BINS)."""
    assert template_ctx.f_template_distogram.shape == (1, N_RES, N_RES, N_TEMPL_BINS)


def test_build_template_context_f_pseudo_beta_mask_shape(template_ctx: TemplateContext):
    """The pseudo-β mask must have one entry per residue per template: shape (1, N_RES)."""
    assert template_ctx.f_pseudo_beta_mask.shape == (1, N_RES)


# ---------------------------------------------------------------------------
# build_template_context — tensor dtypes
# ---------------------------------------------------------------------------


def test_build_template_context_f_template_distogram_dtype_is_long(template_ctx: TemplateContext):
    """Template distogram stored as long (class indices), not float, for embedding lookup."""
    assert template_ctx.f_template_distogram.dtype == torch.long


def test_build_template_context_f_pseudo_beta_mask_dtype_is_long(template_ctx: TemplateContext):
    """The pseudo-β mask stored as long (binary integer), consistent with the distogram dtype."""
    assert template_ctx.f_pseudo_beta_mask.dtype == torch.long


# ---------------------------------------------------------------------------
# build_template_context — distogram properties
# ---------------------------------------------------------------------------


def test_build_template_context_distogram_is_valid_one_hot(template_ctx: TemplateContext):
    """Each (i, j) distogram entry is valid one-hot bin index: bin sums equal 1.0."""
    bin_sums = reduce(template_ctx.f_template_distogram.float(), "b i j bins -> b i j", "sum")
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums))


def test_build_template_context_distogram_is_symmetric(template_ctx: TemplateContext):
    """The Cβ-distance distogram must be symmetric: d(i, j) == d(j, i) for all residue pairs."""
    disto = template_ctx.f_template_distogram
    assert torch.equal(disto, rearrange(disto, "b i j k -> b j i k"))


def test_build_template_context_different_proteins_give_different_distograms(
    templ_disto: Distogram,
):
    """Two distinct proteins produce different distograms, confirming coordinate sensitivity."""
    rng_b = np.random.RandomState(2)
    prot_a = _make_protein(N_RES)
    prot_b = Protein(
        atom_positions=(rng_b.randn(N_RES, 37, 3) * 20).astype(np.float64),
        aatype=np.zeros(N_RES, dtype=np.intp),
        atom_mask=np.ones((N_RES, 37), dtype=np.float64),
        residue_index=np.arange(N_RES, dtype=np.intp),
        chain_index=np.zeros(N_RES, dtype=np.intp),
        b_factors=np.ones((N_RES, 37), dtype=np.float64),
    )
    ctx_a = build_template_context(prot_a, batch_size=1, distogram_fn=templ_disto, device="cpu")
    ctx_b = build_template_context(prot_b, batch_size=1, distogram_fn=templ_disto, device="cpu")
    assert not torch.equal(ctx_a.f_template_distogram, ctx_b.f_template_distogram)


# ---------------------------------------------------------------------------
# build_template_context — mask values
# ---------------------------------------------------------------------------


def test_build_template_context_full_ca_mask_gives_all_ones_pseudo_beta_mask(
    template_ctx: TemplateContext,
):
    """When all atoms (including alpha C) are present, every residue has pseudo-β mask value 1."""
    assert (template_ctx.f_pseudo_beta_mask == 1).all()


# ===========================================================================
# Use-case tests for build_sampling_context
#
# Six conditioning scenarios that cover the intended sampling API.
# Each fixture encodes exactly one conditioning regime; tests verify only the
# invariants that distinguish that regime from the others.
# ===========================================================================

# Shared constants for this section
SEQ_4 = "ACDE"  # 4-character diverse sequence (N_RES = 4)
N_PARTIAL = N_RES // 2  # 2 — residues covered by the partial template / partial atoms

# ---------------------------------------------------------------------------
# PDB file fixtures (written to tmp_path so protein_from_pdb can read them)
# ---------------------------------------------------------------------------


@pytest.fixture
def full_template_pdb(tmp_path: pathlib.Path) -> str:
    """Write a full-length (N_RES residues) template PDB to disk and return its path."""
    pdb_path = tmp_path / "full_template.pdb"
    pdb_path.write_text(to_pdb(_make_protein(N_RES)))
    return str(pdb_path)


@pytest.fixture
def partial_template_pdb(tmp_path: pathlib.Path) -> str:
    """PDB with only N_PARTIAL residues — covers a subset of the target N_RES."""
    pdb_path = tmp_path / "partial_template.pdb"
    pdb_path.write_text(to_pdb(_make_protein(N_PARTIAL)))
    return str(pdb_path)


# ---------------------------------------------------------------------------
# Partial-atom helpers (first half present, second half zero)
# ---------------------------------------------------------------------------


@pytest.fixture
def partial_atom_pos() -> Float[torch.Tensor, "N_RES 37 3"]:
    """Provide atom37 positions where only first N_PARTIAL residues have non-zero coordinates."""
    pos = torch.zeros(N_RES, 37, 3)
    pos[:N_PARTIAL] = torch.randn(N_PARTIAL, 37, 3, generator=torch.Generator().manual_seed(10))
    return pos


@pytest.fixture
def partial_atom_msk() -> Float[torch.Tensor, "N_RES 37"]:
    """Provide an atom37 mask that marks only the first N_PARTIAL residues as present."""
    mask = torch.zeros(N_RES, 37)
    mask[:N_PARTIAL] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Use-case fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unconditional_ctx(atom_disto_fn: Distogram, templ_disto: Distogram) -> FeaturizedBatch:
    """Use case 1: no sequence, no atoms, no template."""
    with torch.no_grad():
        return build_sampling_context(
            atom_positions=torch.zeros(N_RES, 37, 3),
            atom_mask=torch.zeros(N_RES, 37),
            residue_index=torch.zeros(N_RES),
            seq="A" * N_RES,
            pdb_file_path=None,
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def seq_only_ctx(atom_disto_fn: Distogram, templ_disto: Distogram) -> FeaturizedBatch:
    """Use case 2: amino-acid sequence + residue positions, no atoms, no template."""
    with torch.no_grad():
        return build_sampling_context(
            atom_positions=torch.zeros(N_RES, 37, 3),
            atom_mask=torch.zeros(N_RES, 37),
            residue_index=torch.arange(N_RES, dtype=torch.float),
            seq=SEQ_4,
            pdb_file_path=None,
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def seq_partial_atoms_ctx(
    partial_atom_pos: Float[torch.Tensor, "N_RES 37 3"],
    partial_atom_msk: Float[torch.Tensor, "N_RES 37"],
    atom_disto_fn: Distogram,
    templ_disto: Distogram,
) -> FeaturizedBatch:
    """Use case 3: sequence + partial structural info (first half of residues), no template."""
    with torch.no_grad():
        return build_sampling_context(
            atom_positions=partial_atom_pos,
            atom_mask=partial_atom_msk,
            residue_index=torch.arange(N_RES, dtype=torch.float),
            seq=SEQ_4,
            pdb_file_path=None,
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto,
            batch_size=1,
            device="cpu",
        )


# @pytest.fixture
# def partial_templ_no_seq_ctx(
#     partial_template_pdb: str, atom_disto_fn: Distogram, templ_disto: Distogram
# ) -> FeaturizedBatch:
#     """Use case 4: partial template (N_PARTIAL < N_RES residues), no sequence, no atoms."""
#     with torch.no_grad():
#         return build_sampling_context(
#             atom_positions=torch.zeros(N_RES, 37, 3),
#             atom_mask=torch.zeros(N_RES, 37),
#             residue_index=torch.zeros(N_RES),
#             seq="A" * N_RES,
#             pdb_files=[partial_template_pdb],
#             atom_distogram_fn=atom_disto_fn,
#             templ_distogram_fn=templ_disto,
#         )


# @pytest.fixture
# def full_templ_no_seq_ctx(
#     full_template_pdb: str, atom_disto_fn: Distogram, templ_disto: Distogram
# ) -> FeaturizedBatch:
#     """Use case 5: full template (all N_RES residues), no sequence, no atoms."""
#     with torch.no_grad():
#         return build_sampling_context(
#             atom_positions=torch.zeros(N_RES, 37, 3),
#             atom_mask=torch.zeros(N_RES, 37),
#             residue_index=torch.zeros(N_RES),
#             seq="A" * N_RES,
#             pdb_files=[full_template_pdb],
#             atom_distogram_fn=atom_disto_fn,
#             templ_distogram_fn=templ_disto,
#         )


# @pytest.fixture
# def full_atoms_partial_templ_ctx(
#     partial_template_pdb: str, atom_disto_fn: Distogram, templ_disto: Distogram
# ) -> FeaturizedBatch:
#     """Use case 6: all atoms present, partial template (N_PARTIAL < N_RES), no sequence."""
#     with torch.no_grad():
#         return build_sampling_context(
#             atom_positions=torch.randn(N_RES, 37, 3, generator=torch.Generator().manual_seed(42)),
#             atom_mask=torch.ones(N_RES, 37),
#             residue_index=torch.arange(N_RES, dtype=torch.float),
#             seq="A" * N_RES,
#             pdb_files=[partial_template_pdb],
#             atom_distogram_fn=atom_disto_fn,
#             templ_distogram_fn=templ_disto,
#         )


# ---------------------------------------------------------------------------
# Use case 1 — unconditional sampling
# ---------------------------------------------------------------------------


def test_unconditional_gt_res_distogram_all_zeros(unconditional_ctx: FeaturizedBatch):
    """Without any template, the residue distogram conditioning tensor must be entirely zero."""
    assert (unconditional_ctx.gt_res_distogram == 0).all()


def test_unconditional_f_pseudo_beta_mask_all_zeros(unconditional_ctx: FeaturizedBatch):
    """Without any template, no residue has pseudo-β coverage so the mask must be all-zero."""
    assert (unconditional_ctx.f_pseudo_beta_mask == 0).all()


def test_unconditional_aa_indices_all_alanine(unconditional_ctx: FeaturizedBatch):
    """All-alanine placeholder sequence maps every residue to index 0 in amino-acid vocabulary."""
    # "A" maps to restype_order index 0 — no sequence information provided
    assert (unconditional_ctx.aa_indices == 0).all()


def test_unconditional_atom5_mask_all_false(unconditional_ctx: FeaturizedBatch):
    """A zero atom_mask, yields atom5_mask full of entirely False elements."""
    # Zero atom_mask → no atoms present in any residue
    assert not unconditional_ctx.atom5_mask.any()


def test_unconditional_r_gt_all_zeros(unconditional_ctx: FeaturizedBatch):
    """With zero atom positions as input, ground-truth coordinate tensor must also be all-zero."""
    assert (unconditional_ctx.r_gt == 0).all()


# ---------------------------------------------------------------------------
# Use case 2 — conditional on amino-acid sequence alone
# ---------------------------------------------------------------------------


def test_seq_only_aa_indices_match_sequence(seq_only_ctx: FeaturizedBatch):
    """The aa_indices tensor must encode each amino acid in SEQ_4 via restype_order mapping."""
    expected = torch.tensor([restype_order[c] for c in SEQ_4], dtype=torch.long)
    assert torch.equal(seq_only_ctx.aa_indices[0], expected)


def test_seq_only_aa_indices_not_all_alanine(seq_only_ctx: FeaturizedBatch):
    """Four-residue sequence (ACDE) produces non-zero amino-acid indices."""
    # "ACDE" encodes to [0, 4, 3, 6] — not all-zero
    assert not (seq_only_ctx.aa_indices == 0).all()


def test_seq_only_gt_res_distogram_all_zeros(seq_only_ctx: FeaturizedBatch):
    """Sequence-only conditioning provides no structural template, distogram remains all-zero."""
    assert (seq_only_ctx.gt_res_distogram == 0).all()


def test_seq_only_f_pseudo_beta_mask_all_zeros(seq_only_ctx: FeaturizedBatch):
    """Without template PDB, pseudo-β mask must be all-zero even when a sequence is provided."""
    assert (seq_only_ctx.f_pseudo_beta_mask == 0).all()


def test_seq_only_atom5_mask_all_false(seq_only_ctx: FeaturizedBatch):
    """Providing only a sequence (no atom coordinates) must leave the atom5 mask entirely False."""
    assert not seq_only_ctx.atom5_mask.any()


def test_seq_only_residue_idx_embedding_differs_from_unconditional(
    seq_only_ctx: FeaturizedBatch, unconditional_ctx: FeaturizedBatch
):
    """Sequential residue indices (arange) produce a different embedding than all-zero indices."""
    # seq_only uses arange residue_index → distinct embedding per position;
    # unconditional uses zeros → same token repeated — the two must differ
    assert not torch.equal(seq_only_ctx.f_residue_idx, unconditional_ctx.f_residue_idx)


# ---------------------------------------------------------------------------
# Use case 3 — conditional on sequence + partial structural information
# ---------------------------------------------------------------------------


def test_seq_partial_atoms_aa_indices_match_sequence(seq_partial_atoms_ctx: FeaturizedBatch):
    """Sequence conditioning correctly encode SEQ_4 even when structural coverage is partial."""
    expected = torch.tensor([restype_order[c] for c in SEQ_4], dtype=torch.long)
    assert torch.equal(seq_partial_atoms_ctx.aa_indices[0], expected)


def test_seq_partial_atoms_atom5_mask_true_for_first_half(seq_partial_atoms_ctx: FeaturizedBatch):
    """First N_PARTIAL residues have atoms present, so their atom5 mask entries must all be True."""
    assert seq_partial_atoms_ctx.atom5_mask[0, : N_PARTIAL * NATOM].all()


def test_seq_partial_atoms_atom5_mask_false_for_second_half(seq_partial_atoms_ctx: FeaturizedBatch):
    """The second half of residues have no atoms, so their atom5 mask entries must all be False."""
    assert not seq_partial_atoms_ctx.atom5_mask[0, N_PARTIAL * NATOM :].any()


def test_seq_partial_atoms_r_gt_nonzero_for_valid_residues(seq_partial_atoms_ctx: FeaturizedBatch):
    """Ground-truth coordinates for residues with atoms contain at least some non-zero values."""
    assert not (seq_partial_atoms_ctx.r_gt[0, : N_PARTIAL * NATOM] == 0).all()


def test_seq_partial_atoms_r_gt_zero_for_absent_residues(seq_partial_atoms_ctx: FeaturizedBatch):
    """The ground-truth coordinates for residues without atoms must be exactly zero."""
    assert (seq_partial_atoms_ctx.r_gt[0, N_PARTIAL * NATOM :] == 0).all()


def test_seq_partial_atoms_no_template_distogram(seq_partial_atoms_ctx: FeaturizedBatch):
    """Without PDB template, both distogram and pseudo-β mask must be zero regardless of atoms."""
    assert (seq_partial_atoms_ctx.gt_res_distogram == 0).all()
    assert (seq_partial_atoms_ctx.f_pseudo_beta_mask == 0).all()


# ---------------------------------------------------------------------------
# Use cases 4, 5, 6 — template-conditioned scenarios
# (fixtures partial_templ_no_seq_ctx / full_templ_no_seq_ctx /
#  full_atoms_partial_templ_ctx are not yet implemented for the new API)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_AA_context — 'X' residue handling
# ---------------------------------------------------------------------------


def test_build_aa_context_x_sequence_does_not_raise(
    atom_disto_fn: Distogram, residue_idx_aa: Float[torch.Tensor, "N_RES_AA"]
):
    """All-unknown ('X') sequence is accepted by build_AA_context without raising any exception."""
    with torch.no_grad():
        ctx = build_AA_context(
            atom_37_coordinate_tensor=torch.zeros(N_RES_AA, 37, 3),
            atom_37_mask=torch.zeros(N_RES_AA, 37),
            residue_index=residue_idx_aa,
            aa_sequence="X" * N_RES_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
        )
    assert isinstance(ctx, AllAtomContext)


def test_build_aa_context_x_maps_to_index_20(
    atom_disto_fn: Distogram, residue_idx_aa: Float[torch.Tensor, "N_RES_AA"]
):
    """Unknown residue 'X' maps to index 20 (the 21st slot beyond the 20 standard amino acids)."""
    with torch.no_grad():
        ctx = build_AA_context(
            atom_37_coordinate_tensor=torch.zeros(N_RES_AA, 37, 3),
            atom_37_mask=torch.zeros(N_RES_AA, 37),
            residue_index=residue_idx_aa,
            aa_sequence="X" * N_RES_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
        )
    assert (ctx.aa_indices == 20).all()


def test_build_aa_context_x_is_distinct_from_alanine(
    atom_disto_fn: Distogram, residue_idx_aa: Float[torch.Tensor, "N_RES_AA"]
):
    """Unknown residue 'X' (index 20) produces different aa_indices than alanine (index 0)."""
    with torch.no_grad():
        ctx_x = build_AA_context(
            atom_37_coordinate_tensor=torch.zeros(N_RES_AA, 37, 3),
            atom_37_mask=torch.zeros(N_RES_AA, 37),
            residue_index=residue_idx_aa,
            aa_sequence="X" * N_RES_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
        )
        ctx_a = build_AA_context(
            atom_37_coordinate_tensor=torch.zeros(N_RES_AA, 37, 3),
            atom_37_mask=torch.zeros(N_RES_AA, 37),
            residue_index=residue_idx_aa,
            aa_sequence="A" * N_RES_AA,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
        )
    assert not torch.equal(ctx_x.aa_indices, ctx_a.aa_indices)


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_all_atom_context_wrong_shape() -> None:
    """Wrong r_gt ndim (2-D instead of 3-D) triggers TypeCheckError."""
    b_local, n_atom_local, n_res_local, k_local = 2, N_ATOM, N_RES, 4
    with pytest.raises(TypeCheckError):
        AllAtomContext(
            r_gt=torch.zeros(b_local, n_atom_local),  # must be 3-D "B N_atom 3"
            atom5_mask=torch.zeros(b_local, n_atom_local, dtype=torch.bool),
            residue_mask=torch.zeros(b_local, n_res_local, dtype=torch.bool),
            gt_atom_distogram_sparse=torch.zeros(b_local, n_atom_local, k_local, N_ATOM_BINS),
            gt_atom_distogram_mask_sparse=torch.zeros(
                b_local, n_atom_local, k_local, dtype=torch.bool
            ),
            aa_indices=torch.zeros(b_local, n_res_local, dtype=torch.long),
            f_residue_idx=torch.zeros(b_local, n_res_local, dtype=torch.long),
        )


def test_build_aa_context_wrong_shape(atom_disto_fn: Distogram) -> None:
    """Wrong coord last dim (4 instead of 3) triggers TypeCheckError."""
    coord_bad = torch.zeros(N_RES, 37, 4)  # last dim must be 3
    with pytest.raises(TypeCheckError):
        build_AA_context(
            atom_37_coordinate_tensor=coord_bad,
            atom_37_mask=torch.ones(N_RES, 37),
            residue_index=torch.arange(N_RES, dtype=torch.float),
            aa_sequence="A" * N_RES,
            atom_distogram_fn=atom_disto_fn,
            batch_size=1,
            device="cpu",
        )


def test_template_context_wrong_shape() -> None:
    """Wrong f_template_distogram ndim (3-D instead of 4-D) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        TemplateContext(
            f_template_distogram=torch.zeros(1, N_RES, N_RES, dtype=torch.long),  # missing bins dim
            f_pseudo_beta_mask=torch.zeros(1, N_RES, dtype=torch.long),
        )


def test_build_template_context_wrong_type(templ_disto: Distogram) -> None:
    """Non-Protein argument triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        build_template_context(42, batch_size=1, distogram_fn=templ_disto, device="cpu")  # type: ignore[arg-type]


def test_build_sampling_context_wrong_shape(
    atom_disto_fn: Distogram, templ_disto: Distogram
) -> None:
    """Wrong atom_positions last dim (4 instead of 3) triggers TypeCheckError."""
    positions_bad = torch.zeros(N_RES, 37, 4)  # last dim must be 3
    with pytest.raises(TypeCheckError):
        build_sampling_context(
            atom_positions=positions_bad,
            atom_mask=torch.ones(N_RES, 37),
            residue_index=torch.arange(N_RES, dtype=torch.float),
            seq="A" * N_RES,
            pdb_file_path=None,
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto,
            batch_size=1,
            device="cpu",
        )


def test_noise_schedule_wrong_type(bare_sampler: EDMSampler) -> None:
    """Passing a plain float instead of a scalar tensor triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        bare_sampler.noise_schedule(3.14)  # type: ignore[arg-type]


def test_edm_sampler_sample_wrong_type(edm_sampler: EDMSampler) -> None:
    """Tuple with non-int element triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        edm_sampler.sample(shape=(1, N_ATOM, "bad"))  # type: ignore[arg-type]


def test_atom5_to_atom37_wrong_shape() -> None:
    """Wrong coords_5 last dim (4 instead of 3) triggers TypeCheckError."""
    coords_bad = torch.zeros(N_RES, 5, 4)  # last dim must be 3
    with pytest.raises(TypeCheckError):
        atom5_to_atom37(coords_bad)

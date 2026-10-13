"""Tests for the conditional sampling loop."""

import dataclasses
from enum import Enum, Flag, auto
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from architecture.main_trunk import PredictedOutputs
from beartype import beartype
from einops import rearrange, reduce
from helpers.alignment import masked_com
from helpers.atom_utils import RESTYPE_NUM_NO_X, Protein, restype_order
from helpers.data import Distogram, FeaturizedBatch
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, TypeCheckError, jaxtyped
from sample.sample_config import SamplerParams
from sample.sampling import (
    NATOM,
    AllAtomContext,
    EDMSampler,
    TemplateContext,
    build_aa_context,
    build_sampling_context,
    build_template_context,
)
from train.train_config import NoiseScheduleParams

_ = manual_seed(0)

CPU_DEVICE_TYPE = "cpu"
B = 2
N_RES = 4
N_ATOM = N_RES * NATOM  # 20
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
N_ATOM_BINS = 22
T_HAT_INIT = 1.0
T_NORM_INIT = 0.5

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
            r_denoised=batch.r_gt_noised.clone(),
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
def context(
    atom_disto_fn: Distogram,
    templ_disto: Distogram,
) -> FeaturizedBatch:
    """Unconditional sampling context (all-alanine, zero atom positions)."""
    with torch.no_grad():
        return build_sampling_context(
            pdb_file_path=None,
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto,
            residue_number=N_RES,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def large_context(
    atom_disto_fn: Distogram,
    templ_disto: Distogram,
) -> FeaturizedBatch:
    """Sampling context with doubled residue count (2 * N_RES)."""
    with torch.no_grad():
        return build_sampling_context(
            pdb_file_path=None,
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto,
            residue_number=N_RES * 2,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def templ_disto_19() -> Distogram:
    """Template distogram with 19 bins spanning 3.25-50.75 Å."""
    return Distogram(
        n_bins=19,
        min_dist=3.25,
        max_dist=50.75,
        overflow_bin=True,
    )


@pytest.fixture
def context_19_bins(
    atom_disto_fn: Distogram,
    templ_disto_19: Distogram,
) -> FeaturizedBatch:
    """Sampling context built with a 19-bin template distogram."""
    with torch.no_grad():
        return build_sampling_context(
            pdb_file_path=None,
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto_19,
            residue_number=N_RES,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def trunk_mock() -> MagicMock:
    """Identity trunk mock that echoes r_input as the denoised output."""
    return _make_trunk_mock()


@pytest.fixture
def edm_sampler(
    trunk_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> EDMSampler:
    """Deterministic EDMSampler (S_churn=0) built on the identity trunk mock."""
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
    """MagicMock trunk that scales noised coordinates by 0.5.

    Unlike the identity or zero denoisers, this produces a non-trivial
    trajectory whose output depends on the full sequence of sigma values,
    making it suitable for step-count and schedule sensitivity tests.
    """
    mock = MagicMock()

    def half_side_effect(batch: FeaturizedBatch) -> PredictedOutputs:
        """Return 0.5 * r_gt as denoised coords and zero sequence logits."""
        B_local = batch.r_gt.shape[0]
        n_atom = batch.r_gt.shape[1]
        n_res = int(batch.tok_idx.max().item()) + 1
        return PredictedOutputs(
            r_denoised=batch.r_gt_noised.clone() * 0.5,
            seq_logits=torch.zeros(B_local, n_res, 20),
            residue_distogram_logits=torch.zeros(B_local, n_res, n_res, 38),
            atom_distogram_logits=torch.zeros(B_local, n_atom, 1, 38),
            intermediate_denoised_coord_stack=[],
            intermediate_pred_aa_logit_stack=[],
        )

    mock.side_effect = half_side_effect
    return mock


@pytest.fixture
def identity_det_sampler(
    identity_denoiser_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> EDMSampler:
    """Deterministic EDMSampler (S_churn=0) using the identity denoiser."""
    return EDMSampler(
        identity_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )


@pytest.fixture
def identity_stoch_sampler_tmin_high(
    identity_denoiser_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> EDMSampler:
    """Stochastic EDMSampler, noise injection suppressed by a high S_tmin.

    Sets ``S_churn=2.0`` (non-zero stochasticity) but ``S_tmin=SIGMA_MAX*10``.
    The churn gate compares ``S_tmin`` with the schedule time
    ``timestep / ddim_steps``, which never exceeds 1, so the gate
    ``S_tmin ≤ timestep / ddim_steps`` is never satisfied and no extra noise
    is added. Any ``S_tmin`` above 1 would do; this value is simply far
    outside the window. Used to verify that the sampler degenerates to the
    deterministic trajectory when the window is empty.
    """
    return EDMSampler(
        identity_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=2.0, S_tmin=SIGMA_MAX * 10),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )


# ---------------------------------------------------------------------------
# build_sampling_context — shapes and field values
# ---------------------------------------------------------------------------


def test_build_sampling_context(context: FeaturizedBatch) -> None:
    """Verify type, shapes, finiteness, index mappings, and defaults.

    Groups checks into three passes: a single dict-comparison for selected
    tensor shapes, a loop that asserts every tensor field is finite, and a
    loop over pre-computed boolean checks for value-level invariants (index
    mappings, distogram defaults, masks, and timing scalars).
    """
    assert isinstance(context, FeaturizedBatch)

    expected_shapes: dict[str, tuple[int, ...]] = {
        "ref_pos": (1, N_ATOM, 3),
        "ref_element": (1, N_ATOM, 4),
        "ref_space_uid": (1, N_ATOM),
        "noised_res_distogram": (1, N_RES, N_RES, N_TEMPL_BINS),
        "f_pseudo_beta_mask": (1, N_RES),
        "f_residue_idx": (1, N_RES),
        "tok_idx": (1, N_ATOM),
        "center_uid": (1, N_ATOM),
    }
    item_dict: dict[str, object] = dataclasses.asdict(context)
    assert {
        k: tuple(cast(torch.Tensor, item_dict[k]).shape)
        for k in expected_shapes
    } == expected_shapes

    for field_name, val in item_dict.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(
                val.float(),
            ).all(), f"non-finite in field '{field_name}'"

    row_sums = reduce(context.ref_element, "b n_atom e -> b n_atom", "sum")
    expected_uid = torch.arange(N_RES).repeat_interleave(NATOM)
    expected_ca = torch.arange(N_RES).repeat_interleave(NATOM) * NATOM + 1
    value_checks: list[tuple[bool, str]] = [
        (
            bool(torch.allclose(row_sums, torch.ones(1, N_ATOM))),
            "ref_element rows sum to 1",
        ),
        (
            bool(torch.equal(context.ref_space_uid[0], expected_uid)),
            "ref_space_uid mapping",
        ),
        (
            bool(torch.equal(context.tok_idx[0], expected_uid)),
            "tok_idx mapping",
        ),
        (
            bool(torch.equal(context.center_uid[0], expected_ca)),
            "center_uid mapping",
        ),
        (
            bool((context.noised_res_distogram.argmax(dim=-1) == 0).all()),
            "distogram peaks at bin 0",
        ),
        (
            bool((context.f_pseudo_beta_mask == 1).all()),
            "f_pseudo_beta_mask all ones",
        ),
        (bool(context.atom5_mask.all()), "atom5_mask all set"),
        (bool((context.t_hat == T_HAT_INIT).all()), "t_hat equals T_HAT_INIT"),
        (
            bool((context.t_normalized == T_NORM_INIT).all()),
            "t_normalized equals T_NORM_INIT",
        ),
        (
            int(context.gt_atom_distogram_sparse.max().item()) < N_ATOM_BINS,
            "sparse distogram bin indices in range",
        ),
    ]
    for ok, msg in value_checks:
        assert ok, msg


# ---------------------------------------------------------------------------
# build_sampling_context — residue scaling
# ---------------------------------------------------------------------------


def test_build_sampling_context_scales_with_residue_number(
    context: FeaturizedBatch,
    large_context: FeaturizedBatch,
) -> None:
    """Doubling residue_number doubles atom and token tensor widths.

    Asserts that ``ref_pos``, ``tok_idx``, and ``center_uid`` each have twice
    as many atoms in their spatial dimension as the base ``N_RES`` context.
    """
    assert large_context.ref_pos.shape[1] == 2 * context.ref_pos.shape[1]
    assert large_context.tok_idx.shape[1] == 2 * context.tok_idx.shape[1]
    assert large_context.center_uid.shape[1] == 2 * context.center_uid.shape[1]


# ---------------------------------------------------------------------------
# build_sampling_context — distogram bin-count propagation
# ---------------------------------------------------------------------------


def test_build_sampling_context_bin_count_propagation(
    context_19_bins: FeaturizedBatch,
) -> None:
    """Template distogram bin count propagates into noised_res_distogram.

    A 19-bin ``templ_distogram_fn`` plus its overflow bin must yield a
    ``noised_res_distogram`` with last dimension 20, confirming that n_bins +
    overflow_bin flows through to the stored distogram tensor.
    """
    assert context_19_bins.noised_res_distogram.shape == (1, N_RES, N_RES, 20)


# ---------------------------------------------------------------------------
# EDMSampler.noise_schedule — boundary values and properties
# ---------------------------------------------------------------------------


class NoiseScheduleScenario(Enum):
    """Noise schedule configurations whose time axis must run min to max.

    Each member is a ``(sigma_data, sigma_min, sigma_max, rho)`` tuple read
    through properties. The ``noise_schedule_curve`` factory fixture builds an
    ``EDMSampler`` from those values; the expected endpoints are the same
    values multiplied out, so they do not reimplement the schedule formula.
    Members are read via their properties rather than branched on.

    Members:
        pallatom_default: The training defaults (sigma_data 16, sigma_min
            4e-4, sigma_max 160, rho 7), i.e. the schedule production
            sampling uses.
        unit_sigma_data: sigma_data of 1, so the endpoints are the raw
            sigma_min and sigma_max; catches a missing or doubled
            sigma_data factor that the default scale would hide.
        low_rho: rho of 3. The interior of the curve changes with rho but
            the endpoints must not.
    """

    pallatom_default = (16.0, 4e-4, 160.0, 7.0)
    unit_sigma_data = (1.0, 0.002, 80.0, 7.0)
    low_rho = (16.0, 4e-4, 160.0, 3.0)

    @property
    def sigma_data(self) -> float:
        """Data standard deviation scaling every noise level."""
        return self.value[0]

    @property
    def sigma_min(self) -> float:
        """Smallest unscaled noise level."""
        return self.value[1]

    @property
    def sigma_max(self) -> float:
        """Largest unscaled noise level."""
        return self.value[2]

    @property
    def rho(self) -> float:
        """Schedule exponent."""
        return self.value[3]

    @property
    def expected_sigma_at_zero(self) -> float:
        """Noise level the schedule must give at t = 0 (minimum noise)."""
        return self.sigma_data * self.sigma_min

    @property
    def expected_sigma_at_one(self) -> float:
        """Noise level the schedule must give at t = 1 (maximum noise)."""
        return self.sigma_data * self.sigma_max


@pytest.fixture
def noise_schedule_curve(
    request: pytest.FixtureRequest,
    zero_denoiser_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> Float[torch.Tensor, "n_t"]:
    """Evaluate one scenario's noise schedule on an even grid over [0, 1].

    Factory fixture: receives a ``NoiseScheduleScenario`` through
    ``@pytest.mark.parametrize(..., indirect=["noise_schedule_curve"])``.
    The grid has 21 points, so it includes both endpoints exactly and
    steps of 0.05 between them. The zero denoiser is never called;
    ``noise_schedule`` only reads the sampler's schedule parameters.

    Args:
        request: Pytest request carrying the ``NoiseScheduleScenario``.
        zero_denoiser_mock: Trunk stand-in required to construct the sampler.
        context: Unconditional sampling context required by the sampler.
        templ_disto: Template distogram module required by the sampler.

    Returns:
        Noise levels at ``t = 0, 0.05, ..., 1``, in grid order.
    """
    scenario = cast(NoiseScheduleScenario, request.param)
    n_grid_points = 21
    sampler = EDMSampler(
        zero_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(rho=scenario.rho),
        NoiseScheduleParams(
            sigma_data=scenario.sigma_data,
            sigma_min=scenario.sigma_min,
            sigma_max=scenario.sigma_max,
        ),
    )
    times = torch.linspace(0.0, 1.0, n_grid_points)
    return torch.stack([sampler.noise_schedule(t) for t in times])


@pytest.mark.parametrize(
    ("noise_schedule_curve", "scenario"),
    [pytest.param(s, s, id=s.name) for s in NoiseScheduleScenario],
    indirect=["noise_schedule_curve"],
)
def test_noise_schedule_runs_from_sigma_min_at_zero_to_sigma_max_at_one(
    noise_schedule_curve: Float[torch.Tensor, "n_t"],
    scenario: NoiseScheduleScenario,
) -> None:
    """noise_schedule maps t = 0 to minimum noise and t = 1 to maximum noise.

    Regression for a sampler that mixed two time conventions: the schedule
    put maximum noise at t = 0 (AF3) while the initial noise level and the
    Euler target assumed maximum noise at t = 1 (Pallatom), so sampling
    started from the bottom of the schedule.

    Arrange: ``noise_schedule_curve`` evaluates the scenario's schedule on
    an even grid from t = 0 to t = 1.
    Act: read the first and last grid values.
    Assert: the first equals ``sigma_data * sigma_min`` and the last equals
    ``sigma_data * sigma_max``, to float32 precision.

    Args:
        noise_schedule_curve: Noise levels on the grid for this scenario.
        scenario: Schedule parameters and expected endpoint values.
    """
    torch.testing.assert_close(
        float(noise_schedule_curve[0]),
        scenario.expected_sigma_at_zero,
        rtol=1e-5,
        atol=0.0,
    )
    torch.testing.assert_close(
        float(noise_schedule_curve[-1]),
        scenario.expected_sigma_at_one,
        rtol=1e-5,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "noise_schedule_curve",
    [pytest.param(s, id=s.name) for s in NoiseScheduleScenario],
    indirect=True,
)
def test_noise_schedule_strictly_increases_with_time(
    noise_schedule_curve: Float[torch.Tensor, "n_t"],
) -> None:
    """Noise levels strictly increase from t = 0 to t = 1.

    Arrange: ``noise_schedule_curve`` evaluates the scenario's schedule on
    an even grid from t = 0 to t = 1.
    Act: compare each grid value with the one after it.
    Assert: every later value is larger. Together with the endpoint test,
    this pins the direction of the time axis: counting t down from 1 to 0
    visits noise levels from maximum to minimum, never revisiting one.

    Args:
        noise_schedule_curve: Noise levels on the grid for this scenario.
    """
    assert bool((noise_schedule_curve[1:] > noise_schedule_curve[:-1]).all())


def test_noise_schedule_different_rho_gives_different_intermediate_values(
    zero_denoiser_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> None:
    """Changing rho must alter intermediate noise levels.

    The boundary values (``sigma_max`` at t = 1 and ``sigma_min`` at t = 0)
    must be identical across schedules; only the intermediate levels may
    differ.
    """
    noise = NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    s1 = EDMSampler(
        zero_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(rho=7.0),
        noise,
    )
    s2 = EDMSampler(
        zero_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(rho=3.0),
        noise,
    )
    ts = torch.linspace(0.1, 0.9, 10)
    vals1 = torch.stack([s1.noise_schedule(t) for t in ts])
    vals2 = torch.stack([s2.noise_schedule(t) for t in ts])
    assert not torch.allclose(vals1, vals2)


# ---------------------------------------------------------------------------
# EDMSampler.sample — mathematical properties of special denoisers
# ---------------------------------------------------------------------------


def test_edm_sampler_identity_denoiser_produces_finite_centred_output(
    identity_denoiser_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> None:
    """Identity denoiser produces finite coordinates and sequence logits.

    Verifies two properties of the sampler output in a single forward pass:

    1. **Finiteness** — both the coordinate tensor and the sequence logits
       contain no NaN or Inf values, and the coordinate tensor has the
       expected shape ``(1, N_ATOM, 3)``.
    2. **Centre-of-mass at origin** — the mean position across all atoms must
       be zero for every batch element (up to ``atol=1e-5``), confirming that
       the sampler zero-centres its output.
    """
    sampler = EDMSampler(
        identity_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0, ddim_steps=4),
        NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX),
    )
    _ = manual_seed(7)
    z, seq = sampler.sample((1, N_ATOM, 3))
    assert z.shape == (1, N_ATOM, 3)
    assert torch.isfinite(z).all()
    assert torch.isfinite(seq).all()
    com: Float[torch.Tensor, "1 3"] = reduce(z, "b n_atom d -> b d", "mean")
    assert torch.allclose(com, torch.zeros_like(com), atol=1e-5)


def test_edm_sampler_zero_denoiser_output_is_zero(
    zero_denoiser_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> None:
    """With a zero denoiser the trajectory drives all coordinates to zero.

    ``D_θ(z, sigma) = 0`` implies ``d = z / sigma``, so each step scales
    ``z`` by ``sigma_next / sigma_hat``. At the final step ``sigma_next = 0``,
    so the trajectory converges exactly to zero.
    """
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


def test_edm_sampler_deterministic_without_s_churn(
    edm_sampler: EDMSampler,
) -> None:
    """With S_churn=0, the sampler must be fully deterministic.

    Given the same random seed, two consecutive calls to ``sample`` must
    return bitwise-identical output tensors for both coordinates and sequence
    logits.
    """
    _ = manual_seed(3)
    out1, seq_out1 = edm_sampler.sample((1, N_ATOM, 3))
    _ = manual_seed(3)
    out2, seq_out2 = edm_sampler.sample((1, N_ATOM, 3))
    assert torch.equal(out1, out2)
    assert torch.equal(seq_out1, seq_out2)


def test_edm_sampler_s_churn_produces_different_result_than_deterministic(
    identity_denoiser_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> None:
    """S_churn > 0 produces output different from the deterministic ODE run.

    ``S_churn > 0`` injects extra noise per step before the predictor. With the
    identity denoiser the injected noise accumulates unfiltered, so the output
    diverges from the ODE run. The zero denoiser is unsuitable here because it
    drives coordinates to zero regardless of ``S_churn``.
    """
    noise = NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    _ = manual_seed(5)
    det = EDMSampler(
        identity_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0),
        noise,
    )
    stoch = EDMSampler(
        identity_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=2.0),
        noise,
    )
    out_det, seq_det = det.sample((1, N_ATOM, 3))
    out_stoch, seq_stoch = stoch.sample((1, N_ATOM, 3))
    assert not torch.allclose(out_det, out_stoch)
    assert torch.isfinite(seq_det).all()
    assert torch.isfinite(seq_stoch).all()


class SamplerTrajectoryScenario(Enum):
    """Sampler settings whose recorded trajectory must follow the schedule.

    Each member is a ``(ddim_steps, jitter, eta_step_scale)`` tuple read
    through properties. ``jitter`` replaces the ``torch.rand()`` draw that
    offsets every step's time, so the visited noise levels are known
    exactly. Members are read via their properties rather than branched on.

    Members:
        two_steps: The smallest step count ``SamplerParams`` allows; the
            first step is also the second-to-last.
        ten_steps: A coarse grid with no jitter, so every step lands on a
            grid point ``t = k / 10``.
        ten_steps_half_jitter: The same grid with every time shifted down by
            half a step, as the sampler's random offset would.
        ten_steps_gentle_step_scale: A smaller Euler step scale, so the
            per-step rescale differs from the default.
        pallatom_default: The production defaults (200 steps, step scale
            2.25) with half-step jitter.

    Every member runs with churn off (``S_churn = 0``), so both model calls
    in a step see the same noise level and the same noisy input.
    """

    two_steps = (2, 0.0, 2.25)
    ten_steps = (10, 0.0, 2.25)
    ten_steps_half_jitter = (10, 0.5, 2.25)
    ten_steps_gentle_step_scale = (10, 0.5, 1.5)
    pallatom_default = (200, 0.5, 2.25)

    @property
    def ddim_steps(self) -> int:
        """Number of sampling steps."""
        return self.value[0]

    @property
    def jitter(self) -> float:
        """Value returned in place of each ``torch.rand()`` draw."""
        return self.value[1]

    @property
    def eta_step_scale(self) -> float:
        """Multiplier on each Euler step."""
        return self.value[2]

    @property
    def expected_model_calls(self) -> int:
        """Model calls a full run must make: two per step."""
        return 2 * self.ddim_steps


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class SamplerTrajectory:
    """What the model saw during one sampling run, with exact expectations.

    Built by the ``sampler_trajectory`` factory fixture. "Centred RMS" is
    the RMS of a step's noisy input about its centroid, which measures how
    much noise the input carries. Recorded and expected sequences use
    separate dimension names on purpose: a sampler that runs the wrong
    number of steps must reach the test and fail its comparison there,
    not fail jaxtyping while the fixture is being built.

    Attributes:
        model_calls: Number of model calls the run made.
        step_sigmas: Noise level passed to the model at each step, in loop
            order.
        expected_step_sigmas: ``noise_schedule(k / T - jitter / T)`` for
            ``k = T, ..., 1``.
        initial_centred_rms: Centred RMS of the first step's input.
        expected_initial_centred_rms: Centred RMS of the seeded initial
            draw, scaled by the noise level the first model call was told.
            Taken from the recorded call rather than from
            ``noise_schedule``, so a schedule evaluated at the wrong end
            cannot make the expectation agree with the bug.
        step_rms_ratios: Centred RMS of each step's input divided by the
            previous step's.
        expected_step_rms_ratios: ``|1 + eta * (sigma_next / sigma - 1)|``
            for each step but the last, where ``sigma_next`` is the
            schedule one step further toward t = 0.
    """

    model_calls: int
    step_sigmas: Float[torch.Tensor, "n_recorded_steps"]
    expected_step_sigmas: Float[torch.Tensor, "n_expected_steps"]
    initial_centred_rms: Float[torch.Tensor, ""]
    expected_initial_centred_rms: Float[torch.Tensor, ""]
    step_rms_ratios: Float[torch.Tensor, "n_recorded_gaps"]
    expected_step_rms_ratios: Float[torch.Tensor, "n_expected_gaps"]


@pytest.fixture
def sampler_trajectory(
    request: pytest.FixtureRequest,
    zero_denoiser_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
    monkeypatch: pytest.MonkeyPatch,
) -> SamplerTrajectory:
    """Run the sampler for one scenario and record every model input.

    Factory fixture: receives a ``SamplerTrajectoryScenario`` through
    ``@pytest.mark.parametrize(..., indirect=["sampler_trajectory"])``.

    Four choices make every expectation exact rather than statistical:

    1. ``torch.rand`` is monkeypatched to return the scenario's jitter.
       The sampler draws ``torch.rand()`` only for the per-step time offset,
       so the visited times are exactly ``k / T - jitter / T``. Following
       Pallatom, the real sampler draws a separate offset for the initial
       noise and for every step, so neighbouring steps can disagree by up to
       one interval. Pinning every draw to one value makes them agree, so
       these tests check the schedule direction and step targets, not that
       per-step independence.
    2. The zero denoiser stands in for the trunk. Its output is always
       zero, so each Euler step only rescales the input by
       ``1 + eta * (sigma_next - sigma) / sigma``. No real model's output
       could be predicted this way.
    3. The global seed is set to 0 twice: once to draw the initial noise
       here, and again just before ``sample()``. With ``torch.rand``
       patched, the sampler's first random draw is that same initial
       ``torch.randn``, so the fixture knows the starting coordinates.
    4. ``centre_random_augment`` is replaced with plain centring. Its random
       rotation and translation do not affect which noise levels are
       visited, but the fresh ~1 Å translation it adds every step would
       swamp the input in float32: with a zero denoiser and 200 steps the
       input shrinks to about 1e-10 Å, far below float32 resolution at
       1 Å.

    The last step's Euler update is discarded (``sample()`` returns the
    denoised prediction), so ratios cover every step but the last.

    Args:
        request: Pytest request carrying the ``SamplerTrajectoryScenario``.
        zero_denoiser_mock: Trunk stand-in that always predicts zeros and
            records every call.
        context: Unconditional sampling context.
        templ_disto: Template distogram module for self-conditioning.
        monkeypatch: Pytest monkeypatch used to pin ``torch.rand`` and to
            replace ``centre_random_augment``.

    Returns:
        SamplerTrajectory with the recorded trajectory and its expectations.
    """
    scenario = cast(SamplerTrajectoryScenario, request.param)
    seed = 0
    shape = (1, context.r_gt.shape[1], 3)
    delta_t = 1 / scenario.ddim_steps
    sampler = EDMSampler(
        zero_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(
            S_churn=0.0,
            ddim_steps=scenario.ddim_steps,
            eta_step_scale=scenario.eta_step_scale,
        ),
        NoiseScheduleParams(),
    )

    def fixed_jitter(
        size: tuple[int, ...],
        device: torch.device,
    ) -> Float[torch.Tensor, "..."]:
        """Stand in for ``torch.rand``, returning the scenario's jitter.

        Args:
            size: Requested output shape.
            device: Requested output device.

        Returns:
            Tensor of ``size`` filled with the scenario's jitter.
        """
        return torch.full(size, scenario.jitter, device=device)

    def centre_only(
        coords: Float[torch.Tensor, "B N_atom 3"],
        mask: Bool[torch.Tensor, "B N_atom"],
    ) -> Float[torch.Tensor, "B N_atom 3"]:
        """Stand in for ``centre_random_augment``: centre with no rotation.

        Args:
            coords: Batched atom coordinates.
            mask: Valid-atom mask used for the centroid.

        Returns:
            ``coords`` shifted so their masked centroid is at the origin.
        """
        return coords - masked_com(coords, mask=mask)

    def centred_rms(
        coords: Float[torch.Tensor, "B N_atom 3"],
    ) -> Float[torch.Tensor, ""]:
        """RMS of ``coords`` about their centroid.

        Args:
            coords: Batched atom coordinates.

        Returns:
            Scalar RMS distance from the centroid.
        """
        centred = coords - reduce(coords, "b n d -> b 1 d", "mean")
        return torch.sqrt(reduce(centred**2, "b n d -> ", "mean"))

    monkeypatch.setattr(torch, "rand", fixed_jitter)
    monkeypatch.setattr("sample.sampling.centre_random_augment", centre_only)
    _ = manual_seed(seed)
    initial_draw = torch.randn(shape)
    _ = manual_seed(seed)
    _ = sampler.sample(shape)

    step_batches = [
        cast(FeaturizedBatch, call.args[0])
        for call in zero_denoiser_mock.call_args_list
    ][::2]
    step_sigmas = torch.stack([batch.t_hat[0] for batch in step_batches])
    step_rms = torch.stack(
        [centred_rms(batch.r_gt_noised) for batch in step_batches],
    )
    times = (
        torch.arange(scenario.ddim_steps, 0, -1) * delta_t
        - scenario.jitter * delta_t
    )
    expected_sigmas = torch.stack([sampler.noise_schedule(t) for t in times])
    next_sigmas = torch.stack(
        [sampler.noise_schedule((t - delta_t).clamp(min=0.0)) for t in times],
    )
    return SamplerTrajectory(
        model_calls=zero_denoiser_mock.call_count,
        step_sigmas=step_sigmas,
        expected_step_sigmas=expected_sigmas,
        initial_centred_rms=step_rms[0],
        expected_initial_centred_rms=centred_rms(
            step_sigmas[0] * centre_only(initial_draw, context.atom5_mask),
        ),
        step_rms_ratios=step_rms[1:] / step_rms[:-1],
        expected_step_rms_ratios=torch.abs(
            1 + scenario.eta_step_scale * (next_sigmas / expected_sigmas - 1),
        )[:-1],
    )


@pytest.mark.parametrize(
    ("sampler_trajectory", "scenario"),
    [pytest.param(s, s, id=s.name) for s in SamplerTrajectoryScenario],
    indirect=["sampler_trajectory"],
)
def test_edm_sampler_calls_model_twice_per_step(
    sampler_trajectory: SamplerTrajectory,
    scenario: SamplerTrajectoryScenario,
) -> None:
    """sample() runs every one of its ddim_steps, with two model calls each.

    Each step calls the model once to build the self-conditioning template
    and once more with that template to take the Euler step. The loop
    previously ran ``ddim_steps - 2`` steps and never reached the lowest
    noise levels.

    Arrange: ``sampler_trajectory`` runs the sampler for the scenario.
    Act: count the recorded model calls.
    Assert: the count is ``2 * ddim_steps``.

    Args:
        sampler_trajectory: Recorded run for this scenario.
        scenario: Step count and expected number of model calls.
    """
    assert sampler_trajectory.model_calls == scenario.expected_model_calls


@pytest.mark.parametrize(
    "sampler_trajectory",
    [pytest.param(s, id=s.name) for s in SamplerTrajectoryScenario],
    indirect=True,
)
def test_edm_sampler_visits_noise_levels_from_max_to_min(
    sampler_trajectory: SamplerTrajectory,
) -> None:
    """The model sees noise_schedule(k / T - jitter / T) for k = T down to 1.

    Arrange: ``sampler_trajectory`` pins the per-step jitter and records
    the noise level passed to the model at each step.
    Act: compare the recorded noise levels with the schedule evaluated at
    the expected times.
    Assert: they match exactly (to float32 precision), so the loop counts
    down from maximum noise at t near 1 to minimum noise at t near 0.

    Args:
        sampler_trajectory: Recorded run and expected noise levels.
    """
    torch.testing.assert_close(
        sampler_trajectory.step_sigmas,
        sampler_trajectory.expected_step_sigmas,
    )


@pytest.mark.parametrize(
    "sampler_trajectory",
    [pytest.param(s, id=s.name) for s in SamplerTrajectoryScenario],
    indirect=True,
)
def test_edm_sampler_initial_noise_matches_first_step(
    sampler_trajectory: SamplerTrajectory,
) -> None:
    """The starting coordinates are drawn at the first step's noise level.

    With the per-step offset pinned (see ``sampler_trajectory``), the initial
    noise and the first step share one offset, so this holds exactly. In
    production they draw separate offsets, as in Pallatom, and share only
    the top schedule interval.

    Regression: the initial noise level was evaluated at the wrong end of
    the schedule, so the first model call saw coordinates of about
    0.007 Å labelled as about 2,500 Å of noise.

    Arrange: ``sampler_trajectory`` seeds the initial draw and records the
    first model call's input and noise level.
    Act: compare that input's centred RMS with the seeded draw's centred
    RMS scaled by the noise level the first call was told.
    Assert: they match exactly (to float32 precision), so the input really
    carries the amount of noise the model is told it has.

    Args:
        sampler_trajectory: Recorded run and expected initial spread.
    """
    torch.testing.assert_close(
        sampler_trajectory.initial_centred_rms,
        sampler_trajectory.expected_initial_centred_rms,
    )


@pytest.mark.parametrize(
    "sampler_trajectory",
    [pytest.param(s, id=s.name) for s in SamplerTrajectoryScenario],
    indirect=True,
)
def test_edm_sampler_euler_steps_target_next_lower_noise_level(
    sampler_trajectory: SamplerTrajectory,
) -> None:
    """Each Euler step targets the next, lower noise level in the schedule.

    Regression: the Euler target was the schedule one step toward higher
    noise, so steps removed too little noise and late in sampling the
    residual noise drifted above the schedule.

    Arrange: ``sampler_trajectory`` runs with a zero denoiser and churn
    off, so each step only rescales the input by
    ``1 + eta * (sigma_next - sigma) / sigma``.
    Act: divide each step's input centred RMS by the previous step's.
    Assert: each ratio equals the magnitude of that rescale, with
    ``sigma_next`` taken one step further toward t = 0.

    Args:
        sampler_trajectory: Recorded run and expected per-step rescales.
    """
    torch.testing.assert_close(
        sampler_trajectory.step_rms_ratios,
        sampler_trajectory.expected_step_rms_ratios,
    )


# ---------------------------------------------------------------------------
# EDMSampler.sample — step count affects trajectory
# ---------------------------------------------------------------------------


def test_edm_sampler_step_count_changes_output_for_nontrivial_denoiser(
    half_denoiser_mock: MagicMock,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> None:
    """Coarser sigma grid integrates differently and yields different output.

    D_θ(z, sigma) = 0.5·z, so the trajectory depends entirely on the sigma
    grid. A coarser grid (fewer steps) integrates differently and must produce
    a distinct final output.
    """
    noise = NoiseScheduleParams(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX)
    _ = manual_seed(9)
    coarse = EDMSampler(
        half_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0, ddim_steps=4),
        noise,
    )
    fine = EDMSampler(
        half_denoiser_mock,
        context,
        templ_disto,
        SamplerParams(S_churn=0.0, ddim_steps=10),
        noise,
    )
    out_coarse, seq_coarse = coarse.sample((1, N_ATOM, 3))
    out_fine, seq_fine = fine.sample((1, N_ATOM, 3))
    assert not torch.allclose(out_coarse, out_fine)
    assert torch.isfinite(seq_coarse).all()
    assert torch.isfinite(seq_fine).all()


# ---------------------------------------------------------------------------
# build_sampling_context — additional fields
# ---------------------------------------------------------------------------


def test_build_sampling_context_additional_fields(
    context: FeaturizedBatch,
) -> None:
    """Verify additional fields of an unconditional sampling context.

    Checks that ``aa_indices`` are all alanine (0), that ``r_gt`` is all
    zeros, that the sparse atom distogram is finite with its mask fully set
    and the expected atom/bin dimensions, and that ``ref_pos``, ``tok_idx``,
    and ``r_gt`` are all placed on CPU.
    """
    assert (context.aa_indices == 0).all()
    assert (context.r_gt == 0).all()
    assert torch.isfinite(context.gt_atom_distogram_sparse.float()).all()
    assert context.gt_atom_distogram_mask_sparse.all()
    assert context.gt_atom_distogram_sparse.shape[1] == N_ATOM
    assert int(context.gt_atom_distogram_sparse.max().item()) < N_ATOM_BINS
    assert context.ref_pos.device.type == CPU_DEVICE_TYPE
    assert context.tok_idx.device.type == CPU_DEVICE_TYPE
    assert context.r_gt.device.type == CPU_DEVICE_TYPE


# ---------------------------------------------------------------------------
# EDMSampler.sample — edge cases and S_churn windowing
# ---------------------------------------------------------------------------


def test_edm_sampler_sample_runs_without_error(edm_sampler: EDMSampler) -> None:
    """sample() returns finite coordinate and seq tensors."""
    out, seq_out = edm_sampler.sample((1, N_ATOM, 3))
    assert out.shape == (1, N_ATOM, 3)
    assert torch.isfinite(out).all()
    assert torch.isfinite(seq_out).all()


def test_edm_sampler_s_tmin_above_one_disables_injection(
    identity_det_sampler: EDMSampler,
    identity_stoch_sampler_tmin_high: EDMSampler,
) -> None:
    """S_tmin above 1 disables stochastic noise injection entirely.

    The churn gate compares ``S_tmin`` with the schedule time
    ``timestep / ddim_steps``, which never exceeds 1. When ``S_tmin > 1``,
    the condition ``S_tmin ≤ timestep / ddim_steps`` is never satisfied at
    any step, so the stochastic sampler degenerates to the deterministic
    (``S_churn=0``) trajectory. Both samplers are seeded identically and
    their outputs are asserted to be numerically equal.
    """
    _ = manual_seed(5)
    out_det, seq_det = identity_det_sampler.sample((1, N_ATOM, 3))
    _ = manual_seed(5)
    out_stoch, seq_stoch = identity_stoch_sampler_tmin_high.sample(
        (1, N_ATOM, 3),
    )
    assert torch.allclose(out_det, out_stoch)
    assert torch.equal(seq_det, seq_stoch)


# ---------------------------------------------------------------------------
# build_aa_context — constants
# ---------------------------------------------------------------------------

N_RES_AA = 6
N_ATOM_AA = N_RES_AA * NATOM  # 30
AA_SEQ_AA = "ACDEFG"  # len == N_RES_AA


# ---------------------------------------------------------------------------
# build_aa_context — fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def atom37_pos() -> Float[torch.Tensor, "N_RES_AA 37 3"]:
    """Random atom37 coordinates for AA context tests."""
    return torch.randn(N_RES_AA, 37, 3)


@pytest.fixture
def atom37_mask_all() -> Float[torch.Tensor, "N_RES_AA 37"]:
    """Provide an all-ones atom37 mask indicating every atom is present."""
    return torch.ones(N_RES_AA, 37)


@pytest.fixture
def residue_idx_aa() -> Float[torch.Tensor, "N_RES_AA"]:
    """Sequential residue index [0, ..., N_RES_AA-1] for AA context tests."""
    return torch.arange(N_RES_AA, dtype=torch.float)


@pytest.fixture
def atom_disto_fn() -> Distogram:
    """Atom distogram function with 22 bins spanning 2-22 Å."""
    return Distogram(
        n_bins=N_ATOM_BINS,
        min_dist=2.0,
        max_dist=22.0,
        overflow_bin=False,
    )


@pytest.fixture
def aa_ctx(
    atom37_pos: Float[torch.Tensor, "N_RES_AA 37 3"],
    atom37_mask_all: Float[torch.Tensor, "N_RES_AA 37"],
    residue_idx_aa: Float[torch.Tensor, "N_RES_AA"],
    atom_disto_fn: Distogram,
) -> AllAtomContext:
    """AllAtomContext for 6-residue diverse sequence with batch_size=2."""
    prot = Protein(
        atom_positions=np.array(atom37_pos, dtype=np.float64),
        atom_mask=np.array(atom37_mask_all, dtype=np.float64),
        residue_index=np.array(residue_idx_aa, dtype=np.intp),
        aatype=np.array(
            [restype_order[c] for c in AA_SEQ_AA],
            dtype=np.intp,
        ),
        chain_index=np.zeros(N_RES_AA, dtype=np.intp),
        b_factors=np.zeros((N_RES_AA, 37), dtype=np.float64),
    )
    with torch.no_grad():
        return build_aa_context(prot, atom_disto_fn, batch_size=2, device="cpu")


# ---------------------------------------------------------------------------
# build_aa_context — return type, shapes, and values
# ---------------------------------------------------------------------------


class ExtraChecks(Flag):
    """Optional per-field assertions beyond shape and finiteness."""

    NONE = 0
    ALL_SET = auto()
    BATCH_EQUAL = auto()
    MATCH_SEQUENCE = auto()


@pytest.mark.parametrize(
    ("attr", "expected", "checks"),
    [
        ("r_gt", (2, N_ATOM_AA, 3), ExtraChecks.BATCH_EQUAL),
        ("atom5_mask", (2, N_ATOM_AA), ExtraChecks.ALL_SET),
        ("residue_mask", (2, N_RES_AA), ExtraChecks.ALL_SET),
        (
            "aa_indices",
            (2, N_RES_AA),
            ExtraChecks.BATCH_EQUAL | ExtraChecks.MATCH_SEQUENCE,
        ),
        ("f_residue_idx", (2, N_RES_AA), ExtraChecks.NONE),
        # K (sparse neighbour count) is variable; None skips that axis.
        (
            "gt_atom_distogram_sparse",
            (2, N_ATOM_AA, None),
            ExtraChecks.NONE,
        ),
        (
            "gt_atom_distogram_mask_sparse",
            (2, N_ATOM_AA, None),
            ExtraChecks.NONE,
        ),
    ],
)
def test_build_aa_context_field_shape(
    aa_ctx: AllAtomContext,
    attr: str,
    expected: tuple[int | None, ...],
    checks: ExtraChecks,
) -> None:
    """Shape, finiteness, and per-field invariant checks for AllAtomContext.

    Each row in the parametrize table is ``(attr, expected, checks)`` where
    ``attr`` is an ``AllAtomContext`` field name, ``expected`` is a shape
    tuple, and ``checks`` is an ``ExtraChecks`` flag that gates optional
    assertions beyond shape and finiteness.

    Shape comparison: ``None`` in ``expected`` acts as a wildcard that skips
    that axis.  This is used for the distogram fields whose third dimension
    ``K`` (sparse neighbour count) is not a module-level constant.

    ``ExtraChecks`` flags:

    - ``ALL_SET``: asserts ``tensor.all()`` — used for the boolean masks
      ``atom5_mask`` and ``residue_mask``, which must be fully ones when
      every atom / residue is present.
    - ``BATCH_EQUAL``: asserts ``tensor[0] == tensor[1]`` — used for
      ``r_gt`` and ``aa_indices``, which are tiled from the same protein so
      every batch item must be identical.
    - ``MATCH_SEQUENCE``: asserts ``tensor[0]`` equals the expected
      amino-acid token indices derived from ``AA_SEQ_AA`` — used for
      ``aa_indices`` to confirm the tokenisation round-trips correctly.

    Args:
        aa_ctx: Fully-populated ``AllAtomContext`` fixture (batch size 2,
            ``N_ATOM_AA`` atoms, ``N_RES_AA`` residues).
        attr: Name of the ``AllAtomContext`` field under test.
        expected: Expected shape tuple; ``None`` entries are wildcards.
        checks: Bit-flag of optional assertions to run for this field.
    """
    tensor = cast(torch.Tensor, getattr(aa_ctx, attr))
    assert all(
        e is None or s == e for s, e in zip(tensor.shape, expected, strict=True)
    )
    assert torch.isfinite(tensor.float()).all()
    if ExtraChecks.ALL_SET in checks:
        assert tensor.all()
    if ExtraChecks.BATCH_EQUAL in checks:
        assert torch.equal(tensor[0], tensor[1])
    if ExtraChecks.MATCH_SEQUENCE in checks:
        assert torch.equal(
            tensor[0],
            torch.tensor(
                [restype_order[c] for c in AA_SEQ_AA],
                dtype=torch.long,
            ),
        )


def test_build_aa_context_batch_size_controls_leading_dim(
    atom37_pos: Float[torch.Tensor, "N_RES_AA 37 3"],
    atom37_mask_all: Float[torch.Tensor, "N_RES_AA 37"],
    residue_idx_aa: Float[torch.Tensor, "N_RES_AA"],
    atom_disto_fn: Distogram,
) -> None:
    """Batch size controls the leading dimension of r_gt and aa_indices.

    Builds an ``AllAtomContext`` with ``batch_size=B`` (``B=2``) and asserts
    that both ``r_gt.shape[0]`` and ``aa_indices.shape[0]`` equal ``B``,
    confirming that the batching axis is set correctly rather than defaulting
    to 1.
    """
    prot = Protein(
        atom_positions=np.array(atom37_pos, dtype=np.float64),
        atom_mask=np.array(atom37_mask_all, dtype=np.float64),
        residue_index=np.array(residue_idx_aa, dtype=np.intp),
        aatype=np.array(
            [restype_order[c] for c in AA_SEQ_AA],
            dtype=np.intp,
        ),
        chain_index=np.zeros(N_RES_AA, dtype=np.intp),
        b_factors=np.zeros((N_RES_AA, 37), dtype=np.float64),
    )
    with torch.no_grad():
        ctx = build_aa_context(prot, atom_disto_fn, batch_size=B, device="cpu")
    assert ctx.r_gt.shape[0] == B
    assert ctx.aa_indices.shape[0] == B


# ---------------------------------------------------------------------------
# build_template_context — constants and helpers
# ---------------------------------------------------------------------------

N_TEMPL_BINS = 39  # 38 distance bins + 1 overflow bin


def _make_protein(n_res: int, mask_value: float = 1.0) -> Protein:
    """Build minimal Protein with random coordinates and uniform atom mask."""
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
    """Full-length Protein with random coordinates and all-present mask."""
    return _make_protein(N_RES)


@pytest.fixture
def np_protein_short() -> Protein:
    """Shorter protein fixture for padding in batched template contexts."""
    return _make_protein(N_RES - 2)  # shorter, for padding test


@pytest.fixture
def templ_disto() -> Distogram:
    """Template distogram: 38 bins + overflow, spanning 3.25-50.75 Å."""
    return Distogram(
        n_bins=N_TEMPL_BINS - 1,
        min_dist=3.25,
        max_dist=50.75,
        overflow_bin=True,
    )


@pytest.fixture
def template_ctx(
    np_protein: Protein,
    templ_disto: Distogram,
) -> TemplateContext:
    """TemplateContext built from np_protein and template distogram."""
    return build_template_context(
        np_protein,
        batch_size=1,
        distogram_fn=templ_disto,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# build_template_context — return type, shapes, dtypes, and distogram properties
# ---------------------------------------------------------------------------


def test_build_template_context(template_ctx: TemplateContext) -> None:
    """Verify type, shape, dtype, and distogram properties of TemplateContext.

    Checks that the returned object is a ``TemplateContext``, that
    ``f_template_distogram`` has shape ``(1, N_RES, N_RES, N_TEMPL_BINS)`` and
    ``f_pseudo_beta_mask`` has shape ``(1, N_RES)``, both with dtype
    ``torch.long``.  Also confirms that the distogram bins sum to one per
    residue pair (a valid probability simplex), that the matrix is symmetric
    under index transposition, and that the pseudo-β mask is fully set.
    """
    assert isinstance(template_ctx, TemplateContext)

    assert template_ctx.f_template_distogram.shape == (
        1,
        N_RES,
        N_RES,
        N_TEMPL_BINS,
    )
    assert template_ctx.f_pseudo_beta_mask.shape == (1, N_RES)

    assert template_ctx.f_template_distogram.dtype == torch.float32
    assert template_ctx.f_pseudo_beta_mask.dtype == torch.long

    bin_sums = reduce(
        template_ctx.f_template_distogram.float(),
        "b i j bins -> b i j",
        "sum",
    )
    assert torch.allclose(bin_sums, torch.ones_like(bin_sums))
    disto = template_ctx.f_template_distogram
    assert torch.equal(disto, rearrange(disto, "b i j k -> b j i k"))
    assert (template_ctx.f_pseudo_beta_mask == 1).all()


def test_build_template_context_different_proteins_give_different_distograms(
    templ_disto: Distogram,
) -> None:
    """Verify that two distinct proteins produce different template distograms.

    Constructs a second protein (``prot_b``) from high-variance random
    coordinates (scale 20 Å) alongside the default ``prot_a``, runs both
    through ``build_template_context``, and asserts the resulting
    ``f_template_distogram`` tensors are not identical — confirming that the
    distogram is sensitive to input coordinates rather than being constant.
    """
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
    ctx_a = build_template_context(
        prot_a,
        batch_size=1,
        distogram_fn=templ_disto,
        device="cpu",
    )
    ctx_b = build_template_context(
        prot_b,
        batch_size=1,
        distogram_fn=templ_disto,
        device="cpu",
    )
    assert not torch.equal(
        ctx_a.f_template_distogram,
        ctx_b.f_template_distogram,
    )


# ===========================================================================
# Use-case tests for build_sampling_context
# ===========================================================================


@pytest.fixture
def unconditional_ctx(
    atom_disto_fn: Distogram,
    templ_disto: Distogram,
) -> FeaturizedBatch:
    """No-template context: all-alanine, zero positions, residue index."""
    with torch.no_grad():
        return build_sampling_context(
            pdb_file_path=None,
            atom_distogram_fn=atom_disto_fn,
            templ_distogram_fn=templ_disto,
            residue_number=N_RES,
            batch_size=1,
            device="cpu",
        )


# ---------------------------------------------------------------------------
# No-template (all-alanine) context
# ---------------------------------------------------------------------------


def test_unconditional(unconditional_ctx: FeaturizedBatch) -> None:
    """Verify unconditional (all-alanine, zero-coordinate) context.

    Checks that the distogram peaks at bin 0, the pseudo-β mask is fully set,
    ``aa_indices`` are all alanine (0), every atom5 entry is unmasked, and the
    ground-truth coordinates are zero.
    """
    assert (unconditional_ctx.noised_res_distogram.argmax(dim=-1) == 0).all()
    assert (unconditional_ctx.f_pseudo_beta_mask == 1).all()
    assert (unconditional_ctx.aa_indices == 0).all()
    assert unconditional_ctx.atom5_mask.all()
    assert (unconditional_ctx.r_gt == 0).all()


# ---------------------------------------------------------------------------
# build_aa_context — 'X' residue handling
# ---------------------------------------------------------------------------


def test_build_aa_context_x_residue_handling(
    atom_disto_fn: Distogram,
) -> None:
    """Verify aatype=RESTYPE_NUM_NO_X is preserved in aa_indices.

    Builds two proteins — one with every residue set to ``RESTYPE_NUM_NO_X``
    and one with every residue set to alanine (index 0) — then checks that
    ``build_aa_context`` round-trips the unknown-residue token into
    ``aa_indices`` unchanged and that the two contexts differ.
    """
    prot_x = Protein(
        atom_positions=np.zeros((N_RES_AA, 37, 3), dtype=np.float64),
        atom_mask=np.zeros((N_RES_AA, 37), dtype=np.float64),
        residue_index=np.arange(N_RES_AA, dtype=np.intp),
        aatype=np.full(N_RES_AA, RESTYPE_NUM_NO_X, dtype=np.intp),
        chain_index=np.zeros(N_RES_AA, dtype=np.intp),
        b_factors=np.zeros((N_RES_AA, 37), dtype=np.float64),
    )
    prot_a = Protein(
        atom_positions=np.zeros((N_RES_AA, 37, 3), dtype=np.float64),
        atom_mask=np.zeros((N_RES_AA, 37), dtype=np.float64),
        residue_index=np.arange(N_RES_AA, dtype=np.intp),
        aatype=np.zeros(N_RES_AA, dtype=np.intp),
        chain_index=np.zeros(N_RES_AA, dtype=np.intp),
        b_factors=np.zeros((N_RES_AA, 37), dtype=np.float64),
    )
    with torch.no_grad():
        ctx_x = build_aa_context(
            prot_x,
            atom_disto_fn,
            batch_size=1,
            device="cpu",
        )
        ctx_a = build_aa_context(
            prot_a,
            atom_disto_fn,
            batch_size=1,
            device="cpu",
        )
    assert isinstance(ctx_x, AllAtomContext)
    assert (ctx_x.aa_indices == RESTYPE_NUM_NO_X).all()
    assert not torch.equal(ctx_x.aa_indices, ctx_a.aa_indices)


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_all_atom_context_wrong_shape() -> None:
    """Wrong r_gt ndim (2-D instead of 3-D) triggers TypeCheckError."""
    b_local, n_atom_local, n_res_local, k_local = 2, N_ATOM, N_RES, 4
    with pytest.raises(TypeCheckError):
        _ = AllAtomContext(
            r_gt=torch.zeros(b_local, n_atom_local),  # must be 3-D "B N_atom 3"
            atom5_mask=torch.zeros(b_local, n_atom_local, dtype=torch.bool),
            residue_mask=torch.zeros(b_local, n_res_local, dtype=torch.bool),
            gt_atom_distogram_sparse=torch.zeros(
                b_local,
                n_atom_local,
                k_local,
                N_ATOM_BINS,
            ),
            gt_atom_distogram_mask_sparse=torch.zeros(
                b_local,
                n_atom_local,
                k_local,
                dtype=torch.bool,
            ),
            aa_indices=torch.zeros(b_local, n_res_local, dtype=torch.long),
            f_residue_idx=torch.zeros(b_local, n_res_local, dtype=torch.long),
        )


def test_template_context_wrong_shape() -> None:
    """Wrong f_template_distogram ndim triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        _ = TemplateContext(
            f_template_distogram=torch.zeros(
                1,
                N_RES,
                N_RES,
                dtype=torch.long,
            ),  # missing bins dim
            f_pseudo_beta_mask=torch.zeros(1, N_RES, dtype=torch.long),
        )

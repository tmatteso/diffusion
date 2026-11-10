"""Tests for the conditional sampling loop.

Covers the unconditional sampling context, AllAtomContext and
TemplateContext from build_aa_context and build_template_context, the
EDM noise schedule's direction and endpoints, and EDMSampler: the exact
noise levels and Euler rescales it visits, its model-call count, the effect
of churn and of S_tmin, and the outputs of zero, identity and half
denoisers.

The trunk is replaced by ``ScaledDenoiser``, a small recording stand-in
that returns ``scale * r_gt_noised``. Its output is known exactly, which is
what lets the trajectory tests compute every expected value; a real trunk's
output could not be predicted.
"""

import dataclasses
import enum
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
import torch
from architecture.main_trunk import MainTrunk, PredictedOutputs
from einops import rearrange, reduce, repeat
from helpers.alignment import masked_com
from helpers.atom_utils import RESTYPE_NUM_NO_X, Protein, restype_order
from helpers.data import Distogram, FeaturizedBatch
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Shaped, TypeCheckError
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


@dataclasses.dataclass
class ScaledDenoiser:
    """Trunk stand-in that returns ``scale * r_gt_noised`` and records calls.

    ``scale = 0`` predicts zeros, ``1`` echoes its input, ``0.5`` halves it;
    sequence and distogram logits are zero.

    Attributes:
        scale: Factor applied to the noisy coordinates.
        calls: Every batch the sampler passed in, in order.
    """

    scale: float
    calls: list[FeaturizedBatch] = dataclasses.field(default_factory=list)

    def __call__(self, batch: FeaturizedBatch) -> PredictedOutputs:
        """Record ``batch`` and return the scaled prediction.

        Args:
            batch: The sampler's model input.

        Returns:
            Outputs shaped as MainTrunk's.
        """
        self.calls.append(batch)
        b, n_atom = batch.r_gt.shape[:2]
        n_res = batch.aa_indices.shape[1]
        return PredictedOutputs(
            r_denoised=self.scale * batch.r_gt_noised,
            seq_logits=torch.zeros(b, n_res, RESTYPE_NUM_NO_X),
            residue_distogram_logits=torch.zeros(b, n_res, n_res, 38),
            atom_distogram_logits=torch.zeros(b, n_atom, 1, 38),
            intermediate_denoised_coord_stack=[],
            intermediate_pred_aa_logit_stack=[],
        )


@dataclasses.dataclass(frozen=True)
class FixedRand:
    """Stand-in for ``torch.rand`` that returns one fixed value.

    The sampler draws ``torch.rand()`` only for its per-step time offset,
    so pinning it makes the visited times exact.

    Attributes:
        value: Value every draw returns.
    """

    value: float

    def __call__(
        self,
        size: tuple[int, ...],
        device: torch.device,
    ) -> Float[torch.Tensor, "..."]:
        """Return a tensor of ``size`` filled with ``value``.

        Args:
            size: Requested output shape.
            device: Requested output device.

        Returns:
            The filled tensor.
        """
        return torch.full(size, self.value, device=device)


def centre_only(
    coords: Float[torch.Tensor, "B N_atom 3"],
    mask: Bool[torch.Tensor, "B N_atom"],
) -> Float[torch.Tensor, "B N_atom 3"]:
    """Stand in for ``centre_random_augment``: centre, no rotation.

    Args:
        coords: Batched atom coordinates.
        mask: Valid-atom mask for the centroid.

    Returns:
        ``coords`` with their masked centroid at the origin.
    """
    return coords - masked_com(coords, mask=mask)


def centred_rms(
    coords: Float[torch.Tensor, "B N 3"],
) -> Float[torch.Tensor, ""]:
    """Return the RMS of ``coords`` about their centroid.

    Args:
        coords: Batched atom coordinates.

    Returns:
        Scalar RMS distance from the centroid.
    """
    centred = coords - reduce(coords, "b n d -> b 1 d", "mean")
    return torch.sqrt(reduce(centred**2, "b n d -> ", "mean"))


def build_sampler(
    denoiser: ScaledDenoiser,
    sampling_context: FeaturizedBatch,
    template_distogram: Distogram,
    params: SamplerParams,
    noise: NoiseScheduleParams,
) -> EDMSampler:
    """Build an EDMSampler whose trunk is ``denoiser``.

    Args:
        denoiser: Trunk stand-in.
        sampling_context: Unconditional context.
        template_distogram: Template distogram for self-conditioning.
        params: Sampler settings.
        noise: Noise schedule.

    Returns:
        The sampler.
    """
    return EDMSampler(
        cast(MainTrunk, cast(object, denoiser)),
        sampling_context,
        template_distogram,
        params,
        noise,
    )


def aa_protein(sequence: str) -> Protein:
    """Build a fully present protein with the given sequence.

    Residue ``r``'s atom ``a`` sits at ``(r + 0.1 a, sin(r + a), 0)``.

    Args:
        sequence: One-letter codes; ``X`` is the unknown residue.

    Returns:
        The protein.
    """
    n = len(sequence)
    r = rearrange(np.arange(n, dtype=np.float64), "r -> r 1")
    a = rearrange(np.arange(37, dtype=np.float64), "a -> 1 a")
    positions = np.stack(
        [r + 0.1 * a, np.sin(r + a), np.zeros((n, 37))],
        axis=-1,
    )
    return Protein(
        atom_positions=positions,
        aatype=np.array([restype_order[c] for c in sequence], dtype=np.intp),
        atom_mask=np.ones((n, 37)),
        residue_index=np.arange(n, dtype=np.intp),
        chain_index=np.zeros(n, dtype=np.intp),
        b_factors=np.zeros((n, 37)),
    )


@dataclasses.dataclass(frozen=True)
class ChurnCase:
    """Churn settings and whether the run must match the churn-free one.

    Attributes:
        churn: ``S_churn`` of the stochastic run.
        s_tmin: ``S_tmin`` of the stochastic run.
        matches_ode: Whether it must equal the ``S_churn = 0`` run.
    """

    churn: float
    s_tmin: float
    matches_ode: bool


class ChurnScenario(enum.Enum):
    """Churn windows that do and do not cover any step.

    Members:
        churn_in_window: ``S_tmin = 0.05``, so churn reaches most steps.
        window_above_one: ``S_tmin = 800`` exceeds every
            ``timestep / ddim_steps``, so churn never applies.
    """

    churn_in_window = ChurnCase(2.0, 0.05, matches_ode=False)
    window_above_one = ChurnCase(2.0, 800.0, matches_ode=True)


@dataclasses.dataclass(frozen=True)
class RhoCurves:
    """The noise schedule on 11 even points for two values of ``rho``.

    Attributes:
        rho_7: Noise levels with ``rho = 7``.
        rho_3: Noise levels with ``rho = 3``.
    """

    rho_7: Float[torch.Tensor, " n_t"]
    rho_3: Float[torch.Tensor, " n_t"]


@dataclasses.dataclass(frozen=True)
class HalvingSamplers:
    """Two samplers over ``D(z) = z / 2`` that differ only in step count.

    Attributes:
        four_steps: ``ddim_steps = 4``.
        ten_steps: ``ddim_steps = 10``.
    """

    four_steps: EDMSampler
    ten_steps: EDMSampler


@dataclasses.dataclass(frozen=True)
class ContextCase:
    """Arguments of an unconditional ``build_sampling_context`` call.

    Attributes:
        n_res: Residues to sample.
        template_bins: Width of the template distogram, overflow included.
    """

    n_res: int
    template_bins: int


class ContextScenario(enum.Enum):
    """Unconditional context sizes.

    Members:
        base: 4 residues, the 39-bin template distogram used in training.
        doubled: 8 residues; every atom and token axis doubles.
        template_20_bins: A 19-bin template plus overflow; the stored
            distogram must take its width from the module passed in.
    """

    base = ContextCase(4, 39)
    doubled = ContextCase(8, 39)
    template_20_bins = ContextCase(4, 20)


@dataclasses.dataclass(frozen=True)
class ScheduleCase:
    """Noise schedule parameters and the endpoints they imply.

    Attributes:
        sigma_data: Data standard deviation scaling every level.
        sigma_min: Smallest unscaled noise level.
        sigma_max: Largest unscaled noise level.
        rho: Schedule exponent.
    """

    sigma_data: float
    sigma_min: float
    sigma_max: float
    rho: float

    @property
    def sigma_at_zero(self) -> float:
        """Noise level the schedule must give at ``t = 0``.

        Returns:
            ``sigma_data * sigma_min``.
        """
        return self.sigma_data * self.sigma_min

    @property
    def sigma_at_one(self) -> float:
        """Noise level the schedule must give at ``t = 1``.

        Returns:
            ``sigma_data * sigma_max``.
        """
        return self.sigma_data * self.sigma_max


class ScheduleScenario(enum.Enum):
    """Schedules whose time axis must run from minimum to maximum noise.

    Members:
        pallatom_default: The training defaults, which production uses.
        unit_sigma_data: ``sigma_data = 1``, so the endpoints are the raw
            sigmas; catches a missing or doubled ``sigma_data`` factor.
        low_rho: ``rho = 3``; the interior changes, the endpoints must not.
    """

    pallatom_default = ScheduleCase(16.0, 4e-4, 160.0, 7.0)
    unit_sigma_data = ScheduleCase(1.0, 0.002, 80.0, 7.0)
    low_rho = ScheduleCase(16.0, 4e-4, 160.0, 3.0)


@dataclasses.dataclass(frozen=True)
class ScheduleCurve:
    """One schedule evaluated on 21 even points over ``[0, 1]``.

    Attributes:
        case: The schedule's parameters.
        sigmas: Noise levels at ``t = 0, 0.05, ..., 1``.
    """

    case: ScheduleCase
    sigmas: Float[torch.Tensor, " n_t"]


@dataclasses.dataclass(frozen=True)
class TrajectoryCase:
    """Sampler settings whose recorded trajectory must follow the schedule.

    Attributes:
        ddim_steps: Number of sampling steps ``T``.
        jitter: Value returned in place of every ``torch.rand()`` draw, so
            step ``k`` visits ``t = k / T - jitter / T`` exactly.
        eta_step_scale: Multiplier on each Euler step.
    """

    ddim_steps: int
    jitter: float
    eta_step_scale: float


class TrajectoryScenario(enum.Enum):
    """Fast sampler settings; all run with churn off.

    Members:
        two_steps: The smallest step count ``SamplerParams`` allows; the
            first step is also the second-to-last.
        ten_steps: A coarse grid with no jitter, landing on ``k / 10``.
        ten_steps_half_jitter: The same grid shifted down half a step, as
            the sampler's random offset would.
        ten_steps_gentle_step_scale: A smaller Euler step scale, so the
            per-step rescale differs from the default.
    """

    two_steps = TrajectoryCase(2, 0.0, 2.25)
    ten_steps = TrajectoryCase(10, 0.0, 2.25)
    ten_steps_half_jitter = TrajectoryCase(10, 0.5, 2.25)
    ten_steps_gentle_step_scale = TrajectoryCase(10, 0.5, 1.5)


class ProductionTrajectory(enum.Enum):
    """The production sampler settings, kept apart because they are slow.

    Members:
        pallatom_default: 200 steps, step scale 2.25, half-step jitter. It
            takes seconds where the others take milliseconds, so only the
            noise-level test runs it.
    """

    pallatom_default = TrajectoryCase(200, 0.5, 2.25)


@dataclasses.dataclass(frozen=True)
class FieldBreak:
    """An AllAtomContext field replaced by a tensor with one axis wrong.

    Attributes:
        field: Name of the field to replace.
        malform: Maps the valid tensor to the malformed one.
    """

    field: str
    malform: Callable[
        [Shaped[torch.Tensor, "..."]],
        Shaped[torch.Tensor, "..."],
    ]


class ContextBreak(enum.Enum):
    """AllAtomContext fields whose shape contract a replacement breaks.

    Members:
        r_gt_flat: ``r_gt`` loses its xyz axis.
        aa_residue_drift: ``aa_indices`` has one residue fewer than the
            residue mask.
    """

    r_gt_flat = FieldBreak("r_gt", lambda t: t[..., 0])
    aa_residue_drift = FieldBreak("aa_indices", lambda t: t[:, 1:])


@dataclasses.dataclass(frozen=True)
class SamplerTrajectory:
    """What the model saw during one sampling run, with exact expectations.

    "Centred RMS" is the RMS of a step's noisy input about its centroid,
    which measures how much noise the input carries. Recorded and expected
    sequences are compared in the tests, so a wrong number of steps fails
    there rather than while the fixture builds.

    Attributes:
        model_calls: Number of model calls the run made.
        step_sigmas: Noise level passed to the model at each step.
        expected_step_sigmas: ``noise_schedule(k / T - jitter / T)`` for
            ``k = T, ..., 1``.
        initial_centred_rms: Centred RMS of the first step's input.
        expected_initial_centred_rms: Centred RMS of the seeded initial
            draw scaled by the noise level the first call was told. Taken
            from the recorded call, so a schedule evaluated at the wrong
            end cannot make the expectation agree with the bug.
        step_rms_ratios: Centred RMS of each step's input over the
            previous step's.
        expected_step_rms_ratios: ``|1 + eta (sigma_next / sigma - 1)|``
            for each step but the last.
    """

    model_calls: int
    step_sigmas: Float[torch.Tensor, " n_recorded"]
    expected_step_sigmas: Float[torch.Tensor, " n_expected"]
    initial_centred_rms: Float[torch.Tensor, ""]
    expected_initial_centred_rms: Float[torch.Tensor, ""]
    step_rms_ratios: Float[torch.Tensor, " n_recorded_gaps"]
    expected_step_rms_ratios: Float[torch.Tensor, " n_expected_gaps"]


@pytest.fixture
def atom_disto() -> Distogram:
    """Return a 22-bin, 2 to 22 angstrom atom distogram without overflow.

    Returns:
        The module.
    """
    return Distogram(n_bins=22, min_dist=2.0, max_dist=22.0, overflow_bin=False)


@pytest.fixture
def templ_disto() -> Distogram:
    """Return the training template distogram: 38 bins plus overflow.

    Returns:
        The module, spanning 3.25 to 50.75 angstroms.
    """
    return Distogram(
        n_bins=38,
        min_dist=3.25,
        max_dist=50.75,
        overflow_bin=True,
    )


@pytest.fixture
def context(atom_disto: Distogram, templ_disto: Distogram) -> FeaturizedBatch:
    """Build the 4-residue unconditional context the sampler tests share.

    Args:
        atom_disto: Atom distogram.
        templ_disto: Template distogram.

    Returns:
        All-alanine context with zero ground-truth positions.
    """
    with torch.no_grad():
        return build_sampling_context(
            pdb_file_path=None,
            atom_distogram_fn=atom_disto,
            templ_distogram_fn=templ_disto,
            residue_number=4,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def context_scenario(request: pytest.FixtureRequest) -> ContextScenario:
    """Return the ``ContextScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(ContextScenario, request.param)


@pytest.fixture
def scenario_context(
    context_scenario: ContextScenario,
    atom_disto: Distogram,
) -> FeaturizedBatch:
    """Build an unconditional context at the scenario's size.

    Args:
        context_scenario: Residue count and template width.
        atom_disto: Atom distogram.

    Returns:
        The context.
    """
    case = context_scenario.value
    templ = Distogram(
        n_bins=case.template_bins - 1,
        min_dist=3.25,
        max_dist=50.75,
        overflow_bin=True,
    )
    with torch.no_grad():
        return build_sampling_context(
            pdb_file_path=None,
            atom_distogram_fn=atom_disto,
            templ_distogram_fn=templ,
            residue_number=case.n_res,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def default_noise() -> NoiseScheduleParams:
    """Return the 0.002 to 80 noise schedule most sampler tests run on.

    Returns:
        The schedule parameters.
    """
    return NoiseScheduleParams(sigma_min=0.002, sigma_max=80.0)


@pytest.fixture
def zero_sampler(
    context: FeaturizedBatch,
    templ_disto: Distogram,
    default_noise: NoiseScheduleParams,
) -> EDMSampler:
    """Build a four-step, churn-free sampler over ``D(z) = 0``.

    Args:
        context: Unconditional context.
        templ_disto: Template distogram.
        default_noise: Noise schedule.

    Returns:
        The sampler.
    """
    params = SamplerParams(S_churn=0.0, ddim_steps=4)
    denoiser = ScaledDenoiser(0.0)
    return build_sampler(denoiser, context, templ_disto, params, default_noise)


@pytest.fixture
def identity_sampler(
    context: FeaturizedBatch,
    templ_disto: Distogram,
    default_noise: NoiseScheduleParams,
) -> EDMSampler:
    """Build a four-step, churn-free sampler over ``D(z) = z``.

    Args:
        context: Unconditional context.
        templ_disto: Template distogram.
        default_noise: Noise schedule.

    Returns:
        The sampler.
    """
    params = SamplerParams(S_churn=0.0, ddim_steps=4)
    denoiser = ScaledDenoiser(1.0)
    return build_sampler(denoiser, context, templ_disto, params, default_noise)


@pytest.fixture
def churn_scenario(request: pytest.FixtureRequest) -> ChurnScenario:
    """Return the ``ChurnScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(ChurnScenario, request.param)


@pytest.fixture
def churned_sampler(
    churn_scenario: ChurnScenario,
    context: FeaturizedBatch,
    templ_disto: Distogram,
    default_noise: NoiseScheduleParams,
) -> EDMSampler:
    """Build a four-step sampler over ``D(z) = z`` with the scenario's churn.

    Args:
        churn_scenario: ``S_churn`` and ``S_tmin``.
        context: Unconditional context.
        templ_disto: Template distogram.
        default_noise: Noise schedule.

    Returns:
        The sampler.
    """
    case = churn_scenario.value
    params = SamplerParams(S_churn=case.churn, S_tmin=case.s_tmin, ddim_steps=4)
    denoiser = ScaledDenoiser(1.0)
    return build_sampler(denoiser, context, templ_disto, params, default_noise)


@pytest.fixture
def halving_samplers(
    context: FeaturizedBatch,
    templ_disto: Distogram,
    default_noise: NoiseScheduleParams,
) -> HalvingSamplers:
    """Build churn-free samplers over ``D(z) = z / 2`` at 4 and 10 steps.

    Args:
        context: Unconditional context.
        templ_disto: Template distogram.
        default_noise: Noise schedule.

    Returns:
        Both samplers.
    """
    four = SamplerParams(S_churn=0.0, ddim_steps=4)
    ten = SamplerParams(S_churn=0.0, ddim_steps=10)
    return HalvingSamplers(
        four_steps=build_sampler(
            ScaledDenoiser(0.5),
            context,
            templ_disto,
            four,
            default_noise,
        ),
        ten_steps=build_sampler(
            ScaledDenoiser(0.5),
            context,
            templ_disto,
            ten,
            default_noise,
        ),
    )


@pytest.fixture
def rho_schedules(
    context: FeaturizedBatch,
    templ_disto: Distogram,
    default_noise: NoiseScheduleParams,
) -> RhoCurves:
    """Evaluate the schedule with ``rho = 7`` and ``rho = 3`` on 11 points.

    The denoiser is never called; ``noise_schedule`` only reads the
    sampler's schedule parameters.

    Args:
        context: Unconditional context.
        templ_disto: Template distogram.
        default_noise: Noise schedule.

    Returns:
        Both curves.
    """
    grid = torch.linspace(0, 1, 11)
    curves = [
        torch.stack(
            [
                build_sampler(
                    ScaledDenoiser(0.0),
                    context,
                    templ_disto,
                    SamplerParams(rho=rho),
                    default_noise,
                ).noise_schedule(t)
                for t in grid
            ],
        )
        for rho in (7.0, 3.0)
    ]
    return RhoCurves(rho_7=curves[0], rho_3=curves[1])


@pytest.fixture
def schedule_curve(
    request: pytest.FixtureRequest,
    context: FeaturizedBatch,
    templ_disto: Distogram,
) -> ScheduleCurve:
    """Evaluate a schedule scenario on 21 even points over ``[0, 1]``.

    The denoiser is never called; ``noise_schedule`` only reads the
    sampler's schedule parameters.

    Args:
        request: Pytest request carrying a ``ScheduleScenario``.
        context: Unconditional context.
        templ_disto: Template distogram.

    Returns:
        The scenario's parameters and its noise levels on the grid.
    """
    case = cast(ScheduleScenario, request.param).value
    sampler = build_sampler(
        ScaledDenoiser(0.0),
        context,
        templ_disto,
        SamplerParams(rho=case.rho),
        NoiseScheduleParams(
            sigma_data=case.sigma_data,
            sigma_min=case.sigma_min,
            sigma_max=case.sigma_max,
        ),
    )
    grid = torch.linspace(0, 1, 21)
    sigmas = torch.stack([sampler.noise_schedule(t) for t in grid])
    return ScheduleCurve(case=case, sigmas=sigmas)


@pytest.fixture
def sampler_trajectory(
    request: pytest.FixtureRequest,
    context: FeaturizedBatch,
    templ_disto: Distogram,
    monkeypatch: pytest.MonkeyPatch,
) -> SamplerTrajectory:
    """Run the sampler for one scenario and record every model input.

    Four choices make every expectation exact rather than statistical:

    1. ``torch.rand`` is patched to return the scenario's jitter. The
       sampler draws ``torch.rand()`` only for the per-step time offset, so
       the visited times are exactly ``k / T - jitter / T``. The real
       sampler draws a separate offset per step, as Pallatom does; pinning
       them makes neighbouring steps agree, so these tests check the
       schedule direction and step targets, not that independence.
    2. A zero denoiser stands in for the trunk, so each Euler step only
       rescales the input by ``1 + eta (sigma_next - sigma) / sigma``.
    3. The seed is set to 0 before drawing the initial noise here and again
       before ``sample()``. With ``torch.rand`` patched, the sampler's first
       draw is that same ``torch.randn``.
    4. ``centre_random_augment`` is replaced by plain centring. Its ~1
       angstrom random translation every step would swamp the input in
       float32: with a zero denoiser over 200 steps the input shrinks to
       about 1e-10 angstrom.

    The last step's Euler update is discarded (``sample()`` returns the
    denoised prediction), so ratios cover every step but the last.

    Args:
        request: Pytest request carrying a trajectory scenario.
        context: Shared context, for the atom mask and count.
        templ_disto: Template distogram.
        monkeypatch: Pins ``torch.rand`` and replaces the augmentation.

    Returns:
        The recorded trajectory and its expectations.
    """
    case = cast(TrajectoryScenario | ProductionTrajectory, request.param).value
    shape = (1, context.r_gt.shape[1], 3)
    delta_t = 1 / case.ddim_steps
    denoiser = ScaledDenoiser(0.0)
    sampler = build_sampler(
        denoiser,
        context,
        templ_disto,
        SamplerParams(
            S_churn=0.0,
            ddim_steps=case.ddim_steps,
            eta_step_scale=case.eta_step_scale,
        ),
        NoiseScheduleParams(),
    )

    monkeypatch.setattr(torch, "rand", FixedRand(case.jitter))
    monkeypatch.setattr("sample.sampling.centre_random_augment", centre_only)
    _ = manual_seed(0)
    initial_draw = torch.randn(shape)
    _ = manual_seed(0)
    _ = sampler.sample(shape)
    steps = denoiser.calls[::2]
    step_sigmas = torch.stack([batch.t_hat[0] for batch in steps])
    step_rms = torch.stack([centred_rms(batch.r_gt_noised) for batch in steps])
    times = (torch.arange(case.ddim_steps, 0, -1) - case.jitter) * delta_t
    sigmas = torch.stack([sampler.noise_schedule(t) for t in times])
    next_sigmas = torch.stack(
        [sampler.noise_schedule((t - delta_t).clamp(min=0.0)) for t in times],
    )
    rescale = 1 + case.eta_step_scale * (next_sigmas / sigmas - 1)
    return SamplerTrajectory(
        model_calls=len(denoiser.calls),
        step_sigmas=step_sigmas,
        expected_step_sigmas=sigmas,
        initial_centred_rms=step_rms[0],
        expected_initial_centred_rms=centred_rms(
            step_sigmas[0] * centre_only(initial_draw, context.atom5_mask),
        ),
        step_rms_ratios=step_rms[1:] / step_rms[:-1],
        expected_step_rms_ratios=rescale.abs()[:-1],
    )


@pytest.fixture
def aa_ctx(atom_disto: Distogram) -> AllAtomContext:
    """Build a batch-of-two all-atom context for the sequence ``ACDEFG``.

    Args:
        atom_disto: Atom distogram.

    Returns:
        The context.
    """
    with torch.no_grad():
        return build_aa_context(
            aa_protein("ACDEFG"),
            atom_disto,
            batch_size=2,
            device="cpu",
        )


@pytest.fixture
def template_ctx(templ_disto: Distogram) -> TemplateContext:
    """Build a template context from a 4-residue protein.

    Args:
        templ_disto: Template distogram.

    Returns:
        The context.
    """
    return build_template_context(
        aa_protein("ACDE"),
        batch_size=1,
        distogram_fn=templ_disto,
        device="cpu",
    )


@pytest.fixture
def aa_batch_size(request: pytest.FixtureRequest) -> int:
    """Return the batch size routed in via indirect param.

    Args:
        request: Pytest request carrying the batch size in ``param``.

    Returns:
        Copies of the protein in the batch.
    """
    return cast(int, request.param)


@pytest.fixture
def batched_aa_ctx(aa_batch_size: int, atom_disto: Distogram) -> AllAtomContext:
    """Build an all-atom context for ``ACDEFG`` at the routed-in batch size.

    Args:
        aa_batch_size: Copies of the protein in the batch.
        atom_disto: Atom distogram.

    Returns:
        The context.
    """
    with torch.no_grad():
        return build_aa_context(
            aa_protein("ACDEFG"),
            atom_disto,
            batch_size=aa_batch_size,
            device="cpu",
        )


@pytest.fixture
def unknown_aa_ctx(atom_disto: Distogram) -> AllAtomContext:
    """Build an all-atom context for ``AXCX``, two of whose residues are X.

    Args:
        atom_disto: Atom distogram.

    Returns:
        The context.
    """
    with torch.no_grad():
        return build_aa_context(
            aa_protein("AXCX"),
            atom_disto,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def spread_template_ctx(templ_disto: Distogram) -> TemplateContext:
    """Build a template context from ``ACDE`` spread out 20 times.

    Args:
        templ_disto: Template distogram.

    Returns:
        The context.
    """
    protein = aa_protein("ACDE")
    spread = dataclasses.replace(
        protein,
        atom_positions=20 * protein.atom_positions,
    )
    return build_template_context(
        spread,
        batch_size=1,
        distogram_fn=templ_disto,
        device="cpu",
    )


@pytest.mark.parametrize(
    "context_scenario",
    list(ContextScenario),
    ids=[case.name for case in ContextScenario],
    indirect=True,
)
def test_build_sampling_context_shapes_and_index_maps(
    context_scenario: ContextScenario,
    scenario_context: FeaturizedBatch,
) -> None:
    """Every field has its shape; atoms map to residues and CA centres.

    Atom ``l`` belongs to residue ``l // 5`` with its CA at ``5 r + 1``, and
    the template distogram takes the template module's full width.

    Args:
        context_scenario: Residue count and template width.
        scenario_context: The context.
    """
    case = context_scenario.value
    x = scenario_context
    n_atom = case.n_res * NATOM
    residue = torch.arange(case.n_res).repeat_interleave(NATOM)
    shapes = {
        "ref_pos": (1, n_atom, 3),
        "ref_element": (1, n_atom, 4),
        "tok_idx": (1, n_atom),
        "center_uid": (1, n_atom),
        "noised_res_distogram": (1, case.n_res, case.n_res, case.template_bins),
        "f_pseudo_beta_mask": (1, case.n_res),
    }
    found = {
        name: tuple(cast(torch.Tensor, getattr(x, name)).shape)
        for name in shapes
    }
    assert found == shapes
    assert torch.equal(x.tok_idx[0], residue)
    assert torch.equal(x.ref_space_uid[0], residue)
    assert torch.equal(x.center_uid[0], NATOM * residue + 1)


@pytest.mark.parametrize(
    "context_scenario",
    list(ContextScenario),
    ids=[case.name for case in ContextScenario],
    indirect=True,
)
def test_build_sampling_context_holds_unconditional_placeholders(
    scenario_context: FeaturizedBatch,
) -> None:
    """The context is a finite all-alanine chain at the origin, all present.

    The template distogram peaks in bin 0, each atom has one element, and
    ``t_hat = 1`` and ``t_normalized = 0.5`` are the placeholders the
    sampler overwrites.

    Args:
        scenario_context: The context.
    """
    x = scenario_context
    tensors = [
        cast(torch.Tensor, getattr(x, f.name)) for f in dataclasses.fields(x)
    ]
    elements = reduce(x.ref_element, "b n e -> b n", "sum")
    checks = {
        "finite": all(bool(torch.isfinite(t.float()).all()) for t in tensors),
        "one_element_per_atom": torch.equal(
            elements,
            torch.ones_like(elements),
        ),
        "distogram_in_bin_0": not x.noised_res_distogram.argmax(-1).any(),
        "all_alanine": not x.aa_indices.any(),
        "at_origin": not x.r_gt.any(),
        "atoms_present": bool(x.atom5_mask.all()),
        "pseudo_beta_present": bool((x.f_pseudo_beta_mask == 1).all()),
        "pairs_present": bool(x.gt_atom_distogram_mask_sparse.all()),
        "t_hat_is_1": torch.equal(x.t_hat, torch.ones(1)),
        "t_normalized_is_half": torch.equal(
            x.t_normalized,
            torch.full_like(x.t_normalized, 0.5),
        ),
    }
    assert checks == dict.fromkeys(checks, True)


@pytest.mark.parametrize(
    "schedule_curve",
    list(ScheduleScenario),
    ids=[case.name for case in ScheduleScenario],
    indirect=True,
)
def test_noise_schedule_rises_from_sigma_min_at_zero_to_sigma_max_at_one(
    schedule_curve: ScheduleCurve,
) -> None:
    """``t = 0`` is minimum noise, ``t = 1`` maximum, strictly rising between.

    Regression for a sampler that mixed time conventions: the schedule put
    maximum noise at ``t = 0`` (AF3) while the initial noise and Euler
    target assumed it at ``t = 1`` (Pallatom), so sampling started at the
    bottom of the schedule.

    Args:
        schedule_curve: Parameters and noise levels on the grid.
    """
    case, sigmas = schedule_curve.case, schedule_curve.sigmas
    ends = torch.stack([sigmas[0], sigmas[-1]])
    expected = torch.tensor([case.sigma_at_zero, case.sigma_at_one])
    torch.testing.assert_close(ends, expected, rtol=1e-5, atol=0)
    assert bool((sigmas[1:] > sigmas[:-1]).all())


def test_noise_schedule_rho_changes_only_the_interior(
    rho_schedules: RhoCurves,
) -> None:
    """``rho = 7`` and ``rho = 3`` share endpoints but differ in between.

    Not parametrized: the endpoint test covers each schedule on its own;
    this compares two.

    Args:
        rho_schedules: The schedule under both values of ``rho``.
    """
    rho_7, rho_3 = rho_schedules.rho_7, rho_schedules.rho_3
    torch.testing.assert_close(rho_7[[0, -1]], rho_3[[0, -1]])
    assert not torch.allclose(rho_7[1:-1], rho_3[1:-1])


@pytest.mark.parametrize(
    "sampler_trajectory",
    list(TrajectoryScenario),
    ids=[case.name for case in TrajectoryScenario],
    indirect=True,
)
def test_edm_sampler_calls_model_twice_per_step(
    sampler_trajectory: SamplerTrajectory,
) -> None:
    """``sample()`` runs every one of its steps, with two model calls each.

    Each step calls the model once to build the self-conditioning template
    and once more to take the Euler step. The loop once ran
    ``ddim_steps - 2`` steps and never reached the lowest noise levels.

    Args:
        sampler_trajectory: Recorded run.
    """
    assert sampler_trajectory.model_calls == 2 * len(
        sampler_trajectory.expected_step_sigmas,
    )


@pytest.mark.parametrize(
    "sampler_trajectory",
    [*TrajectoryScenario, *ProductionTrajectory],
    ids=[case.name for case in [*TrajectoryScenario, *ProductionTrajectory]],
    indirect=True,
)
def test_edm_sampler_visits_noise_levels_from_max_to_min(
    sampler_trajectory: SamplerTrajectory,
) -> None:
    """The model sees ``noise_schedule(k / T - jitter / T)`` for ``k = T..1``.

    So the loop counts down from maximum noise near ``t = 1`` to minimum
    noise near ``t = 0``, including at the production settings.

    Args:
        sampler_trajectory: Recorded run and expected noise levels.
    """
    torch.testing.assert_close(
        sampler_trajectory.step_sigmas,
        sampler_trajectory.expected_step_sigmas,
    )


@pytest.mark.parametrize(
    "sampler_trajectory",
    list(TrajectoryScenario),
    ids=[case.name for case in TrajectoryScenario],
    indirect=True,
)
def test_edm_sampler_initial_noise_matches_first_step(
    sampler_trajectory: SamplerTrajectory,
) -> None:
    """The starting coordinates carry the first step's noise level.

    Regression: the initial noise level was evaluated at the wrong end of
    the schedule, so the first model call saw about 0.007 angstrom of
    noise labelled as about 2,500.

    Args:
        sampler_trajectory: Recorded run and expected initial spread.
    """
    torch.testing.assert_close(
        sampler_trajectory.initial_centred_rms,
        sampler_trajectory.expected_initial_centred_rms,
    )


@pytest.mark.parametrize(
    "sampler_trajectory",
    list(TrajectoryScenario),
    ids=[case.name for case in TrajectoryScenario],
    indirect=True,
)
def test_edm_sampler_euler_steps_target_next_lower_noise_level(
    sampler_trajectory: SamplerTrajectory,
) -> None:
    """Each Euler step rescales the input toward the next lower noise level.

    Regression: the Euler target was one step toward higher noise, so late
    in sampling the residual noise drifted above the schedule.

    Args:
        sampler_trajectory: Recorded run and expected per-step rescales.
    """
    torch.testing.assert_close(
        sampler_trajectory.step_rms_ratios,
        sampler_trajectory.expected_step_rms_ratios,
    )


def test_edm_sampler_zero_denoiser_drives_output_to_zero(
    zero_sampler: EDMSampler,
    context: FeaturizedBatch,
) -> None:
    """With ``D(z, sigma) = 0`` each step scales ``z`` toward 0, ending at 0.

    The last step targets ``sigma = 0``, so the output is exactly zero, as
    are the zero sequence logits. Not parametrized: the result does not
    depend on the step count.

    Args:
        zero_sampler: Four-step sampler over ``D(z) = 0``.
        context: Supplies the atom count.
    """
    out, seq = zero_sampler.sample((1, context.r_gt.shape[1], 3))
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0)
    assert torch.count_nonzero(seq) == 0


def test_edm_sampler_identity_denoiser_gives_finite_centred_output(
    identity_sampler: EDMSampler,
    context: FeaturizedBatch,
) -> None:
    """With an identity denoiser the output is finite and centred.

    The tolerance is 1e-6 of the largest coordinate: the identity keeps
    the coordinates at the starting noise scale (about 1,280 angstroms),
    where float32 rounding moves the centroid about 1e-5. Without the final
    centring the ~1 angstrom per-step translation would remain, hundreds of
    times the tolerance. Not parametrized: one seeded run shows both.

    Args:
        identity_sampler: Four-step sampler over ``D(z) = z``.
        context: Supplies the atom count.
    """
    shape = (1, context.r_gt.shape[1], 3)
    _ = manual_seed(7)
    out, seq = identity_sampler.sample(shape)
    centroid = reduce(out, "b n d -> b d", "mean")
    assert out.shape == shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(seq).all()
    tolerance = 1e-6 * float(out.abs().max())
    torch.testing.assert_close(
        centroid,
        torch.zeros_like(centroid),
        atol=tolerance,
        rtol=0,
    )


@pytest.mark.parametrize(
    "churn_scenario",
    list(ChurnScenario),
    ids=[case.name for case in ChurnScenario],
    indirect=True,
)
def test_edm_sampler_churn_applies_only_inside_its_window(
    churn_scenario: ChurnScenario,
    identity_sampler: EDMSampler,
    churned_sampler: EDMSampler,
    context: FeaturizedBatch,
) -> None:
    """Churn changes the result, unless ``S_tmin`` puts every step outside.

    The gate compares ``S_tmin`` with ``timestep / ddim_steps``, which never
    exceeds 1, so ``S_tmin > 1`` disables injection and the run equals the
    ``S_churn = 0`` one under the same seed. With the identity denoiser the
    injected noise is not filtered, so any injection shows.

    Args:
        churn_scenario: Churn settings and whether the runs must match.
        identity_sampler: The churn-free run.
        churned_sampler: The same sampler with the scenario's churn.
        context: Supplies the atom count.
    """
    shape = (1, context.r_gt.shape[1], 3)
    _ = manual_seed(5)
    out_ode, _ = identity_sampler.sample(shape)
    _ = manual_seed(5)
    out_churned, seq = churned_sampler.sample(shape)
    matches = torch.allclose(out_ode, out_churned)
    assert matches == churn_scenario.value.matches_ode
    assert torch.isfinite(seq).all()


def test_edm_sampler_repeats_exactly_under_the_same_seed(
    identity_sampler: EDMSampler,
    context: FeaturizedBatch,
) -> None:
    """A second ``sample()`` with the same seed gives bit-identical output.

    Shows no state carries over between calls, such as a mutated context.
    Not parametrized: churn is off, so the seed fixes every draw.

    Args:
        identity_sampler: Four-step sampler over ``D(z) = z``.
        context: Supplies the atom count.
    """
    shape = (1, context.r_gt.shape[1], 3)
    _ = manual_seed(3)
    first = identity_sampler.sample(shape)
    _ = manual_seed(3)
    second = identity_sampler.sample(shape)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_edm_sampler_step_count_changes_a_nontrivial_trajectory(
    halving_samplers: HalvingSamplers,
    context: FeaturizedBatch,
) -> None:
    """With ``D(z) = z / 2``, 4 and 10 steps integrate to different outputs.

    The output then depends on the whole noise grid, so a sampler that
    ignored ``ddim_steps`` would give the same result. Not parametrized:
    two step counts suffice.

    Args:
        halving_samplers: The same sampler at 4 and at 10 steps.
        context: Supplies the atom count.
    """
    shape = (1, context.r_gt.shape[1], 3)
    _ = manual_seed(9)
    four, _ = halving_samplers.four_steps.sample(shape)
    _ = manual_seed(9)
    ten, _ = halving_samplers.ten_steps.sample(shape)
    assert not torch.allclose(four, ten)


@pytest.mark.parametrize("aa_batch_size", [1, 2], indirect=True)
def test_build_aa_context_shapes_follow_batch_and_protein(
    aa_batch_size: int,
    batched_aa_ctx: AllAtomContext,
) -> None:
    """Every field leads with ``batch_size``, then the atom or residue axis.

    ``K``, the sparse neighbour count, is read from the distogram field
    itself and must match between the distogram and its mask.

    Args:
        aa_batch_size: Copies of the protein in the batch.
        batched_aa_ctx: Context for ``ACDEFG`` at that batch size.
    """
    ctx = batched_aa_ctx
    b, n_res, n_atom = aa_batch_size, 6, 6 * NATOM
    k = ctx.gt_atom_distogram_sparse.shape[2]
    shapes = {
        f.name: tuple(cast(torch.Tensor, getattr(ctx, f.name)).shape)
        for f in dataclasses.fields(ctx)
    }
    assert shapes == {
        "r_gt": (b, n_atom, 3),
        "atom5_mask": (b, n_atom),
        "residue_mask": (b, n_res),
        "gt_atom_distogram_sparse": (b, n_atom, k),
        "gt_atom_distogram_mask_sparse": (b, n_atom, k),
        "aa_indices": (b, n_res),
        "f_residue_idx": (b, n_res),
    }


def test_build_aa_context_tiles_the_protein_into_every_batch_item(
    aa_ctx: AllAtomContext,
) -> None:
    """Both batch items hold the same atoms and the input sequence.

    Every atom and residue is present, so both masks are fully set. Not
    parametrized: the batch size is covered by the shape test.

    Args:
        aa_ctx: Context for ``ACDEFG`` tiled twice.
    """
    sequence = torch.tensor([restype_order[c] for c in "ACDEFG"])
    assert torch.equal(aa_ctx.r_gt[0], aa_ctx.r_gt[1])
    assert torch.equal(aa_ctx.aa_indices, repeat(sequence, "n -> 2 n"))
    assert bool(aa_ctx.atom5_mask.all())
    assert bool(aa_ctx.residue_mask.all())
    assert torch.isfinite(aa_ctx.r_gt).all()


def test_build_aa_context_keeps_unknown_residues_as_x(
    unknown_aa_ctx: AllAtomContext,
) -> None:
    """An ``X`` residue reaches ``aa_indices`` as ``RESTYPE_NUM_NO_X``.

    It must not be remapped to a canonical residue, or sequence-conditioned
    sampling would condition on a residue that is not there. Not
    parametrized: one mixed sequence covers known and unknown residues.

    Args:
        unknown_aa_ctx: Context for ``AXCX``.
    """
    ctx = unknown_aa_ctx
    a, c, x = restype_order["A"], restype_order["C"], RESTYPE_NUM_NO_X
    assert torch.equal(ctx.aa_indices[0], torch.tensor([a, x, c, x]))


def test_build_template_context_is_a_symmetric_one_hot_distogram(
    template_ctx: TemplateContext,
) -> None:
    """The template distogram is float, one-hot per pair, and symmetric.

    The pseudo-Cβ mask is a fully set ``long`` tensor. Not parametrized:
    the properties hold for any protein.

    Args:
        template_ctx: Context for a 4-residue protein.
    """
    disto = template_ctx.f_template_distogram
    assert disto.shape == (1, 4, 4, 39)
    assert disto.dtype == torch.float32
    assert torch.equal(
        reduce(disto, "b i j k -> b i j", "sum"),
        torch.ones(1, 4, 4),
    )
    assert torch.equal(disto, rearrange(disto, "b i j k -> b j i k"))
    assert template_ctx.f_pseudo_beta_mask.dtype == torch.long
    assert bool((template_ctx.f_pseudo_beta_mask == 1).all())


def test_build_template_context_follows_the_coordinates(
    spread_template_ctx: TemplateContext,
    template_ctx: TemplateContext,
) -> None:
    """Spreading the same protein out 20 times changes its template distogram.

    A constant output would pass the property test above. Not
    parametrized: one rescaling shows the coordinates are read.

    Args:
        spread_template_ctx: Context for the protein spread out 20 times.
        template_ctx: Context for the unscaled protein.
    """
    assert not torch.equal(
        spread_template_ctx.f_template_distogram,
        template_ctx.f_template_distogram,
    )


@pytest.mark.parametrize(
    "context_break",
    list(ContextBreak),
    ids=[case.name for case in ContextBreak],
)
def test_all_atom_context_rejects_a_malformed_field(
    aa_ctx: AllAtomContext,
    context_break: ContextBreak,
) -> None:
    """Rebuilding a valid context with one malformed field raises.

    Every other field stays valid, so the error comes from the replaced
    one.

    Args:
        aa_ctx: Valid context.
        context_break: Field to replace and how to break it.
    """
    case = context_break.value
    old = cast(Shaped[torch.Tensor, "..."], getattr(aa_ctx, case.field))
    with pytest.raises(TypeCheckError):
        _ = dataclasses.replace(aa_ctx, **{case.field: case.malform(old)})


def test_template_context_rejects_a_distogram_without_bins(
    template_ctx: TemplateContext,
) -> None:
    """A template distogram with no bin axis fails the contract.

    Not parametrized: the mask has no axis left to drop independently.

    Args:
        template_ctx: Valid context.
    """
    flat = template_ctx.f_template_distogram[..., 0]
    with pytest.raises(TypeCheckError):
        _ = dataclasses.replace(template_ctx, f_template_distogram=flat)

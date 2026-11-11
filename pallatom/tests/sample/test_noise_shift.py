"""Tests for the sampler noise shift (``SamplerParams.noise_shift``).

The noise shift multiplies every noise level of the sampling schedule by a
constant ``alpha``. It is the EDM form of the rectified-flow timestep shift of
Esser et al. (2024), and is swept on trained checkpoints without retraining.
"""

import dataclasses
import enum

import pytest
import torch
from architecture.main_trunk import MainTrunk
from helpers.data import Distogram, FeaturizedBatch, build_distogram_module
from jaxtyping import Float
from pydantic import ValidationError
from sample.sample_config import SamplerParams
from sample.sampling import EDMSampler, build_sampling_context
from train.train_config import (
    AtomDistogramParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
    TemplateDistogramParams,
)


@dataclasses.dataclass(frozen=True)
class ShiftCase:
    """One value of the sampler noise shift.

    Attributes:
        noise_shift: Multiplier applied to every schedule noise level.
    """

    noise_shift: float


class NoiseShift(enum.Enum):
    """Valid noise shifts the schedule must scale by exactly.

    Members:
        identity: 1.0, the default; the schedule must be unchanged.
        halved: 0.5, a shift toward lower noise.
        doubled: 2.0, a shift toward higher noise.
        sd3_value: 3.0, the value Esser et al. (2024) chose at 1024x1024.
    """

    identity = ShiftCase(noise_shift=1.0)
    halved = ShiftCase(noise_shift=0.5)
    doubled = ShiftCase(noise_shift=2.0)
    sd3_value = ShiftCase(noise_shift=3.0)


class InvalidNoiseShift(enum.Enum):
    """Noise shifts ``SamplerParams`` must reject.

    Members:
        zero: 0.0 would collapse every noise level to zero.
        negative: -1.0 would flip the sign of every noise level.
    """

    zero = ShiftCase(noise_shift=0.0)
    negative = ShiftCase(noise_shift=-1.0)


@pytest.fixture
def template_distogram() -> Distogram:
    """Template distogram module built from the library's default params.

    Returns:
        The self-conditioning template distogram used in training.
    """
    return build_distogram_module(TemplateDistogramParams())


@pytest.fixture
def atom_distogram() -> Distogram:
    """Atom distogram module built from the library's default params.

    Returns:
        The sparse atom-pair distogram used in training.
    """
    return build_distogram_module(AtomDistogramParams())


@pytest.fixture
def sampling_context(
    atom_distogram: Distogram,
    template_distogram: Distogram,
) -> FeaturizedBatch:
    """Unconditional all-alanine sampling context of four residues.

    Four residues is the smallest protein that still gives every residue
    neighbours on both sides of the chain, and keeps the end-to-end test
    fast on CPU.

    Args:
        atom_distogram: Atom distogram module used to build the context.
        template_distogram: Template distogram module used to build the
            context.

    Returns:
        A batch-of-one sampling context on CPU.
    """
    with torch.no_grad():
        return build_sampling_context(
            pdb_file_path=None,
            atom_distogram_fn=atom_distogram,
            templ_distogram_fn=template_distogram,
            residue_number=4,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def small_trunk() -> MainTrunk:
    """Untrained two-unit MainTrunk in eval mode.

    Two decoder units instead of the default eight keep the end-to-end
    sampling test fast. The weights are random; the tests check how the
    sampler drives the model, not what the model predicts.

    Returns:
        A MainTrunk built from default params apart from ``K_unit``.
    """
    return MainTrunk(
        model_params=ModelParams(K_unit=2),
        template_distogram_params=TemplateDistogramParams(),
        residue_distogram_params=ResidueDistogramParams(),
        atom_distogram_params=AtomDistogramParams(),
        noise_params=NoiseScheduleParams(),
    ).eval()


@pytest.fixture
def schedule_times() -> Float[torch.Tensor, "n_t"]:
    """Schedule times from minimum (0) to maximum (1) noise in steps of 0.25.

    Returns:
        The five times ``0, 0.25, 0.5, 0.75, 1``.
    """
    return torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.fixture
def unshifted_sampler(
    small_trunk: MainTrunk,
    sampling_context: FeaturizedBatch,
    template_distogram: Distogram,
) -> EDMSampler:
    """Deterministic two-step sampler with the default (unshifted) schedule.

    Args:
        small_trunk: Model the sampler drives.
        sampling_context: Static per-protein sampling inputs.
        template_distogram: Template distogram module for self-conditioning.

    Returns:
        An EDMSampler built from default ``SamplerParams`` apart from churn
        and step count.
    """
    return EDMSampler(
        small_trunk,
        sampling_context,
        template_distogram,
        SamplerParams(S_churn=0.0, ddim_steps=2),
        NoiseScheduleParams(),
    )


@pytest.fixture
def shifted_sampler(
    noise_shift_scenario: NoiseShift,
    small_trunk: MainTrunk,
    sampling_context: FeaturizedBatch,
    template_distogram: Distogram,
) -> EDMSampler:
    """Deterministic two-step sampler with the scenario's noise shift.

    Factory fixture: identical to ``unshifted_sampler`` except for
    ``noise_shift``.

    Args:
        noise_shift_scenario: The shift to apply; the requesting test
            parametrizes it.
        small_trunk: Model the sampler drives.
        sampling_context: Static per-protein sampling inputs.
        template_distogram: Template distogram module for self-conditioning.

    Returns:
        An EDMSampler whose schedule is shifted by the scenario's value.
    """
    return EDMSampler(
        small_trunk,
        sampling_context,
        template_distogram,
        SamplerParams(
            S_churn=0.0,
            ddim_steps=2,
            noise_shift=noise_shift_scenario.value.noise_shift,
        ),
        NoiseScheduleParams(),
    )


@pytest.fixture
def unshifted_schedule(
    unshifted_sampler: EDMSampler,
    schedule_times: Float[torch.Tensor, "n_t"],
) -> Float[torch.Tensor, "n_t"]:
    """Unshifted noise levels at each schedule time.

    Args:
        unshifted_sampler: Sampler with the default schedule.
        schedule_times: Times at which to evaluate the schedule.

    Returns:
        ``noise_schedule(t)`` for each time, in order.
    """
    return torch.stack(
        [unshifted_sampler.noise_schedule(t) for t in schedule_times],
    )


@pytest.fixture
def shifted_schedule(
    shifted_sampler: EDMSampler,
    schedule_times: Float[torch.Tensor, "n_t"],
) -> Float[torch.Tensor, "n_t"]:
    """Shifted noise levels at each schedule time.

    Args:
        shifted_sampler: Sampler with the scenario's noise shift.
        schedule_times: Times at which to evaluate the schedule.

    Returns:
        ``noise_schedule(t)`` for each time, in order.
    """
    return torch.stack(
        [shifted_sampler.noise_schedule(t) for t in schedule_times],
    )


@pytest.fixture
def shifted_sample_coords(
    shifted_sampler: EDMSampler,
    sampling_context: FeaturizedBatch,
) -> Float[torch.Tensor, "B N_atom 3"]:
    """Coordinates from a full shifted sampling run with a real trunk.

    Args:
        shifted_sampler: Sampler with the scenario's noise shift.
        sampling_context: Context whose coordinate shape the run produces.

    Returns:
        The sampled atom5 coordinates.
    """
    batch, n_atom, n_dim = sampling_context.r_gt.shape
    with torch.no_grad():
        coords, _ = shifted_sampler.sample(shape=(batch, n_atom, n_dim))
    return coords


@pytest.mark.parametrize(
    "noise_shift_scenario",
    list(NoiseShift),
    ids=[case.name for case in NoiseShift],
)
def test_noise_shift_multiplies_every_schedule_level(
    shifted_schedule: Float[torch.Tensor, "n_t"],
    unshifted_schedule: Float[torch.Tensor, "n_t"],
    noise_shift_scenario: NoiseShift,
) -> None:
    """The shifted schedule is the unshifted schedule times the shift.

    Arrange: two samplers that differ only in ``noise_shift``, evaluated at
    the same times from minimum to maximum noise.
    Act: compare the two schedules.
    Assert: at every time, including both endpoints, the shifted noise level
    equals ``noise_shift`` times the unshifted one. The ``identity`` case
    shows the default leaves the schedule unchanged.

    Args:
        shifted_schedule: Noise levels with the scenario's shift.
        unshifted_schedule: Noise levels with the default schedule.
        noise_shift_scenario: The shift under test; parametrized.
    """
    torch.testing.assert_close(
        shifted_schedule,
        noise_shift_scenario.value.noise_shift * unshifted_schedule,
    )


@pytest.mark.parametrize(
    "invalid_noise_shift",
    list(InvalidNoiseShift),
    ids=[case.name for case in InvalidNoiseShift],
)
def test_sampler_params_reject_non_positive_noise_shift(
    invalid_noise_shift: InvalidNoiseShift,
) -> None:
    """``SamplerParams`` refuses a noise shift that is not positive.

    Arrange: a non-positive shift from the scenario.
    Act: construct ``SamplerParams`` with it.
    Assert: pydantic raises ``ValidationError``, so a sweep config with a
    typo fails at load time instead of producing a degenerate schedule.

    Args:
        invalid_noise_shift: The shift that must be rejected; parametrized.
    """
    with pytest.raises(ValidationError):
        _ = SamplerParams(noise_shift=invalid_noise_shift.value.noise_shift)


@pytest.mark.parametrize(
    "noise_shift_scenario",
    list(NoiseShift),
    ids=[case.name for case in NoiseShift],
)
def test_shifted_sampling_runs_end_to_end(
    shifted_sample_coords: Float[torch.Tensor, "B N_atom 3"],
    sampling_context: FeaturizedBatch,
) -> None:
    """A full sampling run with a real trunk works at every shift.

    Arrange: a two-step deterministic sampler with the scenario's shift,
    driving an untrained ``MainTrunk``.
    Act: run ``sample`` to completion.
    Assert: the output has the context's coordinate shape and is finite, so
    shifting the schedule, including the prior's scale and the template
    weight computed from the shifted sigma, does not break the loop.

    Args:
        shifted_sample_coords: Coordinates from the shifted run.
        sampling_context: Context whose coordinate shape the run must match.
    """
    assert shifted_sample_coords.shape == sampling_context.r_gt.shape
    assert bool(torch.isfinite(shifted_sample_coords).all())

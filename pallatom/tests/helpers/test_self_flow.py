"""Tests for Self-Flow's dual-timestep noising helpers.

Covers the per-residue noise levels of the student and teacher views, the
residue mask, zero-centre-of-mass noise, the pairwise template time, the
batched view container, and the ``SelfFlowParams`` bounds.
"""

import dataclasses
import enum

import pytest
import torch
from helpers.self_flow import (
    DualTimestepSigmas,
    NoisedView,
    NoisedViewBatch,
    centered_noise_displacement,
    dual_timestep_sigmas,
    pair_template_time,
    per_atom_sigma,
    sample_residue_mask,
)
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float
from pydantic import ValidationError
from train.train_config import SelfFlowParams

# ---------------------------------------------------------------------------
# dual_timestep_sigmas
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SigmaCase:
    """Inputs and expected noise levels for one three-residue protein.

    Attributes:
        first: Noise level of unmasked residues.
        second: Noise level of masked residues.
        noise_mask: 1 for residues that take ``second``.
        residue_mask: 1 for residues that exist (0 for padding).
        student: Expected student noise level of each residue.
        teacher: Expected teacher noise level.
    """

    first: float
    second: float
    noise_mask: tuple[int, int, int]
    residue_mask: tuple[int, int, int]
    student: tuple[float, float, float]
    teacher: float


class SigmaScenario(enum.Enum):
    """Residue masks and the noise levels they must produce.

    Members:
        second_higher: One residue at the higher second level; the teacher
            takes the first level.
        second_lower: One residue at the lower second level; the teacher
            takes the second level.
        none_masked: No residue masked; both views are at the first level.
        all_masked: Every residue masked; both views are at the second.
        padding_ignored: Only a padded residue is masked at a lower level;
            it must not lower the teacher.
        no_valid: No residue exists; the teacher falls back to the first
            level.
    """

    second_higher = SigmaCase(2, 5, (0, 1, 0), (1, 1, 1), (2, 5, 2), 2)
    second_lower = SigmaCase(5, 2, (1, 0, 0), (1, 1, 1), (2, 5, 5), 2)
    none_masked = SigmaCase(3, 1, (0, 0, 0), (1, 1, 1), (3, 3, 3), 3)
    all_masked = SigmaCase(4, 6, (1, 1, 1), (1, 1, 1), (6, 6, 6), 6)
    padding_ignored = SigmaCase(5, 1, (0, 0, 1), (1, 1, 0), (5, 5, 1), 5)
    no_valid = SigmaCase(4, 1, (1, 0, 0), (0, 0, 0), (1, 4, 4), 4)


@pytest.fixture
def sigmas(sigma_scenario: SigmaScenario) -> DualTimestepSigmas:
    """Noise levels computed for the scenario.

    Args:
        sigma_scenario: Inputs to ``dual_timestep_sigmas``; the requesting
            test parametrizes it.

    Returns:
        The student and teacher noise levels.
    """
    case = sigma_scenario.value
    return dual_timestep_sigmas(
        torch.tensor(case.first, dtype=torch.float32),
        torch.tensor(case.second, dtype=torch.float32),
        torch.tensor(case.noise_mask, dtype=torch.bool),
        torch.tensor(case.residue_mask, dtype=torch.bool),
    )


@pytest.mark.parametrize(
    "sigma_scenario",
    list(SigmaScenario),
    ids=[case.name for case in SigmaScenario],
)
def test_dual_timestep_sigmas_student_and_teacher_levels(
    sigmas: DualTimestepSigmas,
    sigma_scenario: SigmaScenario,
) -> None:
    """Masked residues take the second level; the teacher the lowest valid.

    Arrange: two noise levels, a residue mask and a padding mask.
    Act: ``dual_timestep_sigmas``.
    Assert: the student's per-residue levels and the teacher's single level
    match the hand-worked values, so the teacher is never noisier than any
    valid student residue and ignores padding.

    Args:
        sigmas: Computed noise levels.
        sigma_scenario: Expected levels; parametrized.
    """
    case = sigma_scenario.value
    torch.testing.assert_close(
        sigmas.student,
        torch.tensor(case.student, dtype=torch.float32),
    )
    torch.testing.assert_close(
        sigmas.teacher,
        torch.tensor(case.teacher, dtype=torch.float32),
    )


# ---------------------------------------------------------------------------
# sample_residue_mask
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MaskRateCase:
    """Masking settings and the fraction of residues they must mask.

    Attributes:
        n_res: Residues drawn in one call.
        mask_ratio: Per-residue masking probability.
        apply_prob: Probability the protein is masked at all.
        expected_fraction: Expected fraction of masked residues.
        tolerance: Allowed deviation from ``expected_fraction``.
    """

    n_res: int
    mask_ratio: float
    apply_prob: float
    expected_fraction: float
    tolerance: float


class MaskRate(enum.Enum):
    """Residue masking rates, exact or within sampling error.

    Members:
        never_applied: ``apply_prob`` 0 masks nothing.
        zero_ratio: ``mask_ratio`` 0 masks nothing.
        quarter: The paper's image setting, 0.25.
        half: The paper's upper bound, 0.5.
    """

    never_applied = MaskRateCase(1000, 0.5, 0.0, 0.0, 0.0)
    zero_ratio = MaskRateCase(1000, 0.0, 1.0, 0.0, 0.0)
    quarter = MaskRateCase(20000, 0.25, 1.0, 0.25, 0.02)
    half = MaskRateCase(20000, 0.5, 1.0, 0.5, 0.02)


@pytest.fixture
def masked_fraction(mask_rate_scenario: MaskRate) -> float:
    """Fraction of residues one seeded mask draw marks.

    Args:
        mask_rate_scenario: Masking settings; the requesting test
            parametrizes it.

    Returns:
        The fraction of masked residues.
    """
    case = mask_rate_scenario.value
    _ = manual_seed(0)
    mask = sample_residue_mask(case.n_res, case.mask_ratio, case.apply_prob)
    return float(mask.float().mean())


@pytest.mark.parametrize(
    "mask_rate_scenario",
    list(MaskRate),
    ids=[case.name for case in MaskRate],
)
def test_sample_residue_mask_masks_at_the_configured_rate(
    masked_fraction: float,
    mask_rate_scenario: MaskRate,
) -> None:
    """Each residue is masked independently at ``mask_ratio``.

    Arrange: masking settings and a fixed seed.
    Act: draw one mask over many residues.
    Assert: the masked fraction is within sampling error of the expected
    rate, and exactly 0 when either probability is 0.

    Args:
        masked_fraction: Fraction of residues masked.
        mask_rate_scenario: Expected fraction and tolerance; parametrized.
    """
    case = mask_rate_scenario.value
    assert abs(masked_fraction - case.expected_fraction) <= case.tolerance


@dataclasses.dataclass(frozen=True)
class ApplyCase:
    """How often whole proteins must be masked.

    Attributes:
        apply_prob: Probability the protein is masked at all.
        tolerance: Allowed deviation of the observed rate.
    """

    apply_prob: float
    tolerance: float


class ApplyRate(enum.Enum):
    """Per-protein application rates.

    Members:
        quarter: A quarter of proteins masked.
        three_quarters: Three quarters of proteins masked.
    """

    quarter = ApplyCase(apply_prob=0.25, tolerance=0.05)
    three_quarters = ApplyCase(apply_prob=0.75, tolerance=0.05)


@pytest.fixture
def masked_protein_rate(apply_rate_scenario: ApplyRate) -> float:
    """Fraction of 2000 seeded 64-residue draws with any residue masked.

    With ``mask_ratio`` 0.5 and 64 residues, a masked protein has at least
    one masked residue except with probability ``0.5**64``, so this
    fraction measures ``apply_prob``.

    Args:
        apply_rate_scenario: Application rate; the requesting test
            parametrizes it.

    Returns:
        The fraction of draws with at least one masked residue.
    """
    _ = manual_seed(0)
    draws = [
        bool(
            sample_residue_mask(
                64,
                0.5,
                apply_rate_scenario.value.apply_prob,
            ).any(),
        )
        for _ in range(2000)
    ]
    return sum(draws) / len(draws)


@pytest.mark.parametrize(
    "apply_rate_scenario",
    list(ApplyRate),
    ids=[case.name for case in ApplyRate],
)
def test_sample_residue_mask_applies_to_the_configured_share_of_proteins(
    masked_protein_rate: float,
    apply_rate_scenario: ApplyRate,
) -> None:
    """A protein is masked at all with probability ``apply_prob``.

    Arrange: many seeded mask draws.
    Act: count draws with any masked residue.
    Assert: the share is within sampling error of ``apply_prob``.

    Args:
        masked_protein_rate: Share of draws with a masked residue.
        apply_rate_scenario: Expected share; parametrized.
    """
    case = apply_rate_scenario.value
    assert abs(masked_protein_rate - case.apply_prob) <= case.tolerance


# ---------------------------------------------------------------------------
# centered_noise_displacement, pair_template_time, per_atom_sigma
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CentringCase:
    """Scaled noise along x for three atoms and the centred result.

    Attributes:
        noise_x: x component of each atom's scaled noise (y, z are 0).
        atom_mask: 1 for atoms that exist.
        expected_x: x component after centring.
    """

    noise_x: tuple[float, float, float]
    atom_mask: tuple[int, int, int]
    expected_x: tuple[float, float, float]


class Centring(enum.Enum):
    """Noise centring with and without a padded atom.

    Members:
        all_valid: Mean 2 is removed from every atom.
        padded_atom: The padded atom's 100 is excluded from the mean (2)
            but still shifted by it.
    """

    all_valid = CentringCase((1.0, 3.0, 2.0), (1, 1, 1), (-1.0, 1.0, 0.0))
    padded_atom = CentringCase((1.0, 3.0, 100.0), (1, 1, 0), (-1.0, 1.0, 98.0))


@pytest.fixture
def centred_noise(centring_scenario: Centring) -> Float[torch.Tensor, "N 3"]:
    """Centred noise for the scenario.

    Args:
        centring_scenario: Noise and mask; the requesting test parametrizes
            it.

    Returns:
        The noise with its valid-atom centre of mass removed.
    """
    case = centring_scenario.value
    noise = torch.zeros(3, 3)
    noise[:, 0] = torch.tensor(case.noise_x)
    return centered_noise_displacement(
        noise,
        torch.tensor(case.atom_mask, dtype=torch.bool),
    )


@pytest.fixture
def expected_centred_noise(
    centring_scenario: Centring,
) -> Float[torch.Tensor, "N 3"]:
    """Hand-computed centred noise for the scenario.

    Args:
        centring_scenario: Expected x components; the requesting test
            parametrizes it.

    Returns:
        The expected centred noise.
    """
    expected = torch.zeros(3, 3)
    expected[:, 0] = torch.tensor(centring_scenario.value.expected_x)
    return expected


@pytest.mark.parametrize(
    "centring_scenario",
    list(Centring),
    ids=[case.name for case in Centring],
)
def test_centered_noise_displacement_removes_valid_atom_mean(
    centred_noise: Float[torch.Tensor, "N 3"],
    expected_centred_noise: Float[torch.Tensor, "N 3"],
) -> None:
    """Centring subtracts the mean over valid atoms from every atom.

    Arrange: scaled noise with and without a padded atom.
    Act: ``centered_noise_displacement``.
    Assert: the result matches the hand-worked values, so the valid atoms'
    centre of mass is zero and padding does not bias it.

    Args:
        centred_noise: Centred noise.
        expected_centred_noise: Hand-computed centred noise.
    """
    torch.testing.assert_close(centred_noise, expected_centred_noise)


@pytest.fixture
def residue_times() -> Float[torch.Tensor, "N_res"]:
    """Normalized template times of three residues.

    A hard-coded data fixture: the pairwise maximum has one obvious
    expected table for distinct times.

    Returns:
        Times 0.2, 0.7 and 0.4.
    """
    return torch.tensor([0.2, 0.7, 0.4])


@pytest.fixture
def expected_pair_times() -> Float[torch.Tensor, "N_res N_res"]:
    """Larger time of each pair of ``residue_times``.

    Returns:
        The hand-written pairwise maximum.
    """
    return torch.tensor(
        [[0.2, 0.7, 0.4], [0.7, 0.7, 0.7], [0.4, 0.7, 0.4]],
    )


def test_pair_template_time_takes_the_noisier_residue(
    residue_times: Float[torch.Tensor, "N_res"],
    expected_pair_times: Float[torch.Tensor, "N_res N_res"],
) -> None:
    """Each pair's template time is the larger of its two residues' times.

    No other cases: the pairwise maximum of three distinct values covers
    every ordering of a pair, and the diagonal covers equal values.

    Args:
        residue_times: Per-residue times.
        expected_pair_times: Hand-written pairwise maximum.
    """
    torch.testing.assert_close(
        pair_template_time(residue_times),
        expected_pair_times,
    )


@pytest.fixture
def two_residue_sigma() -> Float[torch.Tensor, "N_res"]:
    """Noise levels 1 and 2 for two residues.

    Returns:
        The two levels.
    """
    return torch.tensor([1.0, 2.0])


def test_per_atom_sigma_repeats_each_residue_over_its_atoms(
    two_residue_sigma: Float[torch.Tensor, "N_res"],
) -> None:
    """Each residue's sigma covers its own atoms, ready to scale xyz noise.

    No other cases: repetition is the whole behaviour, and two residues of
    two atoms show both the order and the broadcast axis.

    Args:
        two_residue_sigma: Per-residue noise levels.
    """
    torch.testing.assert_close(
        per_atom_sigma(two_residue_sigma, 2),
        torch.tensor([[1.0], [1.0], [2.0], [2.0]]),
    )


# ---------------------------------------------------------------------------
# NoisedViewBatch
# ---------------------------------------------------------------------------


@pytest.fixture
def views() -> list[NoisedView]:
    """Two two-residue views with every field filled by its view index.

    Returns:
        A view of zeros and a view of ones, ten atoms and four bins each.
    """
    return [
        NoisedView(
            r_noised=torch.full((10, 3), float(i)),
            noised_res_distogram=torch.full((2, 2, 4), float(i)),
            t_hat=torch.full((2,), float(i)),
            t_normalized=torch.full((2, 2), float(i)),
        )
        for i in range(2)
    ]


@pytest.fixture
def view_batch(views: list[NoisedView]) -> NoisedViewBatch:
    """``views`` stacked and moved to CPU.

    Args:
        views: Per-protein views.

    Returns:
        The stacked batch after ``to("cpu")``.
    """
    return NoisedViewBatch.stack(views).to("cpu", non_blocking=False)


def test_noised_view_batch_stacks_views_in_order(
    view_batch: NoisedViewBatch,
    views: list[NoisedView],
) -> None:
    """Stacking keeps each view's values at its batch index.

    No other cases: two views with distinct fill values show both the new
    batch axis and the order on every field.

    Args:
        view_batch: The stacked views.
        views: The per-protein views that were stacked; view ``i`` holds
            the value ``i`` everywhere.
    """
    fields: list[Float[torch.Tensor, "B ..."]] = [
        view_batch.r_noised,
        view_batch.noised_res_distogram,
        view_batch.t_hat,
        view_batch.t_normalized,
    ]
    for tensor in fields:
        assert tensor.shape[0] == len(views)
        for index in range(len(views)):
            assert bool((tensor[index] == float(index)).all())


# ---------------------------------------------------------------------------
# SelfFlowParams
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class InvalidParamsCase:
    """Self-Flow settings outside their allowed range.

    Attributes:
        mask_ratio: Per-residue masking probability.
        apply_prob: Probability a protein is masked at all.
    """

    mask_ratio: float
    apply_prob: float


class InvalidSelfFlowParams(enum.Enum):
    """Settings ``SelfFlowParams`` must reject.

    Members:
        ratio_above_half: The paper keeps ``R_M`` at or below 0.5.
        negative_ratio: Probabilities cannot be negative.
        apply_above_one: Probabilities cannot exceed 1.
    """

    ratio_above_half = InvalidParamsCase(mask_ratio=0.6, apply_prob=1.0)
    negative_ratio = InvalidParamsCase(mask_ratio=-0.1, apply_prob=1.0)
    apply_above_one = InvalidParamsCase(mask_ratio=0.25, apply_prob=1.5)


@pytest.mark.parametrize(
    "invalid_self_flow_params",
    list(InvalidSelfFlowParams),
    ids=[case.name for case in InvalidSelfFlowParams],
)
def test_self_flow_params_reject_out_of_range_settings(
    invalid_self_flow_params: InvalidSelfFlowParams,
) -> None:
    """Out-of-range Self-Flow settings fail when the config is loaded.

    Arrange: an invalid masking setting.
    Act: construct ``SelfFlowParams``.
    Assert: pydantic raises ``ValidationError``.

    Args:
        invalid_self_flow_params: The invalid settings; parametrized.
    """
    case = invalid_self_flow_params.value
    with pytest.raises(ValidationError):
        _ = SelfFlowParams(
            mask_ratio=case.mask_ratio,
            apply_prob=case.apply_prob,
        )


@pytest.fixture
def default_mask(
    default_self_flow_params: SelfFlowParams,
) -> Bool[torch.Tensor, "N_res"]:
    """One seeded mask draw over 8 residues with the default settings.

    Args:
        default_self_flow_params: Default Self-Flow settings.

    Returns:
        The drawn mask.
    """
    _ = manual_seed(0)
    return sample_residue_mask(
        8,
        default_self_flow_params.mask_ratio,
        default_self_flow_params.apply_prob,
    )


@pytest.fixture
def default_self_flow_params() -> SelfFlowParams:
    """Self-Flow settings with every field at its default.

    Returns:
        ``SelfFlowParams()``.
    """
    return SelfFlowParams()


def test_self_flow_is_off_by_default(
    default_self_flow_params: SelfFlowParams,
    default_mask: Bool[torch.Tensor, "N_res"],
) -> None:
    """Self-Flow stays off unless a config turns it on.

    No other cases: the defaults are one fixed configuration. The default
    mask settings must still be usable once it is enabled.

    Args:
        default_self_flow_params: Default Self-Flow settings.
        default_mask: A mask drawn with the default settings.
    """
    assert not default_self_flow_params.enabled
    assert default_mask.shape == (8,)

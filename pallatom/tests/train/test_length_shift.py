"""Tests for the length-dependent noise shift (``length_noise_shift``)."""

import dataclasses
import enum
import math

import pytest
from pydantic import ValidationError
from train.train_config import NoiseScheduleParams, length_noise_shift


@dataclasses.dataclass(frozen=True)
class ShiftCase:
    """A protein length, shift settings and the multiplier they give.

    Attributes:
        n_res: Protein length.
        beta: ``length_shift_beta``.
        ref: ``length_shift_ref``.
        expected: Expected multiplier on sigma.
    """

    n_res: int
    beta: float
    ref: int
    expected: float


class LengthShift(enum.Enum):
    """Hand-worked length shifts ``(n_res / ref) ** beta``.

    Members:
        off: beta 0 gives exactly 1 at any length.
        longer: 4x the reference length at beta 0.5 doubles sigma.
        shorter: A quarter of the reference at beta 0.5 halves it.
        at_reference: The reference length gives 1 at any beta.
        quarter_power: 16x the reference at beta 0.25 doubles sigma.
    """

    off = ShiftCase(n_res=300, beta=0.0, ref=128, expected=1.0)
    longer = ShiftCase(n_res=512, beta=0.5, ref=128, expected=2.0)
    shorter = ShiftCase(n_res=32, beta=0.5, ref=128, expected=0.5)
    at_reference = ShiftCase(n_res=128, beta=0.5, ref=128, expected=1.0)
    quarter_power = ShiftCase(n_res=2048, beta=0.25, ref=128, expected=2.0)


@pytest.mark.parametrize(
    "length_shift_scenario",
    list(LengthShift),
    ids=[case.name for case in LengthShift],
)
def test_length_noise_shift_scales_with_relative_length(
    length_shift_scenario: LengthShift,
) -> None:
    """The multiplier is ``(n_res / ref) ** beta``.

    Arrange: a length and shift settings.
    Act: ``length_noise_shift``.
    Assert: the multiplier matches the hand-worked value; beta 0, the
    default, is exactly 1, so training and sampling are unchanged.

    Args:
        length_shift_scenario: Inputs and expected multiplier; parametrized.
    """
    case = length_shift_scenario.value
    noise = NoiseScheduleParams(
        length_shift_beta=case.beta,
        length_shift_ref=case.ref,
    )
    assert math.isclose(length_noise_shift(case.n_res, noise), case.expected)


@dataclasses.dataclass(frozen=True)
class BadShiftCase:
    """Length-shift settings ``NoiseScheduleParams`` must reject.

    Attributes:
        beta: ``length_shift_beta``.
        ref: ``length_shift_ref``.
    """

    beta: float
    ref: int


class InvalidLengthShift(enum.Enum):
    """Out-of-range length-shift settings.

    Members:
        negative_beta: A negative exponent would denoise longer proteins.
        beta_above_one: Above 1 is far past the 0.5 derived for images.
        zero_reference: The reference length must be positive.
    """

    negative_beta = BadShiftCase(beta=-0.1, ref=128)
    beta_above_one = BadShiftCase(beta=1.5, ref=128)
    zero_reference = BadShiftCase(beta=0.5, ref=0)


@pytest.mark.parametrize(
    "invalid_length_shift",
    list(InvalidLengthShift),
    ids=[case.name for case in InvalidLengthShift],
)
def test_noise_schedule_rejects_invalid_length_shift(
    invalid_length_shift: InvalidLengthShift,
) -> None:
    """Out-of-range length-shift settings fail when the config is loaded.

    Arrange: invalid settings.
    Act: construct ``NoiseScheduleParams``.
    Assert: pydantic raises ``ValidationError``.

    Args:
        invalid_length_shift: The invalid settings; parametrized.
    """
    case = invalid_length_shift.value
    with pytest.raises(ValidationError):
        _ = NoiseScheduleParams(
            length_shift_beta=case.beta,
            length_shift_ref=case.ref,
        )

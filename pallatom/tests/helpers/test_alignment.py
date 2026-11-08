"""Tests for structure alignment helpers.

Covers Kabsch superposition of exact rigid copies (single and batched),
its refusal to superimpose a mirror image, weighting and masking in
``kabsch_rmsd`` and ``rmsd`` against hand-computed values, ``masked_com``
with and without padding, the rotation-plus-translation contract and
randomness of ``centre_random_augment``, and every function's shape
contract.

Coordinates come from the shared ``make_pattern`` fixture and rotations
are exact integer matrices, so every expected value is hand-checkable.
"""

import dataclasses
import enum
import functools
from collections.abc import Callable
from typing import cast

import pytest
import torch
from einops import einsum, rearrange, reduce, repeat
from helpers.alignment import (
    apply_transform,
    centre_random_augment,
    kabsch_align,
    kabsch_rmsd,
    kabsch_rotation,
    masked_com,
    rmsd,
)
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, TypeCheckError
from tests.conftest import CloudFactory

type Coords = Float[torch.Tensor, "N 3"]
type BatchCoords = Float[torch.Tensor, "B N 3"]
type Keep = Bool[torch.Tensor, "N"]
type Scalar = Float[torch.Tensor, ""]


@dataclasses.dataclass(frozen=True)
class RigidCase:
    """A rigid motion ``x -> x R + t``, with ``R`` a proper rotation.

    Attributes:
        rotation: Integer rotation matrix ``R``, row by row.
        translation: Offset ``t``.
    """

    rotation: tuple[tuple[int, int, int], ...]
    translation: tuple[float, float, float]


class RigidScenario(enum.Enum):
    """Rigid motions Kabsch must undo exactly.

    Members:
        identity: No motion; aligning a structure to itself.
        translate: Pure shift by ``(3, -2, 5)``.
        rotate_z: Quarter turn about the z axis.
        screw: Third of a turn about ``(1, 1, 1)`` plus the shift.
    """

    identity = RigidCase(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))
    translate = RigidCase(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (3, -2, 5))
    rotate_z = RigidCase(((0, -1, 0), (1, 0, 0), (0, 0, 1)), (0, 0, 0))
    screw = RigidCase(((0, 0, 1), (1, 0, 0), (0, 1, 0)), (3, -2, 5))


@dataclasses.dataclass(frozen=True)
class SelectCase:
    """A ``kabsch_rmsd`` call that restricts which atoms count.

    Attributes:
        call: ``call(mobile, target, keep)`` with ``keep`` True for atoms
            that count.
    """

    call: Callable[[Coords, Coords, Keep], Scalar]


class SelectScenario(enum.Enum):
    """The ways ``kabsch_rmsd`` can be told to ignore atoms.

    Members:
        weights: Zero float weights on ignored atoms.
        mask: A boolean mask.
        both: Weights and mask together, combined by the function.
    """

    weights = SelectCase(lambda m, t, k: kabsch_rmsd(m, t, weights=k.float()))
    mask = SelectCase(lambda m, t, k: kabsch_rmsd(m, t, mask=k))
    both = SelectCase(lambda m, t, k: kabsch_rmsd(m, t, k.float(), k))


@dataclasses.dataclass(frozen=True)
class RmsdCase:
    """An ``rmsd`` call and its value on the shifted test structure.

    Attributes:
        call: ``call(predicted, reference, keep)``.
        expected: RMSD in angstroms.
    """

    call: Callable[[Coords, Coords, Keep], Scalar]
    expected: float


class RmsdScenario(enum.Enum):
    """``rmsd`` over 9 atoms shifted by ``(0, 3, 4)`` and 3 by ``(0, 0, 11)``.

    Members:
        unweighted: ``sqrt((9 * 25 + 3 * 121) / 12) = sqrt(49) = 7``.
        mask: Only the 9 kept atoms count, each 5 away.
        weights: Zero weights on the 3 dropped atoms, same as the mask.
    """

    unweighted = RmsdCase(lambda p, r, _: rmsd(p, r), 7.0)
    mask = RmsdCase(lambda p, r, k: rmsd(p, r, mask=k), 5.0)
    weights = RmsdCase(lambda p, r, k: rmsd(p, r, weights=k.float()), 5.0)


@dataclasses.dataclass(frozen=True)
class AugmentResult:
    """One ``centre_random_augment`` call, split into rotation and shift.

    Attributes:
        out: The augmented coordinates.
        rotation: Matrix ``Q`` solving ``centred @ Q = out - mean(out)``
            by least squares, per batch item.
        shift: ``mean(out)`` per batch item, the added translation.
    """

    out: BatchCoords
    rotation: Float[torch.Tensor, "B 3 3"]
    shift: Float[torch.Tensor, "B 3"]


@pytest.fixture
def cloud(make_cloud: CloudFactory) -> Coords:
    """Return a 12-atom chiral cloud spanning about 10 angstroms.

    Args:
        make_cloud: Point-cloud factory.

    Returns:
        ``(12, 3)`` coordinates.
    """
    return 5 * make_cloud(1, 12)[0]


@pytest.fixture
def keep(cloud: Coords) -> Keep:
    """Return a mask keeping the first 9 of the cloud's 12 atoms.

    Args:
        cloud: Supplies the atom count.

    Returns:
        Boolean ``(12,)`` mask.
    """
    n = cloud.shape[0]
    return torch.arange(n) < n - 3


@pytest.fixture
def move() -> Callable[[Coords, RigidCase], Coords]:
    """Return a function applying a ``RigidCase`` to coordinates.

    Returns:
        ``move(x, case) = x R + t``.
    """

    def apply(x: Coords, case: RigidCase) -> Coords:
        """Rotate then translate ``x``.

        Args:
            x: Coordinates to move.
            case: Motion to apply.

        Returns:
            Moved coordinates.
        """
        rotation = torch.tensor(case.rotation, dtype=torch.float32)
        moved = einsum(x, rotation, "n d, d e -> n e")
        return moved + torch.tensor(case.translation)

    return apply


@pytest.fixture
def rigid_scenario(request: pytest.FixtureRequest) -> RigidScenario:
    """Return the ``RigidScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(RigidScenario, request.param)


@pytest.fixture
def select_scenario(request: pytest.FixtureRequest) -> SelectScenario:
    """Return the ``SelectScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(SelectScenario, request.param)


@pytest.fixture
def rmsd_scenario(request: pytest.FixtureRequest) -> RmsdScenario:
    """Return the ``RmsdScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(RmsdScenario, request.param)


@pytest.fixture
def com_coords() -> BatchCoords:
    """Return four real atoms followed by two zero-padded ones.

    The real atoms are the origin and points 2, 4 and 6 along the axes, so
    their centre of mass is ``(0.5, 1, 1.5)``.

    Returns:
        ``(1, 6, 3)`` coordinates.
    """
    real = torch.tensor([[0.0, 0, 0], [2, 0, 0], [0, 4, 0], [0, 0, 6]])
    return rearrange(torch.cat([real, torch.zeros(2, 3)]), "n d -> 1 n d")


@pytest.fixture
def aug_coords(make_cloud: CloudFactory) -> BatchCoords:
    """Return four 12-atom structures far from the origin.

    The clouds are scaled by 10 and offset by 20 so that skipping the
    centring step would leave a large, visible offset.

    Args:
        make_cloud: Point-cloud factory.

    Returns:
        ``(4, 12, 3)`` coordinates.
    """
    return 10 * make_cloud(4, 12) + 20


@pytest.fixture
def centred(aug_coords: BatchCoords) -> BatchCoords:
    """Return ``aug_coords`` with each structure's centroid subtracted.

    Args:
        aug_coords: Structures to centre.

    Returns:
        Centred coordinates.
    """
    return aug_coords - reduce(aug_coords, "b n d -> b 1 d", "mean")


@pytest.fixture
def padding(aug_coords: BatchCoords) -> Bool[torch.Tensor, "B N"]:
    """Return a mask marking the last 4 of every 12 atoms as padding.

    Args:
        aug_coords: Supplies the batch and atom counts.

    Returns:
        Boolean ``(B, N)`` mask, True for real atoms.
    """
    b, n, _ = aug_coords.shape
    return repeat(torch.arange(n) < n - 4, "n -> b n", b=b)


@pytest.fixture
def augmented(
    aug_coords: BatchCoords,
    centred: BatchCoords,
) -> AugmentResult:
    """Augment once under a fixed seed and recover the rotation and shift.

    The rotation comes from the normal equations
    ``Q = (Cᵀ C)⁻¹ Cᵀ O`` for centred input ``C`` and centred output
    ``O``, so it is whatever the augmentation applied, reflection or not.

    Args:
        aug_coords: Input structures.
        centred: The same structures, centred.

    Returns:
        The output and its decomposition.
    """
    _ = manual_seed(0)
    out = centre_random_augment(aug_coords)
    shift = reduce(out, "b n d -> b d", "mean")
    out_centred = out - rearrange(shift, "b d -> b 1 d")
    gram = einsum(centred, centred, "b n i, b n j -> b i j")
    cross = einsum(centred, out_centred, "b n i, b n j -> b i j")
    rotation = einsum(torch.inverse(gram), cross, "b i k, b k j -> b i j")
    return AugmentResult(out=out, rotation=rotation, shift=shift)


@pytest.mark.parametrize(
    "rigid_scenario",
    list(RigidScenario),
    ids=[case.name for case in RigidScenario],
    indirect=True,
)
def test_kabsch_undoes_a_rigid_motion(
    rigid_scenario: RigidScenario,
    cloud: Coords,
    move: Callable[[Coords, RigidCase], Coords],
) -> None:
    """Kabsch maps a rigidly moved copy back onto the original.

    The RMSD after alignment is at float32 SVD noise, the aligned copy
    matches the target, and the returned rotation and centroids rebuild the
    target through ``apply_transform``.

    Args:
        rigid_scenario: Motion applied to the copy.
        cloud: Target structure.
        move: Applies the motion.
    """
    mobile = move(cloud, rigid_scenario.value)
    transform = kabsch_align(
        rearrange(mobile, "n d -> 1 n d"),
        rearrange(cloud, "n d -> 1 n d"),
        return_transform=True,
    )
    out, rot, c_m, c_t = transform  # pylint: disable=unpacking-non-sequence
    rebuilt = apply_transform(mobile, rot, c_m, c_t)
    value = kabsch_rmsd(mobile, cloud)
    torch.testing.assert_close(value, torch.zeros(()), atol=1e-3, rtol=0)
    torch.testing.assert_close(out[0], cloud, atol=1e-4, rtol=0)
    torch.testing.assert_close(rebuilt[0], cloud, atol=1e-4, rtol=0)


def test_kabsch_rmsd_aligns_each_batch_item_separately(
    cloud: Coords,
    move: Callable[[Coords, RigidCase], Coords],
) -> None:
    """A batch holding every rigid scenario's copy is near zero per item.

    Each item has a different motion, so one shared rotation could not fit
    them all. Not parametrized: the batch already holds every scenario.

    Args:
        cloud: Target structure.
        move: Applies each motion.
    """
    mobile = torch.stack([move(cloud, case.value) for case in RigidScenario])
    target = repeat(cloud, "n d -> b n d", b=len(mobile))
    rmsds = kabsch_rmsd(mobile, target)
    zeros = torch.zeros(len(mobile))
    torch.testing.assert_close(rmsds, zeros, atol=1e-3, rtol=0)


def test_kabsch_does_not_superimpose_a_mirror_image(cloud: Coords) -> None:
    """A reflected copy of a chiral cloud keeps a clearly nonzero RMSD.

    Kabsch must return a proper rotation (det +1); allowing a reflection
    would map the mirror image exactly. Not parametrized: any reflection of
    a chiral cloud is equally unreachable.

    Args:
        cloud: Chiral target structure.
    """
    mirrored = cloud * torch.tensor([-1.0, 1.0, 1.0])
    rotation = kabsch_rotation(mirrored, cloud)
    det = torch.det(rotation)
    torch.testing.assert_close(det, torch.ones(()), atol=1e-5, rtol=0)
    assert kabsch_rmsd(mirrored, cloud).item() > 1


@pytest.mark.parametrize(
    "select_scenario",
    list(SelectScenario),
    ids=[case.name for case in SelectScenario],
    indirect=True,
)
def test_kabsch_rmsd_ignores_deselected_atoms(
    select_scenario: SelectScenario,
    cloud: Coords,
    keep: Keep,
    move: Callable[[Coords, RigidCase], Coords],
) -> None:
    """Displaced atoms that are deselected do not spoil the superposition.

    The copy is moved rigidly, then its 3 deselected atoms are pushed 10
    angstroms further. Unweighted Kabsch cannot absorb that; restricted to
    the 9 kept atoms it must be exact again.

    Args:
        select_scenario: How the atoms are deselected.
        cloud: Target structure.
        keep: True for the 9 atoms that count.
        move: Applies the screw motion.
    """
    mobile = move(cloud, RigidScenario.screw.value)
    mobile[~keep] += 10
    value = select_scenario.value.call(mobile, cloud, keep)
    assert kabsch_rmsd(mobile, cloud).item() > 1
    torch.testing.assert_close(value, torch.zeros(()), atol=1e-3, rtol=0)


@pytest.mark.parametrize(
    "rmsd_scenario",
    list(RmsdScenario),
    ids=[case.name for case in RmsdScenario],
    indirect=True,
)
def test_rmsd_matches_hand_computed_value(
    rmsd_scenario: RmsdScenario,
    cloud: Coords,
    keep: Keep,
) -> None:
    """``rmsd`` without alignment equals the RMS of the per-atom shifts.

    Args:
        rmsd_scenario: Call and expected value.
        cloud: Reference structure.
        keep: True for the 9 atoms shifted by ``(0, 3, 4)``.
    """
    shifted = cloud + torch.where(
        rearrange(keep, "n -> n 1"),
        torch.tensor([0.0, 3.0, 4.0]),
        torch.tensor([0.0, 0.0, 11.0]),
    )
    value = rmsd_scenario.value.call(shifted, cloud, keep)
    torch.testing.assert_close(
        value,
        torch.tensor(rmsd_scenario.value.expected),
    )


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(kabsch_rotation, id="kabsch_rotation"),
        pytest.param(
            functools.partial(kabsch_align, return_transform=False),
            id="kabsch_align",
        ),
        pytest.param(rmsd, id="rmsd"),
        pytest.param(kabsch_rmsd, id="kabsch_rmsd"),
    ],
)
def test_alignment_functions_reject_four_coordinate_atoms(
    cloud: Coords,
    function: Callable[[Float[torch.Tensor, "N 4"], Coords], object],
) -> None:
    """Each function's contract rejects coordinates of shape ``(N, 4)``.

    Args:
        cloud: Valid second argument.
        function: Function under test, called with the bad coordinates
            first.
    """
    bad = torch.cat([cloud, cloud[:, :1]], dim=-1)
    with pytest.raises(TypeCheckError):
        _ = function(bad, cloud)


def test_apply_transform_rejects_four_coordinate_atoms(cloud: Coords) -> None:
    """``apply_transform`` rejects coordinates of shape ``(N, 4)``.

    Not parametrized: the rotation and centroids are valid, so only the
    coordinates' shape is under test.

    Args:
        cloud: Supplies valid centroids and the coordinates to extend.
    """
    bad = torch.cat([cloud, cloud[:, :1]], dim=-1)
    with pytest.raises(TypeCheckError):
        _ = apply_transform(bad, torch.eye(3), cloud[:1], cloud[:1])


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        pytest.param(None, (1 / 3, 2 / 3, 1.0), id="no_mask"),
        pytest.param((1, 1, 1, 1, 1, 1), (1 / 3, 2 / 3, 1.0), id="all_valid"),
        pytest.param((1, 1, 1, 1, 0, 0), (0.5, 1.0, 1.5), id="padding_masked"),
    ],
)
def test_masked_com_averages_only_valid_atoms(
    com_coords: BatchCoords,
    mask: tuple[int, ...] | None,
    expected: tuple[float, float, float],
) -> None:
    """The centre of mass counts exactly the atoms the mask keeps.

    Without a mask, or with an all-true one, the two padding zeros pull the
    mean toward the origin; masking them recovers the real atoms' centre.

    Args:
        com_coords: Four real atoms then two zero-padded ones.
        mask: Per-atom validity, or None for no mask.
        expected: Centre of mass.
    """
    bool_mask = None if mask is None else torch.tensor([mask], dtype=torch.bool)
    out = masked_com(com_coords, mask=bool_mask)
    torch.testing.assert_close(out, torch.tensor([[expected]]))


@pytest.mark.parametrize(
    "masked",
    [False, True],
    ids=["no_mask", "padding_mask"],
)
def test_centre_random_augment_centres_before_rotating(
    aug_coords: BatchCoords,
    padding: Bool[torch.Tensor, "B N"],
    masked: bool,  # noqa: FBT001
) -> None:
    """With ``s_trans=0``, the counted atoms' centroid ends at the origin.

    A rotation about the origin keeps a centred cloud centred, so this
    holds only if the centroid was removed first, from the right atoms. The
    padded atoms are zeroed, as featurisation pads them, so centring on all
    atoms would pull the centroid off the valid ones.

    Args:
        aug_coords: Structures far from the origin.
        padding: True for the 8 real atoms of each 12.
        masked: Whether to pad and pass the mask.
    """
    mask = padding if masked else torch.ones_like(padding)
    coords = aug_coords * rearrange(mask.float(), "b n -> b n 1")
    out = centre_random_augment(coords, s_trans=0.0, mask=mask)
    weights = rearrange(mask.float(), "b n -> b n 1")
    centroid = reduce(out * weights, "b n d -> b d", "sum") / reduce(
        weights,
        "b n d -> b d",
        "sum",
    )
    assert out.shape == coords.shape
    torch.testing.assert_close(
        centroid,
        torch.zeros_like(centroid),
        atol=1e-4,
        rtol=0,
    )


def test_centre_random_augment_mask_changes_padded_output(
    aug_coords: BatchCoords,
    padding: Bool[torch.Tensor, "B N"],
) -> None:
    """On zero-padded input, the mask changes the valid atoms' output.

    Centring on every atom includes the padding zeros, which shifts the
    centroid; under the same seed the masked and unmasked outputs must
    therefore differ on real atoms. Not parametrized: one padding layout
    is enough to show the mask is used.

    Args:
        aug_coords: Structures far from the origin.
        padding: True for the 8 real atoms of each 12.
    """
    coords = aug_coords * rearrange(padding.float(), "b n -> b n 1")
    _ = manual_seed(0)
    masked = centre_random_augment(coords, s_trans=0.0, mask=padding)
    _ = manual_seed(0)
    unmasked = centre_random_augment(coords, s_trans=0.0)
    assert not torch.allclose(masked[padding], unmasked[padding])


def test_centre_random_augment_applies_a_proper_rotation(
    centred: BatchCoords,
    augmented: AugmentResult,
) -> None:
    """The output is ``centred @ Q + t`` with ``Q`` orthogonal and det +1.

    Reconstruction from the recovered ``Q`` shows the map is linear plus a
    shift; ``QᵀQ = I`` rules out scaling and shear, which would change
    inner products; ``det Q = +1`` rules out a reflection. Not
    parametrized: every batch item is checked.

    Args:
        centred: Centred input structures.
        augmented: Output and recovered rotation and shift.
    """
    q = augmented.rotation
    rebuilt = einsum(centred, q, "b n i, b i j -> b n j") + rearrange(
        augmented.shift,
        "b d -> b 1 d",
    )
    gram = einsum(q, q, "b k i, b k j -> b i j")
    identity = repeat(torch.eye(3), "i j -> b i j", b=len(q))
    torch.testing.assert_close(rebuilt, augmented.out, atol=1e-3, rtol=0)
    torch.testing.assert_close(gram, identity, atol=1e-4, rtol=0)
    torch.testing.assert_close(
        torch.det(q),
        torch.ones(len(q)),
        atol=1e-4,
        rtol=0,
    )


def test_centre_random_augment_draws_per_batch_item(
    augmented: AugmentResult,
) -> None:
    """Every batch item gets its own rotation and its own translation.

    Two Haar-random rotations or two Gaussian shifts coincide with
    probability zero. Not parametrized: all pairs of items are compared.

    Args:
        augmented: Output and recovered rotation and shift.
    """
    q = rearrange(augmented.rotation, "b i j -> b (i j)")
    assert torch.cdist(q, q).fill_diagonal_(1).min() > 0
    shift = augmented.shift
    assert torch.cdist(shift, shift).fill_diagonal_(1).min() > 0


@pytest.mark.parametrize(
    ("s_trans", "factor"),
    [
        pytest.param(None, 1.0, id="default_is_one_angstrom"),
        pytest.param(2.0, 2.0, id="doubled"),
        pytest.param(0.5, 0.5, id="halved"),
    ],
)
def test_centre_random_augment_translation_scales_with_s_trans(
    aug_coords: BatchCoords,
    s_trans: float | None,
    factor: float,
) -> None:
    """Under the same seed, the output centroid is ``s_trans`` times unit's.

    The translation is ``s_trans * N(0, I)``, so for a fixed draw the
    centroid scales linearly; omitting ``s_trans`` must match ``1.0``,
    AlphaFold 3's Algorithm 19 default.

    Args:
        aug_coords: Input structures.
        s_trans: Value passed, or None to use the default.
        factor: Expected centroid ratio to ``s_trans=1``.
    """
    _ = manual_seed(0)
    unit = reduce(
        centre_random_augment(aug_coords, 1.0),
        "b n d -> b d",
        "mean",
    )
    _ = manual_seed(0)
    out = (
        centre_random_augment(aug_coords)
        if s_trans is None
        else centre_random_augment(aug_coords, s_trans)
    )
    centroid = reduce(out, "b n d -> b d", "mean")
    torch.testing.assert_close(centroid, factor * unit, atol=1e-5, rtol=0)


def test_centre_random_augment_ignores_input_position(
    aug_coords: BatchCoords,
) -> None:
    """Shifting the input changes nothing: centring removes the shift.

    Not parametrized: any shift is removed by the same subtraction.

    Args:
        aug_coords: Input structures.
    """
    _ = manual_seed(0)
    base = centre_random_augment(aug_coords)
    _ = manual_seed(0)
    shifted = centre_random_augment(aug_coords + torch.tensor([5.0, -3, 2]))
    torch.testing.assert_close(base, shifted, atol=1e-4, rtol=0)


def test_centre_random_augment_single_atom_maps_to_origin(
    aug_coords: BatchCoords,
) -> None:
    """A one-atom structure centres to the origin, which rotation keeps.

    With ``s_trans=0`` the output is exactly zero. Not parametrized: one
    atom is the only degenerate cloud.

    Args:
        aug_coords: Supplies one atom per batch item.
    """
    out = centre_random_augment(aug_coords[:, :1], s_trans=0.0)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-6, rtol=0)


def test_centre_random_augment_is_seeded_and_fresh_per_call(
    aug_coords: BatchCoords,
) -> None:
    """The same seed repeats the draw; consecutive calls draw anew.

    Not parametrized: the two properties are two sides of one RNG contract.

    Args:
        aug_coords: Input structures.
    """
    _ = manual_seed(0)
    first = centre_random_augment(aug_coords)
    second = centre_random_augment(aug_coords)
    _ = manual_seed(0)
    repeated = centre_random_augment(aug_coords)
    assert torch.equal(first, repeated)
    assert not torch.allclose(first, second)


@pytest.mark.parametrize(
    "function",
    [
        pytest.param(masked_com, id="masked_com"),
        pytest.param(centre_random_augment, id="centre_random_augment"),
    ],
)
def test_batched_functions_reject_mismatched_mask(
    aug_coords: BatchCoords,
    padding: Bool[torch.Tensor, "B N"],
    function: Callable[..., object],
) -> None:
    """A mask with one atom more than the coordinates fails the contract.

    Args:
        aug_coords: Valid coordinates.
        padding: Valid mask to extend.
        function: Function under test, called with ``mask=``.
    """
    longer = torch.cat([padding, padding[:, :1]], dim=1)
    with pytest.raises(TypeCheckError):
        _ = function(aug_coords, mask=longer)


def test_centre_random_augment_rejects_four_coordinate_atoms(
    aug_coords: BatchCoords,
) -> None:
    """Coordinates of shape ``(B, N, 4)`` fail the contract.

    Not parametrized: the mask is covered by the test above.

    Args:
        aug_coords: Valid coordinates to extend.
    """
    bad = torch.cat([aug_coords, aug_coords[..., :1]], dim=-1)
    with pytest.raises(TypeCheckError):
        _ = centre_random_augment(bad)

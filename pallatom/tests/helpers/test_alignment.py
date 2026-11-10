"""Tests for structure alignment helpers.

Covers kabsch_rotation, kabsch_align, kabsch_rmsd, rmsd, apply_transform,
and centre_random_augment — including correctness, shape-contract enforcement,
and numerical tolerance checks.
"""

import math

import pytest
import torch
from architecture.losses import atom_loss
from einops import einsum, rearrange, reduce
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
from hypothesis import given
from hypothesis import strategies as st
from jaxtyping import Bool, Float, Int, TypeCheckError
from tests.strategies import (
    CloudPair,
    MaskedCloudPair,
    ShuffledCloudPair,
    batched_clouds,
    cloud_pairs,
    masked_cloud_pairs,
    point_clouds,
    rotate,
    rotations,
    shuffled_cloud_pairs,
    translations,
)

_ = manual_seed(42)
N, B = 50, 8
TOLERANCE = 1e-5
TIGHT_TOLERANCE = 1e-3
GEOMETRY_TOLERANCE = 1e-6
MIN_CHIRAL_ATOMS = 4
MAX_SEED = 2**31 - 1
SEEDS = st.integers(0, MAX_SEED)
SCALES = st.floats(0.1, 10.0, width=64)


@pytest.fixture
def uniform_aa_indices() -> Int[torch.Tensor, "B N"]:
    """Amino-acid indices of a single nonpolar residue type ('A' = 0).

    Used by atom_loss tests that don't care about residue-type weighting;
    a uniform residue type makes that weighting a constant factor.

    Returns:
        Integer tensor of shape (B, N), all entries 0.
    """
    return torch.zeros(B, N, dtype=torch.long)


def test_atom_loss_perfect_prediction_near_zero(
    uniform_aa_indices: Int[torch.Tensor, "B N"],
) -> None:
    """atom_loss is near zero when prediction equals ground truth.

    Verifies that atom_loss returns a negligible value when the predicted
    coordinates are identical to the target coordinates.

    Args:
        uniform_aa_indices: Amino-acid indices, shape (B, N), all the same
            residue type so the residue-type weighting has no effect.
    """
    r = torch.randn(B, N, 3)
    assert (
        atom_loss(
            r,
            r,
            aa_indices=uniform_aa_indices,
            lambda_sigma_weight=torch.ones(B),
        )
        .mean()
        .item()
        < TOLERANCE
    )


def test_atom_loss_noisy_prediction_positive_and_finite(
    uniform_aa_indices: Int[torch.Tensor, "B N"],
) -> None:
    """atom_loss is positive and finite for a noisy prediction.

    Verifies that atom_loss returns strictly positive, finite values when
    the predicted coordinates differ from the target by Gaussian noise.

    Args:
        uniform_aa_indices: Amino-acid indices, shape (B, N), all the same
            residue type so the residue-type weighting has no effect.
    """
    r = torch.randn(B, N, 3)
    r_noisy = r + 0.5 * torch.randn(B, N, 3)
    loss = atom_loss(
        r,
        r_noisy,
        aa_indices=uniform_aa_indices,
        lambda_sigma_weight=torch.ones(B),
    )
    assert (loss > 0).all()
    assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_kabsch_rotation_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError.

    Verifies that kabsch_rotation enforces the shape contract and raises
    TypeCheckError when coordinates have an unexpected last dimension.
    """
    mobile_bad = torch.zeros(N, 4)  # last dim must be 3
    reference = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        _ = kabsch_rotation(mobile_bad, reference)


def test_kabsch_align_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError.

    Verifies that kabsch_align enforces the shape contract and raises
    TypeCheckError when the mobile tensor has an unexpected last dimension.
    """
    mobile_bad = torch.zeros(N, 4)  # last dim must be 3
    target = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        _ = kabsch_align(mobile_bad, target, return_transform=False)


def test_rmsd_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError.

    Verifies that rmsd enforces the shape contract and raises TypeCheckError
    when the predicted tensor has an unexpected last dimension.
    """
    predicted_bad = torch.zeros(N, 4)  # last dim must be 3
    reference = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        _ = rmsd(predicted_bad, reference)


def test_kabsch_rmsd_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError.

    Verifies that kabsch_rmsd enforces the shape contract and raises
    TypeCheckError when the mobile tensor has an unexpected last dimension.
    """
    mobile_bad = torch.zeros(N, 4)  # last dim must be 3
    target = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        _ = kabsch_rmsd(mobile_bad, target)


def test_apply_transform_wrong_shape() -> None:
    """Wrong coords last dim (4 instead of 3) triggers TypeCheckError.

    Verifies that apply_transform enforces the shape contract on the
    coordinates argument and raises TypeCheckError for an unexpected last dim.
    """
    coords_bad = torch.zeros(N, 4)  # last dim must be 3
    R = rearrange(torch.eye(3), "r c -> 1 r c")
    t = torch.zeros(1, 1, 3)
    with pytest.raises(TypeCheckError):
        _ = apply_transform(coords_bad, R, t, t)


# ---------------------------------------------------------------------------
# masked_com
# ---------------------------------------------------------------------------


def test_masked_com_wrong_shape_raises_type_error(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """A mask with a mismatched atom dimension raises TypeCheckError."""
    bad_mask: Bool[torch.Tensor, "B N_plus_one"] = torch.ones(
        B,
        N + 1,
        dtype=torch.bool,
    )
    with pytest.raises(TypeCheckError):
        _ = masked_com(aug_coords, mask=bad_mask)


# ---------------------------------------------------------------------------
# centre_random_augment
# ---------------------------------------------------------------------------


@pytest.fixture
def aug_coords() -> Float[torch.Tensor, "B N 3"]:
    """Batched random atom coords at a large scale to expose centring errors.

    The scale factor of 10.0 ensures that any failure to subtract the centroid
    before rotating would produce a large, detectable offset.
    """
    _ = manual_seed(7)
    return torch.randn(B, N, 3) * 10.0


def test_centre_random_augment_output_shape(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Output shape is identical to input shape.

    Verifies that centre_random_augment does not alter the batch, atom, or
    coordinate dimensions of the input tensor.
    """
    out = centre_random_augment(aug_coords)
    assert out.shape == aug_coords.shape


def test_centre_random_augment_default_s_trans_matches_one(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Omitting s_trans is equivalent to passing s_trans=1.0.

    Verifies the default matches AF3 Algorithm 19's s_trans = 1 Å default,
    for identical RNG state.
    """
    _ = manual_seed(11)
    out_default = centre_random_augment(aug_coords)
    _ = manual_seed(11)
    out_explicit = centre_random_augment(aug_coords, s_trans=1.0)
    assert torch.equal(out_default, out_explicit)


def test_centre_random_augment_invertible_via_kabsch(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Applying Kabsch-recovered rotation inverse restores centred input.

    centre_random_augment computes out = centred @ Q + t for some Q ∈ SO(3)
    and translation t. Re-centring out removes t, and kabsch_rotation on the
    re-centred output recovers exactly Q, so out_centred @ Q^T = centred.
    """
    centred: Float[torch.Tensor, "B N 3"] = aug_coords - reduce(
        aug_coords,
        "b n d -> b 1 d",
        "mean",
    )
    out: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    out_centred: Float[torch.Tensor, "B N 3"] = out - reduce(
        out,
        "b n d -> b 1 d",
        "mean",
    )
    # kabsch_rotation(mobile, reference) → R  s.t. mobile @ R^T ≈ reference
    Q: Float[torch.Tensor, "B 3 3"] = kabsch_rotation(out_centred, centred)
    recovered: Float[torch.Tensor, "B N 3"] = einsum(
        out_centred,
        Q.mH,
        "b n i, b i j -> b n j",
    )
    assert torch.allclose(recovered, centred, atol=1e-3)


def test_centre_random_augment_recovered_rotation_is_proper_so3(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Rotation Q applied by the augmentation has det = +1 and is orthogonal.

    Recovers Q directly via the normal equations on the re-centred output
    (which removes the added translation t):
        centred @ Q = out_centred
        Q = (centred^T centred)^{-1} centred^T out_centred

    This tests the rotation centre_random_augment applies, not the
    Kabsch-recovered rotation (which always has det = +1 by its own
    reflection correction).
    """
    centred: Float[torch.Tensor, "B N 3"] = aug_coords - reduce(
        aug_coords,
        "b n d -> b 1 d",
        "mean",
    )
    out: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    out_centred: Float[torch.Tensor, "B N 3"] = out - reduce(
        out,
        "b n d -> b 1 d",
        "mean",
    )
    # Normal equations: gram @ Q_actual = centred^T @ out_centred
    gram: Float[torch.Tensor, "B 3 3"] = einsum(
        centred,
        centred,
        "b n i, b n j -> b i j",
    )
    cT_out: Float[torch.Tensor, "B 3 3"] = einsum(
        centred,
        out_centred,
        "b n i, b n j -> b i j",
    )
    Q_actual: Float[torch.Tensor, "B 3 3"] = einsum(
        torch.inverse(gram),
        cT_out,
        "b i k, b k j -> b i j",
    )
    # Orthogonality: Q^T Q = I
    eye_approx: Float[torch.Tensor, "B 3 3"] = einsum(
        Q_actual.mH,
        Q_actual,
        "b i j, b j k -> b i k",
    )
    assert torch.allclose(eye_approx, torch.eye(3).expand(B, -1, -1), atol=1e-3)
    # Proper rotation: det = +1 (not -1 reflection)
    dets: Float[torch.Tensor, "B"] = torch.det(Q_actual)
    assert torch.allclose(dets, torch.ones(B), atol=1e-3)


def test_centre_random_augment_applies_independent_rotations_per_batch(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Different batch elements receive distinct SO(3) rotations.

    The probability that two Haar-uniform SO(3) samples are equal is zero;
    with float32 the empirical distance between any two rotation matrices
    should be well above numerical noise. Re-centring the output removes
    the per-batch translation noise before recovering the rotation.
    """
    centred: Float[torch.Tensor, "B N 3"] = aug_coords - reduce(
        aug_coords,
        "b n d -> b 1 d",
        "mean",
    )
    out: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    out_centred: Float[torch.Tensor, "B N 3"] = out - reduce(
        out,
        "b n d -> b 1 d",
        "mean",
    )
    Q: Float[torch.Tensor, "B 3 3"] = kabsch_rotation(out_centred, centred)
    # Every pair of rotation matrices in the batch should differ by > 1e-3
    for i in range(B - 1):
        diff = (Q[i] - Q[i + 1]).abs().max().item()
        assert (
            diff > TIGHT_TOLERANCE
        ), f"Batch elements {i} and {i+1} received the same rotation"


def test_centre_random_augment_applies_independent_translations(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Different batch elements receive distinct translation noise.

    The probability that two independent Gaussian translation draws are
    equal is zero, so every pair of per-batch output centroids should
    differ well above numerical noise.
    """
    out: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    centroid: Float[torch.Tensor, "B 3"] = reduce(out, "b n d -> b d", "mean")
    for i in range(B - 1):
        diff = (centroid[i] - centroid[i + 1]).abs().max().item()
        assert (
            diff > TIGHT_TOLERANCE
        ), f"Batch elements {i} and {i+1} received the same translation"


def test_centre_random_augment_is_deterministic_under_fixed_seed(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Two calls with the same RNG seed produce bit-identical results.

    Verifies that centre_random_augment is fully deterministic under a fixed
    PyTorch RNG state, with no hidden non-deterministic operations.
    """
    _ = manual_seed(55)
    out_a: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    _ = manual_seed(55)
    out_b: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    assert torch.equal(out_a, out_b)


def test_centre_random_augment_is_stochastic_across_calls(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Repeated calls without reseeding produce different outputs each time.

    Verifies that centre_random_augment draws fresh rotation and translation
    noise on every invocation rather than caching state. Five consecutive
    calls on identical input are compared pairwise; coordinates are float32,
    so any pair differing by more than TIGHT_TOLERANCE must reflect a
    genuinely different random draw rather than rounding noise (~1e-7).
    """
    outputs: list[Float[torch.Tensor, "B N 3"]] = [
        centre_random_augment(aug_coords) for _ in range(5)
    ]
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            diff = (outputs[i] - outputs[j]).abs().max().item()
            assert (
                diff > TIGHT_TOLERANCE
            ), f"Calls {i} and {j} produced the same coordinates"


def test_centre_random_augment_wrong_last_dim_raises_type_error() -> None:
    """Passing coords with last dim ≠ 3 triggers a jaxtyping TypeCheckError.

    Verifies that centre_random_augment enforces its shape contract and rejects
    inputs whose last dimension is not 3.
    """
    bad: Float[torch.Tensor, "B N 4"] = torch.randn(B, N, 4)
    with pytest.raises(TypeCheckError):
        _ = centre_random_augment(bad)


# ---------------------------------------------------------------------------
# centre_random_augment — optional mask
# ---------------------------------------------------------------------------


@pytest.fixture
def padding_mask() -> Bool[torch.Tensor, "B N"]:
    """Boolean mask marking the second half of atoms as padding.

    Used to test that centre_random_augment's centroid computation can
    ignore zero-padded atoms when a mask is supplied.
    """
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, N // 2 :] = False
    return mask


@pytest.fixture
def padded_coords(
    aug_coords: Float[torch.Tensor, "B N 3"],
    padding_mask: Bool[torch.Tensor, "B N"],
) -> Float[torch.Tensor, "B N 3"]:
    """aug_coords with every padding-mask position zeroed out.

    Mimics zero-padded atoms appended to a shorter real structure, as
    produced by featurize_single_item's F.pad calls.
    """
    return aug_coords * rearrange(padding_mask.float(), "b n -> b n 1")


def test_centre_random_augment_mask_changes_output_with_padding(
    padded_coords: Float[torch.Tensor, "B N 3"],
    padding_mask: Bool[torch.Tensor, "B N"],
) -> None:
    """Padding-aware centring gives a different result than the naive mean.

    When zero-padded atoms are present, centring on all atoms (mask=None)
    biases the centroid toward the origin relative to the true, valid-only
    centroid, so with identical RNG state the masked and unmasked outputs
    for the valid atoms must differ.
    """
    _ = manual_seed(31)
    out_masked: Float[torch.Tensor, "B N 3"] = centre_random_augment(
        padded_coords,
        s_trans=0.0,
        mask=padding_mask,
    )
    _ = manual_seed(31)
    out_unmasked: Float[torch.Tensor, "B N 3"] = centre_random_augment(
        padded_coords,
        s_trans=0.0,
    )
    weights: Float[torch.Tensor, "B N 1"] = rearrange(
        padding_mask.float(),
        "b n -> b n 1",
    )
    diff = ((out_masked - out_unmasked) * weights).abs().max().item()
    assert diff > TIGHT_TOLERANCE


def test_centre_random_augment_mask_wrong_shape_raises_type_error(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """A mask with a mismatched atom dimension raises TypeCheckError.

    Verifies that the optional mask argument is shape-checked against
    N_atom just like the coords argument.
    """
    bad_mask: Bool[torch.Tensor, "B N_plus_one"] = torch.ones(
        B,
        N + 1,
        dtype=torch.bool,
    )
    with pytest.raises(TypeCheckError):
        _ = centre_random_augment(aug_coords, mask=bad_mask)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def _centred(
    coords: Float[torch.Tensor, "... n 3"],
) -> Float[torch.Tensor, "... n 3"]:
    """Subtract each structure's mean atom position."""
    return coords - reduce(coords, "... n d -> ... 1 d", "mean")


def _chirality(
    coords: Float[torch.Tensor, "... n 3"],
) -> Float[torch.Tensor, "..."]:
    """Return the signed volume of the first four atoms.

    Rotations and translations preserve this value, while a reflection
    negates it.

    Args:
        coords: Coordinates with at least four atoms.

    Returns:
        The determinant of the three edges leaving the first atom.
    """
    return torch.det(coords[..., 1:4, :] - coords[..., :1, :])


@given(points=point_clouds(), rotation=rotations(), shift=translations())
def test_kabsch_align_undoes_rigid_motion(
    points: Float[torch.Tensor, "N 3"],
    rotation: Float[torch.Tensor, "3 3"],
    shift: Float[torch.Tensor, "1 3"],
) -> None:
    """Aligning a rigidly moved copy of a cloud restores the original."""
    moved = rotate(points, rotation) + shift
    (aligned,) = kabsch_align(  # pylint: disable=unpacking-non-sequence
        moved,
        points,
        return_transform=False,
    )
    assert torch.allclose(aligned, points, atol=GEOMETRY_TOLERANCE)


@given(pair=cloud_pairs())
def test_kabsch_rotation_is_proper_for_any_clouds(pair: CloudPair) -> None:
    """The result is orthogonal with det = +1, even for unrelated clouds."""
    R = kabsch_rotation(_centred(pair.mobile), _centred(pair.target))
    identity = einsum(R, R.mH, "i j, j k -> i k")
    expected = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(identity, expected, atol=GEOMETRY_TOLERANCE)
    assert math.isclose(torch.det(R).item(), 1.0, rel_tol=GEOMETRY_TOLERANCE)


@given(points=point_clouds(min_atoms=MIN_CHIRAL_ATOMS))
def test_kabsch_rotation_is_proper_for_mirror_images(
    points: Float[torch.Tensor, "N 3"],
) -> None:
    """A mirror image can't be matched by reflecting: det stays +1."""
    mirror = points * torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)
    R = kabsch_rotation(_centred(points), _centred(mirror))
    assert math.isclose(torch.det(R).item(), 1.0, rel_tol=GEOMETRY_TOLERANCE)


@given(pair=cloud_pairs(), rotation=rotations())
def test_kabsch_align_is_at_least_as_good_as_any_rotation(
    pair: CloudPair,
    rotation: Float[torch.Tensor, "3 3"],
) -> None:
    """No other rotation about the centroid gives a lower RMSD."""
    (aligned,) = kabsch_align(  # pylint: disable=unpacking-non-sequence
        pair.mobile,
        pair.target,
        return_transform=False,
    )
    centroid = reduce(pair.target, "n d -> 1 d", "mean")
    other = rotate(_centred(pair.mobile), rotation) + centroid
    best = rmsd(aligned, pair.target).item()
    assert best <= rmsd(other, pair.target).item() + GEOMETRY_TOLERANCE


@given(pair=cloud_pairs(), shift=translations())
def test_kabsch_align_maps_centroid_onto_target(
    pair: CloudPair,
    shift: Float[torch.Tensor, "1 3"],
) -> None:
    """The aligned centroid lands on the target, wherever the target is.

    Coordinates aren't compared directly: when one cloud is a single repeated
    point the best rotation isn't unique, so only the centroid and the RMSD
    are defined.
    """
    target = pair.target + shift
    (aligned,) = kabsch_align(  # pylint: disable=unpacking-non-sequence
        pair.mobile,
        target,
        return_transform=False,
    )
    assert torch.allclose(
        reduce(aligned, "n d -> d", "mean"),
        reduce(target, "n d -> d", "mean"),
        atol=GEOMETRY_TOLERANCE,
    )


@given(pair=cloud_pairs(), shift=translations())
def test_kabsch_rmsd_ignores_translation(
    pair: CloudPair,
    shift: Float[torch.Tensor, "1 3"],
) -> None:
    """Shifting either cloud leaves the best achievable RMSD unchanged."""
    base = kabsch_rmsd(pair.mobile, pair.target)
    moved_mobile = kabsch_rmsd(pair.mobile + shift, pair.target)
    moved_target = kabsch_rmsd(pair.mobile, pair.target + shift)
    assert torch.allclose(moved_mobile, base, atol=GEOMETRY_TOLERANCE)
    assert torch.allclose(moved_target, base, atol=GEOMETRY_TOLERANCE)


@given(shuffled=shuffled_cloud_pairs())
def test_kabsch_rmsd_ignores_atom_order(shuffled: ShuffledCloudPair) -> None:
    """Reordering both clouds in the same way leaves the RMSD unchanged."""
    pair, order = shuffled.pair, shuffled.order
    assert torch.allclose(
        kabsch_rmsd(pair.mobile[order], pair.target[order]),
        kabsch_rmsd(pair.mobile, pair.target),
        atol=GEOMETRY_TOLERANCE,
    )


@given(pair=cloud_pairs())
def test_kabsch_rmsd_is_symmetric(pair: CloudPair) -> None:
    """Swapping mobile and target gives the same RMSD."""
    assert torch.allclose(
        kabsch_rmsd(pair.mobile, pair.target),
        kabsch_rmsd(pair.target, pair.mobile),
        atol=GEOMETRY_TOLERANCE,
    )


@given(masked=masked_cloud_pairs())
def test_kabsch_rmsd_mask_matches_selected_atoms(
    masked: MaskedCloudPair,
) -> None:
    """Masking atoms out is the same as deleting them."""
    pair, mask = masked.pair, masked.mask
    assert torch.allclose(
        kabsch_rmsd(pair.mobile, pair.target, mask=mask),
        kabsch_rmsd(pair.mobile[mask], pair.target[mask]),
        atol=TOLERANCE,
    )


@given(pair=cloud_pairs(), scale=SCALES)
def test_kabsch_rmsd_uniform_weights_match_unweighted(
    pair: CloudPair,
    scale: float,
) -> None:
    """Equal weights of any size match no weights at all."""
    weights = torch.full_like(pair.mobile[:, 0], scale)
    assert torch.allclose(
        kabsch_rmsd(pair.mobile, pair.target, weights=weights),
        kabsch_rmsd(pair.mobile, pair.target),
        atol=TOLERANCE,
    )


@given(masked=masked_cloud_pairs(), scale=SCALES)
def test_kabsch_rmsd_weights_and_mask_combine(
    masked: MaskedCloudPair,
    scale: float,
) -> None:
    """Equal weights on top of a mask match using the mask alone."""
    pair, mask = masked.pair, masked.mask
    weights = torch.full_like(pair.mobile[:, 0], scale)
    assert torch.allclose(
        kabsch_rmsd(pair.mobile, pair.target, weights=weights, mask=mask),
        kabsch_rmsd(pair.mobile[mask], pair.target[mask]),
        atol=TOLERANCE,
    )


@given(pair=cloud_pairs(), rotation=rotations(), shift=translations())
def test_apply_transform_reproduces_kabsch_align(
    pair: CloudPair,
    rotation: Float[torch.Tensor, "3 3"],
    shift: Float[torch.Tensor, "1 3"],
) -> None:
    """Reusing the returned transform gives the same coordinates."""
    moved = rotate(pair.target, rotation) + shift
    aligned, R, c_mobile, c_target = (  # pylint: disable=unpacking-non-sequence
        kabsch_align(moved, pair.mobile, return_transform=True)
    )
    assert torch.allclose(
        apply_transform(moved, R, c_mobile, c_target),
        aligned,
        atol=GEOMETRY_TOLERANCE,
    )


@given(pair=cloud_pairs())
def test_rmsd_is_symmetric(pair: CloudPair) -> None:
    """Swapping the arguments gives the same RMSD."""
    assert torch.allclose(
        rmsd(pair.mobile, pair.target),
        rmsd(pair.target, pair.mobile),
    )


@given(masked=masked_cloud_pairs())
def test_rmsd_mask_matches_selected_atoms(masked: MaskedCloudPair) -> None:
    """Masking atoms out is the same as deleting them."""
    pair, mask = masked.pair, masked.mask
    assert torch.allclose(
        rmsd(pair.mobile, pair.target, mask=mask),
        rmsd(pair.mobile[mask], pair.target[mask]),
    )


@given(masked=masked_cloud_pairs())
def test_rmsd_weights_match_mask(masked: MaskedCloudPair) -> None:
    """Weights of 1 and 0 select atoms the same way a mask does."""
    pair, mask = masked.pair, masked.mask
    assert torch.allclose(
        rmsd(pair.mobile, pair.target, weights=mask.double()),
        rmsd(pair.mobile, pair.target, mask=mask),
        atol=TOLERANCE,
    )


@given(masked=masked_cloud_pairs(), shift=translations())
def test_masked_com_matches_selected_atoms(
    masked: MaskedCloudPair,
    shift: Float[torch.Tensor, "1 3"],
) -> None:
    """The centre uses only valid atoms and follows a translation."""
    coords = rearrange(masked.pair.mobile, "n d -> 1 n d")
    mask = rearrange(masked.mask, "n -> 1 n")
    expected = reduce(masked.pair.mobile[masked.mask], "n d -> 1 1 d", "mean")
    moved = masked_com(coords + shift, mask=mask)
    assert torch.allclose(
        masked_com(coords, mask=mask),
        expected,
        atol=GEOMETRY_TOLERANCE,
    )
    assert torch.allclose(moved, expected + shift, atol=GEOMETRY_TOLERANCE)


@given(coords=batched_clouds())
def test_masked_com_all_valid_matches_mean(
    coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """With every atom valid, or no mask at all, the result is the mean."""
    mask = torch.ones_like(coords[..., 0], dtype=torch.bool)
    expected = reduce(coords, "b n d -> b 1 d", "mean")
    assert torch.equal(masked_com(coords), expected)
    assert torch.allclose(masked_com(coords, mask=mask), expected)


@given(coords=batched_clouds(), seed=SEEDS, s_trans=SCALES)
def test_centre_random_augment_is_rigid(
    coords: Float[torch.Tensor, "B N 3"],
    seed: int,
    s_trans: float,
) -> None:
    """Inner products between centred atoms are unchanged."""
    _ = manual_seed(seed)
    out = _centred(centre_random_augment(coords, s_trans=s_trans))
    before = _centred(coords)
    assert torch.allclose(
        einsum(out, out, "b n d, b m d -> b n m"),
        einsum(before, before, "b n d, b m d -> b n m"),
        rtol=GEOMETRY_TOLERANCE,
        atol=TOLERANCE,
    )


@given(coords=batched_clouds(min_atoms=MIN_CHIRAL_ATOMS), seed=SEEDS)
def test_centre_random_augment_never_reflects(
    coords: Float[torch.Tensor, "B N 3"],
    seed: int,
) -> None:
    """Signed volume is preserved, so left-handed atoms stay left-handed."""
    _ = manual_seed(seed)
    out = centre_random_augment(coords)
    assert torch.allclose(
        _chirality(out),
        _chirality(coords),
        rtol=GEOMETRY_TOLERANCE,
        atol=TOLERANCE,
    )


@given(coords=batched_clouds(), seed=SEEDS)
def test_centre_random_augment_without_translation_is_centred(
    coords: Float[torch.Tensor, "B N 3"],
    seed: int,
) -> None:
    """With s_trans=0 the output centroid is the origin."""
    _ = manual_seed(seed)
    out = centre_random_augment(coords, s_trans=0.0)
    centroid = reduce(out, "b n d -> b d", "mean")
    assert centroid.abs().max().item() < GEOMETRY_TOLERANCE


@given(coords=batched_clouds(), seed=SEEDS, scale=SCALES)
def test_centre_random_augment_centroid_scales_with_s_trans(
    coords: Float[torch.Tensor, "B N 3"],
    seed: int,
    scale: float,
) -> None:
    """For the same draw, the output centroid is linear in s_trans."""
    _ = manual_seed(seed)
    base = centre_random_augment(coords, s_trans=1.0)
    _ = manual_seed(seed)
    scaled = centre_random_augment(coords, s_trans=scale)
    assert torch.allclose(
        reduce(scaled, "b n d -> b d", "mean"),
        scale * reduce(base, "b n d -> b d", "mean"),
        atol=GEOMETRY_TOLERANCE,
    )


@given(coords=batched_clouds(), shift=translations(), seed=SEEDS)
def test_centre_random_augment_ignores_input_position(
    coords: Float[torch.Tensor, "B N 3"],
    shift: Float[torch.Tensor, "1 3"],
    seed: int,
) -> None:
    """A translated input gives the same output for the same draw."""
    _ = manual_seed(seed)
    base = centre_random_augment(coords)
    _ = manual_seed(seed)
    moved = centre_random_augment(coords + rearrange(shift, "o d -> 1 o d"))
    assert torch.allclose(moved, base, atol=GEOMETRY_TOLERANCE)


@given(masked=masked_cloud_pairs(), seed=SEEDS)
def test_centre_random_augment_mask_ignores_padding_values(
    masked: MaskedCloudPair,
    seed: int,
) -> None:
    """Coordinates of padded atoms can't affect the valid atoms' output."""
    coords = rearrange(masked.pair.mobile, "n d -> 1 n d")
    other = rearrange(masked.pair.target, "n d -> 1 n d")
    mask = rearrange(masked.mask, "n -> 1 n")
    padded_differently = torch.where(
        rearrange(mask, "b n -> b n 1"),
        coords,
        other,
    )
    _ = manual_seed(seed)
    base = centre_random_augment(coords, mask=mask)
    _ = manual_seed(seed)
    moved = centre_random_augment(padded_differently, mask=mask)
    assert torch.allclose(moved[mask], base[mask], atol=GEOMETRY_TOLERANCE)

"""Tests for structure alignment helpers.

Covers kabsch_rotation, kabsch_align, kabsch_rmsd, rmsd, apply_transform,
and centre_random_augment — including correctness, shape-contract enforcement,
and numerical tolerance checks.
"""

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
from jaxtyping import Bool, Float, Int, TypeCheckError

_ = manual_seed(42)
N, B = 50, 8
TOLERANCE = 1e-5
TIGHT_TOLERANCE = 1e-3


@pytest.fixture
def uniform_aa_indices() -> Int[torch.Tensor, "B N"]:
    """Amino-acid indices of a single nonpolar residue type ('A' = 0).

    Used by atom_loss tests that don't care about residue-type weighting;
    a uniform residue type makes that weighting a constant factor.

    Returns:
        Integer tensor of shape (B, N), all entries 0.
    """
    return torch.zeros(B, N, dtype=torch.long)


@pytest.fixture
def ref() -> Float[torch.Tensor, "N 3"]:
    """Provide random reference coordinates (N, 3).

    Used as the fixed target in single-structure alignment tests.
    """
    return torch.randn(N, 3)


@pytest.fixture
def rigid_mobile(ref: Float[torch.Tensor, "N 3"]) -> Float[torch.Tensor, "N 3"]:
    """Ref after random rigid rotation+translation.

    A proper rotation (det = +1) is drawn from QR decomposition and a random
    translation is applied, giving a mobile structure that should align to ref
    with RMSD ≈ 0.
    """
    q = torch.qr(torch.randn(3, 3)).Q
    if q.det() < 0:
        q[:, 0] *= -1
    return einsum(ref, q, "n d, e d -> n e") + torch.randn(1, 3)


@pytest.fixture
def batch_ref() -> Float[torch.Tensor, "B N 3"]:
    """Provide batched random reference coordinates (B, N, 3).

    Seeded with 42 for reproducibility across batched alignment tests.
    """
    _ = manual_seed(42)
    return torch.randn(B, N, 3)


@pytest.fixture
def noisy_mobile(
    batch_ref: Float[torch.Tensor, "B N 3"],
) -> Float[torch.Tensor, "B N 3"]:
    """Provide batch_ref with small Gaussian noise (sigma=0.1) added.

    Gives a mobile batch that is close but not identical to the reference,
    suitable for testing RMSD is small but non-zero.
    """
    return batch_ref + 0.1 * torch.randn(B, N, 3)


@pytest.fixture
def tail_weights() -> Float[torch.Tensor, "N"]:
    """Provide per-atom weights that zero out residues 40+.

    Used to verify that masked Kabsch alignment ignores the tail atoms
    when computing RMSD.
    """
    w = torch.ones(N)
    w[40:] = 0.0
    return w


def test_rotation_translation_rmsd_near_zero(
    ref: Float[torch.Tensor, "N 3"],
    rigid_mobile: Float[torch.Tensor, "N 3"],
) -> None:
    """Kabsch RMSD is near zero after an exact rigid rotation+translation.

    Verifies that kabsch_rmsd correctly recovers the superimposition for a
    structure transformed by a known proper rotation and translation.
    """
    assert kabsch_rmsd(rigid_mobile, ref).item() < TIGHT_TOLERANCE


def test_kabsch_rmsd_batched_rigid_transforms_near_zero(
    batch_ref: Float[torch.Tensor, "B N 3"],
) -> None:
    """Batched Kabsch RMSD ~zero for rigid-transformed structures.

    Each batch element is transformed with a distinct proper rotation and
    translation; kabsch_rmsd must recover near-zero RMSD for all elements
    simultaneously.
    """
    _ = manual_seed(0)
    mobiles: list[Float[torch.Tensor, "N 3"]] = []
    for i in range(B):
        q = torch.qr(torch.randn(3, 3)).Q
        if q.det() < 0:
            q[:, 0] *= -1
        t: Float[torch.Tensor, "1 3"] = torch.randn(1, 3)
        mobiles.append(einsum(batch_ref[i], q, "n d, e d -> n e") + t)
    mobile_batch: Float[torch.Tensor, "B N 3"] = torch.stack(mobiles)
    rmsds = kabsch_rmsd(mobile_batch, batch_ref)
    assert rmsds.shape == (B,)
    # Float32 Kabsch SVD residuals on random 50-atom clouds run ~2e-4;
    # 1e-3 gives headroom
    assert (rmsds < TIGHT_TOLERANCE).all()


def test_masked_rmsd_near_zero(
    ref: Float[torch.Tensor, "N 3"],
    rigid_mobile: Float[torch.Tensor, "N 3"],
    tail_weights: Float[torch.Tensor, "N"],
) -> None:
    """Masked Kabsch RMSD is near zero after an exact rigid transform.

    Verifies that per-atom weights are respected and the masked alignment
    achieves the same near-zero RMSD as the unmasked case.
    """
    assert (
        kabsch_rmsd(rigid_mobile, ref, weights=tail_weights).item()
        < TIGHT_TOLERANCE
    )


def test_apply_transform_reconstructs_target(
    ref: Float[torch.Tensor, "N 3"],
    rigid_mobile: Float[torch.Tensor, "N 3"],
) -> None:
    """apply_transform with Kabsch-recovered transform reconstructs target.

    Checks the round-trip: kabsch_align returns the rotation and centroids,
    and apply_transform using those values must reproduce the reference
    coordinates within TIGHT_TOLERANCE.
    """
    _, R, c_mob, c_tgt = kabsch_align(  # pylint: disable=unpacking-non-sequence
        rearrange(rigid_mobile, "n d -> 1 n d"),
        rearrange(ref, "n d -> 1 n d"),
        return_transform=True,
    )
    aligned = apply_transform(rigid_mobile, R, c_mob, c_tgt)
    assert torch.allclose(aligned, ref, atol=TIGHT_TOLERANCE)


def test_identity_alignment_unchanged(ref: Float[torch.Tensor, "N 3"]) -> None:
    """Aligning a structure to itself leaves it unchanged.

    Verifies that kabsch_align is idempotent when mobile and target are the
    same tensor — the identity transform should be applied.
    """
    (aligned,) = kabsch_align(  # pylint: disable=unpacking-non-sequence
        ref,
        ref,
        return_transform=False,
    )
    assert torch.allclose(aligned, ref, atol=TOLERANCE)


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


def test_masked_com_no_mask_matches_plain_mean(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Without a mask, masked_com reduces to the plain per-batch mean."""
    out: Float[torch.Tensor, "B 1 3"] = masked_com(aug_coords)
    expected: Float[torch.Tensor, "B 1 3"] = reduce(
        aug_coords,
        "b n d -> b 1 d",
        "mean",
    )
    assert torch.allclose(out, expected, atol=TOLERANCE)


def test_masked_com_ignores_zero_padded_atoms() -> None:
    """Zero-padded atoms don't bias the centre of mass toward the origin.

    A structure with real atoms offset from the origin, plus zero-padded
    atoms appended, must have the same masked_com as the real atoms alone.
    """
    real: Float[torch.Tensor, "1 N 3"] = rearrange(
        torch.randn(N, 3) + 5.0,
        "n d -> 1 n d",
    )
    padded: Float[torch.Tensor, "1 N2 3"] = torch.cat(
        [real, torch.zeros(1, N, 3)],
        dim=1,
    )
    mask: Bool[torch.Tensor, "1 N2"] = torch.cat(
        [
            torch.ones(1, N, dtype=torch.bool),
            torch.zeros(1, N, dtype=torch.bool),
        ],
        dim=1,
    )
    out: Float[torch.Tensor, "1 1 3"] = masked_com(padded, mask=mask)
    expected: Float[torch.Tensor, "1 1 3"] = reduce(
        real,
        "b n d -> b 1 d",
        "mean",
    )
    assert torch.allclose(out, expected, atol=TOLERANCE)


def test_masked_com_all_valid_mask_matches_unmasked(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """An all-True mask reproduces the unmasked result exactly."""
    mask: Bool[torch.Tensor, "B N"] = torch.ones(B, N, dtype=torch.bool)
    out_masked: Float[torch.Tensor, "B 1 3"] = masked_com(aug_coords, mask=mask)
    out_unmasked: Float[torch.Tensor, "B 1 3"] = masked_com(aug_coords)
    assert torch.equal(out_masked, out_unmasked)


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


def test_centre_random_augment_zero_s_trans_gives_zero_centroid(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """With s_trans=0, each batch element is exactly centred (mean ≈ 0).

    Verifies that centre_random_augment subtracts the centroid for every
    batch element and, with translation noise disabled, applies no further
    offset before returning.
    """
    out = centre_random_augment(aug_coords, s_trans=0.0)
    centroid: Float[torch.Tensor, "B 3"] = reduce(out, "b n d -> b d", "mean")
    assert centroid.abs().max().item() < TOLERANCE


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


def test_centre_random_augment_translation_scales_with_s_trans(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Output centroid scales linearly with s_trans.

    Per t ~ s_trans * N(0, I3), the underlying standard-normal translation
    draw is identical regardless of s_trans for a fixed RNG state, so
    doubling s_trans must exactly double the resulting centroid offset.
    """
    _ = manual_seed(99)
    out_a = centre_random_augment(aug_coords, s_trans=2.0)
    centroid_a: Float[torch.Tensor, "B 3"] = reduce(
        out_a,
        "b n d -> b d",
        "mean",
    )

    _ = manual_seed(99)
    out_b = centre_random_augment(aug_coords, s_trans=4.0)
    centroid_b: Float[torch.Tensor, "B 3"] = reduce(
        out_b,
        "b n d -> b d",
        "mean",
    )

    assert torch.allclose(centroid_b, 2.0 * centroid_a, atol=TOLERANCE)


def test_centre_random_augment_preserves_gram_matrix(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """All pairwise inner products of the re-centred output are preserved.

    For a pure rotation R, (XR)^T (XR) = X^T X, so the Gram matrix of the
    output, re-centred about its own mean to remove the translation noise,
    is invariant. Catches accidental scaling or reflection.
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
    gram_before: Float[torch.Tensor, "B N N"] = einsum(
        centred,
        centred,
        "b n d, b m d -> b n m",
    )
    gram_after: Float[torch.Tensor, "B N N"] = einsum(
        out_centred,
        out_centred,
        "b n d, b m d -> b n m",
    )
    assert torch.allclose(gram_before, gram_after, atol=1e-3)


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


def test_centre_random_augment_is_translation_invariant(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Adding a global translation to input leaves the output unchanged.

    After centring, (coords + shift) - mean(coords + shift) = coords -
    mean(coords), so the same random rotation and translation noise act on
    identical centred data.
    """
    shift: Float[torch.Tensor, "1 1 3"] = rearrange(
        torch.tensor([5.0, -3.0, 2.0]),
        "d -> 1 1 d",
    )
    _ = manual_seed(123)
    out_original: Float[torch.Tensor, "B N 3"] = centre_random_augment(
        aug_coords,
    )
    _ = manual_seed(123)
    out_shifted: Float[torch.Tensor, "B N 3"] = centre_random_augment(
        aug_coords + shift,
    )
    assert torch.allclose(out_original, out_shifted, atol=TOLERANCE)


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


def test_centre_random_augment_single_atom_zero_s_trans_is_zero() -> None:
    """Single-atom cloud has zero centred coords; rotation maps zero to zero.

    Verifies that centre_random_augment handles the degenerate single-atom
    case correctly with translation noise disabled — only centred position
    is the origin, and any rotation leaves it there.
    """
    single: Float[torch.Tensor, "B 1 3"] = torch.randn(B, 1, 3) * 5.0
    out: Float[torch.Tensor, "B 1 3"] = centre_random_augment(
        single,
        s_trans=0.0,
    )
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_centre_random_augment_single_atom_equals_translation() -> None:
    """Single-atom output is independent of the input position.

    A single point is invariant under rotation about the origin, so once
    centred (collapsing to the origin) the only remaining contribution to
    the output is the added translation t — identical for any single-atom
    input under the same RNG state.
    """
    single: Float[torch.Tensor, "B 1 3"] = torch.randn(B, 1, 3) * 5.0
    _ = manual_seed(17)
    out: Float[torch.Tensor, "B 1 3"] = centre_random_augment(single)
    _ = manual_seed(17)
    zeroed: Float[torch.Tensor, "B 1 3"] = centre_random_augment(
        torch.zeros_like(single),
    )
    assert torch.allclose(out, zeroed, atol=1e-6)


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


def test_centre_random_augment_mask_recovers_valid_centroid(
    padded_coords: Float[torch.Tensor, "B N 3"],
    padding_mask: Bool[torch.Tensor, "B N"],
) -> None:
    """With a mask, only valid atoms are centred to zero mean.

    With s_trans=0 and a proper rotation about the origin, the valid atoms'
    weighted centroid stays at the origin post-augmentation only if the
    centroid used for centring was computed from valid atoms alone.
    """
    out: Float[torch.Tensor, "B N 3"] = centre_random_augment(
        padded_coords,
        s_trans=0.0,
        mask=padding_mask,
    )
    weights: Float[torch.Tensor, "B N 1"] = rearrange(
        padding_mask.float(),
        "b n -> b n 1",
    )
    valid_centroid: Float[torch.Tensor, "B 3"] = reduce(
        out * weights,
        "b n d -> b d",
        "sum",
    ) / reduce(weights, "b n d -> b d", "sum")
    assert valid_centroid.abs().max().item() < TOLERANCE


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


def test_centre_random_augment_all_valid_mask_matches_unmasked(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """An all-True mask reproduces the unmasked result exactly.

    When every atom is valid, the mask-aware centroid reduces to the plain
    mean, so passing an all-ones mask must be bit-identical to omitting the
    mask entirely under the same RNG state.
    """
    mask: Bool[torch.Tensor, "B N"] = torch.ones(B, N, dtype=torch.bool)
    _ = manual_seed(64)
    out_masked: Float[torch.Tensor, "B N 3"] = centre_random_augment(
        aug_coords,
        mask=mask,
    )
    _ = manual_seed(64)
    out_unmasked: Float[torch.Tensor, "B N 3"] = centre_random_augment(
        aug_coords,
    )
    assert torch.equal(out_masked, out_unmasked)


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

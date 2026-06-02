"""Tests for structure alignment helpers."""

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
    rmsd,
)
from helpers.useful_objects import manual_seed
from jaxtyping import Float, TypeCheckError

manual_seed(42)
N, B = 50, 8


@pytest.fixture
def ref() -> Float[torch.Tensor, "N 3"]:
    """Provide random reference coordinates (N, 3)."""
    return torch.randn(N, 3)


@pytest.fixture
def rigid_mobile(ref: Float[torch.Tensor, "N 3"]) -> Float[torch.Tensor, "N 3"]:
    """Provide ref after a random rigid rotation+translation (exact superimposition target)."""
    q = torch.qr(torch.randn(3, 3)).Q
    if q.det() < 0:
        q[:, 0] *= -1
    return einsum(ref, q, "n d, e d -> n e") + torch.randn(1, 3)


@pytest.fixture
def batch_ref() -> Float[torch.Tensor, "B N 3"]:
    """Provide batched random reference coordinates (B, N, 3)."""
    manual_seed(42)
    return torch.randn(B, N, 3)


@pytest.fixture
def noisy_mobile(batch_ref: Float[torch.Tensor, "B N 3"]) -> Float[torch.Tensor, "B N 3"]:
    """Provide batch_ref with small Gaussian noise (sigma=0.1) added."""
    return batch_ref + 0.1 * torch.randn(B, N, 3)


@pytest.fixture
def tail_weights() -> Float[torch.Tensor, "N"]:
    """Provide per-atom weights that zero out residues 40+."""
    w = torch.ones(N)
    w[40:] = 0.0
    return w


def test_rotation_translation_rmsd_near_zero(
    ref: Float[torch.Tensor, "N 3"], rigid_mobile: Float[torch.Tensor, "N 3"]
):
    """Kabsch RMSD is near zero after an exact rigid rotation+translation."""
    assert kabsch_rmsd(rigid_mobile, ref).item() < 1e-3


def test_kabsch_rmsd_batched_rigid_transforms_near_zero(batch_ref: Float[torch.Tensor, "B N 3"]):
    """Batched Kabsch RMSD is near zero for independently rigid-transformed structures."""
    manual_seed(0)
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
    # Float32 Kabsch SVD residuals on random 50-atom clouds run ~2e-4; 1e-3 gives headroom
    assert (rmsds < 1e-3).all()


def test_masked_rmsd_near_zero(
    ref: Float[torch.Tensor, "N 3"],
    rigid_mobile: Float[torch.Tensor, "N 3"],
    tail_weights: Float[torch.Tensor, "N"],
):
    """Masked Kabsch RMSD is near zero after an exact rigid transform."""
    assert kabsch_rmsd(rigid_mobile, ref, weights=tail_weights).item() < 1e-3


def test_apply_transform_reconstructs_target(
    ref: Float[torch.Tensor, "N 3"], rigid_mobile: Float[torch.Tensor, "N 3"]
):
    """apply_transform with the Kabsch-recovered transform reconstructs the target exactly."""
    _, R, c_mob, c_tgt = kabsch_align(
        rearrange(rigid_mobile, "n d -> 1 n d"),
        rearrange(ref, "n d -> 1 n d"),
        return_transform=True,
    )
    aligned = apply_transform(rigid_mobile, R, c_mob, c_tgt)
    assert torch.allclose(aligned, ref, atol=1e-4)


def test_identity_alignment_unchanged(ref: Float[torch.Tensor, "N 3"]):
    """Aligning a structure to itself leaves it unchanged."""
    (aligned,) = kabsch_align(ref, ref, return_transform=False)
    assert torch.allclose(aligned, ref, atol=1e-5)


def test_atom_loss_perfect_prediction_near_zero():
    """atom_loss is near zero when prediction equals ground truth."""
    r = torch.randn(B, N, 3)
    assert atom_loss(r, r).mean().item() < 1e-5


def test_atom_loss_noisy_prediction_positive_and_finite():
    """atom_loss is positive and finite for a noisy prediction."""
    r = torch.randn(B, N, 3)
    r_noisy = r + 0.5 * torch.randn(B, N, 3)
    loss = atom_loss(r, r_noisy)
    assert (loss > 0).all()
    assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_kabsch_rotation_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError."""
    mobile_bad = torch.zeros(N, 4)  # last dim must be 3
    reference = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        kabsch_rotation(mobile_bad, reference)


def test_kabsch_align_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError."""
    mobile_bad = torch.zeros(N, 4)  # last dim must be 3
    target = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        kabsch_align(mobile_bad, target, return_transform=False)


def test_rmsd_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError."""
    predicted_bad = torch.zeros(N, 4)  # last dim must be 3
    reference = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        rmsd(predicted_bad, reference)


def test_kabsch_rmsd_wrong_shape() -> None:
    """Wrong last dim (4 instead of 3) triggers TypeCheckError."""
    mobile_bad = torch.zeros(N, 4)  # last dim must be 3
    target = torch.zeros(N, 3)
    with pytest.raises(TypeCheckError):
        kabsch_rmsd(mobile_bad, target)


def test_apply_transform_wrong_shape() -> None:
    """Wrong coords last dim (4 instead of 3) triggers TypeCheckError."""
    coords_bad = torch.zeros(N, 4)  # last dim must be 3
    R = rearrange(torch.eye(3), "r c -> 1 r c")
    t = torch.zeros(1, 1, 3)
    with pytest.raises(TypeCheckError):
        apply_transform(coords_bad, R, t, t)


# ---------------------------------------------------------------------------
# centre_random_augment
# ---------------------------------------------------------------------------


@pytest.fixture
def aug_coords() -> Float[torch.Tensor, "B N 3"]:
    """Batched random atom coordinates at a large scale to expose centring errors."""
    manual_seed(7)
    return torch.randn(B, N, 3) * 10.0


def test_centre_random_augment_output_shape(aug_coords: Float[torch.Tensor, "B N 3"]) -> None:
    """Output shape is identical to input shape."""
    out = centre_random_augment(aug_coords)
    assert out.shape == aug_coords.shape


def test_centre_random_augment_centroid_is_zero(aug_coords: Float[torch.Tensor, "B N 3"]) -> None:
    """Each batch element is exactly centred (mean ≈ 0) after augmentation."""
    out = centre_random_augment(aug_coords)
    centroid: Float[torch.Tensor, "B 3"] = reduce(out, "b n d -> b d", "mean")
    assert centroid.abs().max().item() < 1e-5


def test_centre_random_augment_preserves_gram_matrix(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """All pairwise inner products are preserved — the map is an isometry.

    For a pure rotation R, (XR)^T (XR) = X^T X, so the Gram matrix is invariant.
    Catches accidental scaling or reflection.
    """
    centred: Float[torch.Tensor, "B N 3"] = aug_coords - reduce(
        aug_coords, "b n d -> b 1 d", "mean"
    )
    out: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    gram_before: Float[torch.Tensor, "B N N"] = einsum(centred, centred, "b n d, b m d -> b n m")
    gram_after: Float[torch.Tensor, "B N N"] = einsum(out, out, "b n d, b m d -> b n m")
    assert torch.allclose(gram_before, gram_after, atol=1e-3)


def test_centre_random_augment_invertible_via_kabsch(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Applying the Kabsch-recovered rotation inverse restores the centred input exactly.

    centre_random_augment computes  out = centred @ Q  for some Q ∈ SO(3).
    kabsch_rotation(out, centred) returns exactly Q, so  out @ Q^T = centred.
    """
    centred: Float[torch.Tensor, "B N 3"] = aug_coords - reduce(
        aug_coords, "b n d -> b 1 d", "mean"
    )
    out: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    # kabsch_rotation(mobile, reference) → R  such that  mobile @ R^T ≈ reference
    Q: Float[torch.Tensor, "B 3 3"] = kabsch_rotation(out, centred)
    recovered: Float[torch.Tensor, "B N 3"] = einsum(out, Q.mH, "b n i, b i j -> b n j")
    assert torch.allclose(recovered, centred, atol=1e-3)


def test_centre_random_augment_recovered_rotation_is_proper_SO3(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """The rotation Q actually applied by the augmentation has det = +1 and is orthogonal.

    Recovers Q directly via the normal equations:
        centred @ Q = out  =>  Q = (centred^T centred)^{-1} centred^T out

    This tests the rotation centre_random_augment applies, not the Kabsch-recovered
    rotation (which always has det = +1 by its own reflection correction).
    """
    centred: Float[torch.Tensor, "B N 3"] = aug_coords - reduce(
        aug_coords, "b n d -> b 1 d", "mean"
    )
    out: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    # Normal equations: gram @ Q_actual = centred^T @ out
    gram: Float[torch.Tensor, "B 3 3"] = einsum(centred, centred, "b n i, b n j -> b i j")
    cT_out: Float[torch.Tensor, "B 3 3"] = einsum(centred, out, "b n i, b n j -> b i j")
    Q_actual: Float[torch.Tensor, "B 3 3"] = torch.linalg.solve(gram, cT_out)
    # Orthogonality: Q^T Q = I
    eye_approx: Float[torch.Tensor, "B 3 3"] = einsum(
        Q_actual.mH, Q_actual, "b i j, b j k -> b i k"
    )
    assert torch.allclose(eye_approx, torch.eye(3).expand(B, -1, -1), atol=1e-3)
    # Proper rotation: det = +1 (not -1 reflection)
    dets: Float[torch.Tensor, "B"] = torch.linalg.det(Q_actual)
    assert torch.allclose(dets, torch.ones(B), atol=1e-3)


def test_centre_random_augment_is_translation_invariant(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Adding a global translation to input leaves the output unchanged.

    After centring, (coords + shift) - mean(coords + shift) = coords - mean(coords),
    so the same random rotation acts on identical centred data.
    """
    shift: Float[torch.Tensor, "1 1 3"] = rearrange(torch.tensor([5.0, -3.0, 2.0]), "d -> 1 1 d")
    torch.manual_seed(123)
    out_original: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    torch.manual_seed(123)
    out_shifted: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords + shift)
    assert torch.allclose(out_original, out_shifted, atol=1e-5)


def test_centre_random_augment_applies_independent_rotations_per_batch(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Different batch elements receive distinct SO(3) rotations.

    The probability that two Haar-uniform SO(3) samples are equal is zero;
    with float32 the empirical distance between any two rotation matrices
    should be well above numerical noise.
    """
    centred: Float[torch.Tensor, "B N 3"] = aug_coords - reduce(
        aug_coords, "b n d -> b 1 d", "mean"
    )
    out: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    Q: Float[torch.Tensor, "B 3 3"] = kabsch_rotation(out, centred)
    # Every pair of rotation matrices in the batch should differ by > 1e-2
    for i in range(B - 1):
        diff = (Q[i] - Q[i + 1]).abs().max().item()
        assert diff > 1e-2, f"Batch elements {i} and {i+1} received the same rotation"


def test_centre_random_augment_single_atom_is_always_zero() -> None:
    """A single-atom cloud has zero centred coordinates; rotation maps zero to zero."""
    single: Float[torch.Tensor, "B 1 3"] = torch.randn(B, 1, 3) * 5.0
    out: Float[torch.Tensor, "B 1 3"] = centre_random_augment(single)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_centre_random_augment_is_deterministic_under_fixed_seed(
    aug_coords: Float[torch.Tensor, "B N 3"],
) -> None:
    """Two calls with the same RNG seed produce bit-identical results."""
    torch.manual_seed(55)
    out_a: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    torch.manual_seed(55)
    out_b: Float[torch.Tensor, "B N 3"] = centre_random_augment(aug_coords)
    assert torch.equal(out_a, out_b)


def test_centre_random_augment_wrong_last_dim_raises_type_error() -> None:
    """Passing coords with last dim ≠ 3 triggers a jaxtyping TypeCheckError."""
    bad: Float[torch.Tensor, "B N 4"] = torch.randn(B, N, 4)
    with pytest.raises(TypeCheckError):
        centre_random_augment(bad)

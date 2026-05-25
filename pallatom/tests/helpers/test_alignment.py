"""Tests for structure alignment helpers."""

import pytest
import torch
from architecture.losses import atom_loss
from einops import einsum, rearrange
from helpers.alignment import apply_transform, kabsch_align, kabsch_rmsd, kabsch_rotation, rmsd
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

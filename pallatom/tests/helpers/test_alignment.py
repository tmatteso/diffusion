import pytest
import torch
from architecture.losses import atom_loss
from einops import einsum
from helpers.alignment import apply_transform, kabsch_align, kabsch_rmsd

torch.manual_seed(42)
N, B = 50, 8


@pytest.fixture
def ref():
    return torch.randn(N, 3)


@pytest.fixture
def rigid_mobile(ref):
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return einsum(ref, Q, "n d, e d -> n e") + torch.randn(1, 3)


@pytest.fixture
def batch_ref():
    return torch.randn(B, N, 3)


@pytest.fixture
def noisy_mobile(batch_ref):
    return batch_ref + 0.1 * torch.randn(B, N, 3)


@pytest.fixture
def tail_weights():
    w = torch.ones(N)
    w[40:] = 0.0
    return w


def test_rotation_translation_rmsd_near_zero(ref, rigid_mobile):
    assert kabsch_rmsd(rigid_mobile, ref).item() < 1e-3


def test_kabsch_rmsd_batched_rigid_transforms_near_zero(batch_ref):
    torch.manual_seed(0)
    mobiles = []
    for i in range(B):
        Q, _ = torch.linalg.qr(torch.randn(3, 3))
        if torch.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = torch.randn(1, 3)
        mobiles.append(einsum(batch_ref[i], Q, "n d, e d -> n e") + t)
    mobiles = torch.stack(mobiles)
    rmsds = kabsch_rmsd(mobiles, batch_ref)
    assert rmsds.shape == (B,)
    assert (rmsds < 5e-4).all()


def test_masked_rmsd_near_zero(ref, rigid_mobile, tail_weights):
    assert kabsch_rmsd(rigid_mobile, ref, weights=tail_weights).item() < 1e-3


def test_apply_transform_reconstructs_target(ref, rigid_mobile):
    _, R, c_mob, c_tgt = kabsch_align(
        rigid_mobile.unsqueeze(0), ref.unsqueeze(0), return_transform=True
    )
    aligned = apply_transform(rigid_mobile, R, c_mob, c_tgt)
    assert torch.allclose(aligned, ref, atol=1e-4)


def test_identity_alignment_unchanged(ref):
    (aligned,) = kabsch_align(ref, ref)
    assert torch.allclose(aligned, ref, atol=1e-5)


def test_atom_loss_perfect_prediction_near_zero():
    r = torch.randn(B, N, 3)
    assert atom_loss(r, r).mean().item() < 1e-5


def test_atom_loss_noisy_prediction_positive_and_finite():
    r = torch.randn(B, N, 3)
    r_noisy = r + 0.5 * torch.randn(B, N, 3)
    loss = atom_loss(r, r_noisy)
    assert (loss > 0).all()
    assert torch.isfinite(loss).all()

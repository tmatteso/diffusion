import pytest
import torch
from architecture.pair_update import (
    DropoutColumnwise,
    DropoutRowwise,
    PairUpdate,
    TransformRBF,
    Transition,
    TriangleAttentionEndingNodeWithBias,
    TriangleAttentionStartingNodeWithBias,
)
from beartype import beartype
from einops import einsum, rearrange, reduce
from jaxtyping import Float, jaxtyped

torch.manual_seed(42)

N_RES = 8
C = 32
N_HEADS = 4
B = 2


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def mean_abs_asymmetry(
    x: Float[torch.Tensor, "B N N C"],
) -> Float[torch.Tensor, ""]:
    return reduce((x - rearrange(x, "b i j c -> b j i c")).abs(), "b i j c -> ", "mean")


@jaxtyped(typechecker=beartype)
def compute_dij(
    r: Float[torch.Tensor, "B N_res 3"],
) -> Float[torch.Tensor, "B N_res N_res"]:
    diff = rearrange(r, "b n d -> b n 1 d") - rearrange(r, "b n d -> b 1 n d")
    return diff.norm(dim=-1)


@jaxtyped(typechecker=beartype)
def random_rotation() -> Float[torch.Tensor, "3 3"]:
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if Q.det() < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@jaxtyped(typechecker=beartype)
def apply_rotation(
    r: Float[torch.Tensor, "B N_res 3"],
    R: Float[torch.Tensor, "3 3"],
) -> Float[torch.Tensor, "B N_res 3"]:
    return einsum(r, R, "b n d, d e -> b n e")


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rbf():
    return TransformRBF(C).eval()


@pytest.fixture
def tri_start():
    return TriangleAttentionStartingNodeWithBias(C, n_heads=N_HEADS).eval()


@pytest.fixture
def tri_end():
    return TriangleAttentionEndingNodeWithBias(C, n_heads=N_HEADS).eval()


@pytest.fixture
def transition():
    return Transition(C).eval()


@pytest.fixture
def pair_update():
    return PairUpdate(C, n_rbf=16, n_heads=N_HEADS, dropout=0.0).eval()


# ---------------------------------------------------------------------------
# Input tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def z():
    return torch.randn(B, N_RES, N_RES, C)


@pytest.fixture
def b():
    return torch.randn(B, N_RES, N_RES, C)


@pytest.fixture
def d():
    pos = torch.randn(B, N_RES, 3)
    diff = rearrange(pos, "b n c -> b n 1 c") - rearrange(pos, "b n c -> b 1 n c")
    return diff.norm(dim=-1)


@pytest.fixture
def r_center():
    return torch.randn(B, N_RES, 3)


# ---------------------------------------------------------------------------
# TransformRBF
# ---------------------------------------------------------------------------


def test_transform_rbf_output_shape(rbf, d):
    with torch.no_grad():
        out = rbf(d)
    assert out.shape == (B, N_RES, N_RES, C)


def test_transform_rbf_output_finite(rbf, d):
    with torch.no_grad():
        out = rbf(d)
    assert torch.isfinite(out).all()


def test_transform_rbf_symmetric_distance_gives_symmetric_output(rbf, d):
    # d is a Euclidean distance matrix, so d[b,i,j] == d[b,j,i]; output must match
    with torch.no_grad():
        out = rbf(d)
    assert mean_abs_asymmetry(out).item() < 1e-5


# ---------------------------------------------------------------------------
# TriangleAttentionStartingNodeWithBias
# ---------------------------------------------------------------------------


def test_tri_start_output_shape(tri_start, z, b):
    with torch.no_grad():
        out = tri_start(z, b)
    assert out.shape == (B, N_RES, N_RES, C)


def test_tri_start_output_finite(tri_start, z, b):
    with torch.no_grad():
        out = tri_start(z, b)
    assert torch.isfinite(out).all()


def test_tri_start_gradient_flows(tri_start, z, b):
    z_g = z.clone().requires_grad_(True)
    out = tri_start(z_g, b)
    reduce(out, "b n m c -> ", "sum").backward()
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_tri_start_row_independence(tri_start, z, b):
    # Starting-node attention fixes row i and attends over columns j.
    # Q, K, V, G for row i all come from z[:, i, :], so modifying row 0 must
    # leave output rows 1..N_RES-1 completely unchanged.
    z_mod = z.clone()
    z_mod[:, 0] = torch.randn_like(z_mod[:, 0])
    with torch.no_grad():
        out_orig = tri_start(z, b)
        out_mod = tri_start(z_mod, b)
    assert torch.allclose(out_orig[:, 1:], out_mod[:, 1:], atol=1e-5)


# ---------------------------------------------------------------------------
# TriangleAttentionEndingNodeWithBias
# ---------------------------------------------------------------------------


def test_tri_end_output_shape(tri_end, z, b):
    with torch.no_grad():
        out = tri_end(z, b)
    assert out.shape == (B, N_RES, N_RES, C)


def test_tri_end_output_finite(tri_end, z, b):
    with torch.no_grad():
        out = tri_end(z, b)
    assert torch.isfinite(out).all()


def test_tri_end_gradient_flows(tri_end, z, b):
    z_g = z.clone().requires_grad_(True)
    out = tri_end(z_g, b)
    reduce(out, "b n m c -> ", "sum").backward()
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_tri_end_col_independence(tri_end, z, b):
    # Ending-node attention fixes column j and attends over rows i.
    # Q, K, V, G for column j all come from z[:, :, j], so modifying column 0
    # must leave output columns 1..N_RES-1 completely unchanged.
    z_mod = z.clone()
    z_mod[:, :, 0, :] = torch.randn_like(z_mod[:, :, 0, :])
    with torch.no_grad():
        out_orig = tri_end(z, b)
        out_mod = tri_end(z_mod, b)
    assert torch.allclose(out_orig[:, :, 1:, :], out_mod[:, :, 1:, :], atol=1e-5)


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


def test_transition_output_shape_3d(transition, z):
    with torch.no_grad():
        out = transition(z)
    assert out.shape == z.shape


def test_transition_output_shape_2d(transition):
    x = torch.randn(N_RES, C)
    with torch.no_grad():
        out = transition(x)
    assert out.shape == (N_RES, C)


def test_transition_output_finite(transition, z):
    with torch.no_grad():
        out = transition(z)
    assert torch.isfinite(out).all()


def test_transition_gradient_flows(transition, z):
    z_g = z.clone().requires_grad_(True)
    out = transition(z_g)
    reduce(out, "b n m c -> ", "sum").backward()
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


# ---------------------------------------------------------------------------
# DropoutRowwise
# ---------------------------------------------------------------------------


def test_dropout_rowwise_eval_is_identity(z):
    drop = DropoutRowwise(p=0.5)
    drop.eval()
    assert torch.equal(drop(z), z)


def test_dropout_rowwise_train_preserves_shape(z):
    drop = DropoutRowwise(p=0.5)
    drop.train()
    out = drop(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_dropout_rowwise_train_zeroes_entire_rows():
    # Use a ones input so a dropped row is all-zero and a kept row is all-ones.
    torch.manual_seed(0)
    x = torch.ones(B, N_RES, N_RES, C)
    drop = DropoutRowwise(p=0.5)
    drop.train()
    out = drop(x)
    for i in range(N_RES):
        row = out[0, i]  # shape (N_RES, C) — check first batch item
        assert torch.allclose(row, torch.zeros_like(row)) or torch.allclose(
            row, torch.ones_like(row)
        )


# ---------------------------------------------------------------------------
# DropoutColumnwise
# ---------------------------------------------------------------------------


def test_dropout_columnwise_eval_is_identity(z):
    drop = DropoutColumnwise(p=0.5)
    drop.eval()
    assert torch.equal(drop(z), z)


def test_dropout_columnwise_train_preserves_shape(z):
    drop = DropoutColumnwise(p=0.5)
    drop.train()
    out = drop(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_dropout_columnwise_train_zeroes_entire_cols():
    torch.manual_seed(0)
    x = torch.ones(B, N_RES, N_RES, C)
    drop = DropoutColumnwise(p=0.5)
    drop.train()
    out = drop(x)
    for j in range(N_RES):
        col = out[0, :, j, :]  # shape (N_RES, C) — check first batch item
        assert torch.allclose(col, torch.zeros_like(col)) or torch.allclose(
            col, torch.ones_like(col)
        )


# ---------------------------------------------------------------------------
# PairUpdate
# ---------------------------------------------------------------------------


def test_pair_update_changes_input(pair_update, z, r_center):
    with torch.no_grad():
        out = pair_update(z, r_center)
    assert not torch.allclose(out, z)


def test_pair_update_gradient_flows_to_z(pair_update, z, r_center):
    z_g = z.clone().requires_grad_(True)
    out = pair_update(z_g, r_center)
    reduce(out, "b n m c -> ", "sum").backward()
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_pair_update_gradient_flows_to_r_center(pair_update, z, r_center):
    r_g = r_center.clone().requires_grad_(True)
    out = pair_update(z, r_g)
    reduce(out, "b n m c -> ", "sum").backward()
    assert r_g.grad is not None
    assert torch.isfinite(r_g.grad).all()


def test_pair_update_no_nan_grad_from_zero_diagonal_distance(pair_update):
    # d_ij[b, i, i] = 0 always; torch.norm's backward clamps the denominator so
    # the gradient must be finite rather than nan at the zero-distance diagonal.
    r_g = torch.randn(B, N_RES, 3, requires_grad=True)
    out = pair_update(torch.randn(B, N_RES, N_RES, C), r_g)
    reduce(out, "b n m c -> ", "sum").backward()
    assert torch.isfinite(r_g.grad).all()


def test_tri_start_changes_with_pair_bias(tri_start, z):
    b1 = torch.randn(B, N_RES, N_RES, C)
    b2 = torch.randn(B, N_RES, N_RES, C)
    with torch.no_grad():
        out1 = tri_start(z, b1)
        out2 = tri_start(z, b2)
    assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Geometric Invariance
# Layer 1: distance computation is invariant to translation and rotation.
# Layer 2: TransformRBF output is invariant (invariant input → invariant output).
# Layer 3: PairUpdate end-to-end output is invariant.
# ---------------------------------------------------------------------------


def test_distance_translation_invariant(r_center):
    t = torch.randn(1, 1, 3)
    with torch.no_grad():
        d_orig = compute_dij(r_center)
        d_shift = compute_dij(r_center + t)
    assert torch.allclose(d_orig, d_shift, atol=1e-5)


def test_distance_rotation_invariant(r_center):
    R = random_rotation()
    with torch.no_grad():
        d_orig = compute_dij(r_center)
        d_rot = compute_dij(apply_rotation(r_center, R))
    assert torch.allclose(d_orig, d_rot, atol=1e-5)


def test_rbf_translation_invariant(rbf, r_center):
    t = torch.randn(1, 1, 3)
    with torch.no_grad():
        b_orig = rbf(compute_dij(r_center))
        b_shift = rbf(compute_dij(r_center + t))
    assert torch.allclose(b_orig, b_shift, atol=1e-5)


def test_rbf_rotation_invariant(rbf, r_center):
    R = random_rotation()
    with torch.no_grad():
        b_orig = rbf(compute_dij(r_center))
        b_rot = rbf(compute_dij(apply_rotation(r_center, R)))
    assert torch.allclose(b_orig, b_rot, atol=1e-5)


def test_pair_update_translation_invariant(pair_update, z, r_center):
    t = torch.randn(1, 1, 3)
    with torch.no_grad():
        out_orig = pair_update(z, r_center)
        out_shift = pair_update(z, r_center + t)
    assert torch.allclose(out_orig, out_shift, atol=1e-5)


def test_pair_update_rotation_invariant(pair_update, z, r_center):
    R = random_rotation()
    with torch.no_grad():
        out_orig = pair_update(z, r_center)
        out_rot = pair_update(z, apply_rotation(r_center, R))
    assert torch.allclose(out_orig, out_rot, atol=1e-5)

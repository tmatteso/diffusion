import pytest
import torch
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float, jaxtyped

from architecture.pair_update import (
    DropoutColumnwise,
    DropoutRowwise,
    PairUpdate,
    TransformRBF,
    Transition,
    TriangleAttentionEndingNodeWithBias,
    TriangleAttentionStartingNodeWithBias,
)

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
        out_mod  = tri_start(z_mod, b)
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
        out_mod  = tri_end(z_mod, b)
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
        row = out[0, i]   # shape (N_RES, C) — check first batch item
        assert torch.allclose(row, torch.zeros_like(row)) or torch.allclose(row, torch.ones_like(row))


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
        col = out[0, :, j, :]   # shape (N_RES, C) — check first batch item
        assert torch.allclose(col, torch.zeros_like(col)) or torch.allclose(col, torch.ones_like(col))


# ---------------------------------------------------------------------------
# PairUpdate
# ---------------------------------------------------------------------------

def test_pair_update_output_shape(pair_update, z, r_center):
    with torch.no_grad():
        out = pair_update(z, r_center)
    assert out.shape == (B, N_RES, N_RES, C)


def test_pair_update_output_finite(pair_update, z, r_center):
    with torch.no_grad():
        out = pair_update(z, r_center)
    assert torch.isfinite(out).all()


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

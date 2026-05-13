"""Tests for node update modules."""

import pytest
import torch
from architecture.node_update import AttentionPairBias, NodeUpdate
from beartype import beartype
from einops import reduce
from jaxtyping import Float, jaxtyped

torch.manual_seed(42)

N_RES = 8
C_RES = 32
C_PAIR = 32
N_HEADS = 4
B = 2


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def scalar_sum(
    x: Float[torch.Tensor, "B N C"],
) -> Float[torch.Tensor, ""]:
    """Sum all elements of x to a scalar for use as a backward root."""
    return reduce(x, "b n c -> ", "sum")


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def attn() -> AttentionPairBias:
    """Provide an AttentionPairBias module in eval mode."""
    return AttentionPairBias(C_RES, C_PAIR, n_heads=N_HEADS).eval()


@pytest.fixture
def node_update() -> NodeUpdate:
    """Provide a NodeUpdate module (no dropout) in eval mode."""
    return NodeUpdate(C_RES, C_PAIR, n_heads=N_HEADS, dropout=0.0).eval()


# ---------------------------------------------------------------------------
# Input tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def s() -> Float[torch.Tensor, "B N_res C_res"]:
    """Provide a random single-embedding tensor (B, N_RES, C_RES)."""
    return torch.randn(B, N_RES, C_RES)


@pytest.fixture
def t() -> Float[torch.Tensor, "B N_res C_res"]:
    """Provide a random time-conditioning tensor (B, N_RES, C_RES)."""
    return torch.randn(B, N_RES, C_RES)


@pytest.fixture
def z() -> Float[torch.Tensor, "B N_res N_res C_pair"]:
    """Provide a random pair-embedding tensor (B, N_RES, N_RES, C_PAIR)."""
    return torch.randn(B, N_RES, N_RES, C_PAIR)


# ---------------------------------------------------------------------------
# AttentionPairBias
# ---------------------------------------------------------------------------


def test_attn_pair_bias_output_shape(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """AttentionPairBias output shape matches (B, N_RES, C_RES)."""
    with torch.no_grad():
        out = attn(s, t, z)
    assert out.shape == (B, N_RES, C_RES)


def test_attn_pair_bias_output_finite(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """AttentionPairBias output contains only finite values."""
    with torch.no_grad():
        out = attn(s, t, z)
    assert torch.isfinite(out).all()


def test_attn_pair_bias_output_dtype(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """AttentionPairBias output dtype matches the input single-embedding dtype."""
    with torch.no_grad():
        out = attn(s, t, z)
    assert out.dtype == s.dtype


def test_attn_pair_bias_gradient_flows_to_s(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Gradient propagates from AttentionPairBias output back to s."""
    s_g = s.clone().requires_grad_(True)
    scalar_sum(attn(s_g, t, z)).backward()
    assert s_g.grad is not None
    assert torch.isfinite(s_g.grad).all()


def test_attn_pair_bias_time_conditioning_affects_output(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Different time-conditioning inputs produce different AttentionPairBias outputs."""
    t1 = torch.randn(B, N_RES, C_RES)
    t2 = torch.randn(B, N_RES, C_RES)
    with torch.no_grad():
        out1 = attn(s, t1, z)
        out2 = attn(s, t2, z)
    assert not torch.allclose(out1, out2)


def test_attn_pair_bias_pair_bias_affects_output(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
) -> None:
    """Different pair-embedding inputs produce different attention-weighted outputs."""
    z1 = torch.randn(B, N_RES, N_RES, C_PAIR)
    z2 = torch.randn(B, N_RES, N_RES, C_PAIR)
    with torch.no_grad():
        out1 = attn(s, t, z1)
        out2 = attn(s, t, z2)
    assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# NodeUpdate
# ---------------------------------------------------------------------------


def test_node_update_output_shape(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """NodeUpdate returns an updated single embedding of the same [B, N_res, C_res] shape."""
    with torch.no_grad():
        out = node_update(s, t, z)
    assert out.shape == (B, N_RES, C_RES)


def test_node_update_output_finite(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """NodeUpdate produces no NaN or Inf values for random valid inputs."""
    with torch.no_grad():
        out = node_update(s, t, z)
    assert torch.isfinite(out).all()


def test_node_update_changes_input(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """NodeUpdate is not an identity — the output differs from the input single embedding."""
    with torch.no_grad():
        out = node_update(s, t, z)
    assert not torch.allclose(out, s)


def test_node_update_gradient_flows_to_s(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Gradients flow back through NodeUpdate to the input single embedding s."""
    s_g = s.clone().requires_grad_(True)
    scalar_sum(node_update(s_g, t, z)).backward()
    assert s_g.grad is not None
    assert torch.isfinite(s_g.grad).all()


def test_node_update_gradient_flows_to_t(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Gradients flow back through NodeUpdate to the time-conditioning input t."""
    t_g = t.clone().requires_grad_(True)
    scalar_sum(node_update(s, t_g, z)).backward()
    assert t_g.grad is not None
    assert torch.isfinite(t_g.grad).all()


def test_node_update_gradient_flows_to_z(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Gradients flow back through NodeUpdate to the pair embedding z (via the attention bias)."""
    z_g = z.clone().requires_grad_(True)
    scalar_sum(node_update(s, t, z_g)).backward()
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_node_update_eval_is_deterministic(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """In eval mode DropoutRowwise is identity, so repeated calls return bit-identical outputs."""
    # DropoutRowwise is identity in eval mode, so repeated calls must match exactly.
    with torch.no_grad():
        out1 = node_update(s, t, z)
        out2 = node_update(s, t, z)
    assert torch.equal(out1, out2)


def test_node_update_train_dropout_preserves_shape_and_finite(
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """With p=0.5 dropout in training mode, NodeUpdate produces correct shape with finite values."""
    model = NodeUpdate(C_RES, C_PAIR, n_heads=N_HEADS, dropout=0.5).train()
    with torch.no_grad():
        out = model(s, t, z)
    assert out.shape == (B, N_RES, C_RES)
    assert torch.isfinite(out).all()

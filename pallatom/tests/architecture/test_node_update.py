import pytest
import torch
from beartype import beartype
from einops import reduce
from jaxtyping import Float, jaxtyped

from architecture.node_update import AttentionPairBias, NodeUpdate

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
    return reduce(x, "b n c -> ", "sum")


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def attn():
    return AttentionPairBias(C_RES, C_PAIR, n_heads=N_HEADS).eval()


@pytest.fixture
def node_update():
    return NodeUpdate(C_RES, C_PAIR, n_heads=N_HEADS, dropout=0.0).eval()


# ---------------------------------------------------------------------------
# Input tensor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def s():
    return torch.randn(B, N_RES, C_RES)


@pytest.fixture
def t():
    return torch.randn(B, N_RES, C_RES)


@pytest.fixture
def z():
    return torch.randn(B, N_RES, N_RES, C_PAIR)


# ---------------------------------------------------------------------------
# AttentionPairBias
# ---------------------------------------------------------------------------

def test_attn_pair_bias_output_shape(attn, s, t, z):
    with torch.no_grad():
        out = attn(s, t, z)
    assert out.shape == (B, N_RES, C_RES)


def test_attn_pair_bias_output_finite(attn, s, t, z):
    with torch.no_grad():
        out = attn(s, t, z)
    assert torch.isfinite(out).all()


def test_attn_pair_bias_output_dtype(attn, s, t, z):
    with torch.no_grad():
        out = attn(s, t, z)
    assert out.dtype == s.dtype


def test_attn_pair_bias_gradient_flows_to_s(attn, s, t, z):
    s_g = s.clone().requires_grad_(True)
    scalar_sum(attn(s_g, t, z)).backward()
    assert s_g.grad is not None
    assert torch.isfinite(s_g.grad).all()


def test_attn_pair_bias_time_conditioning_affects_output(attn, s, z):
    t1 = torch.randn(B, N_RES, C_RES)
    t2 = torch.randn(B, N_RES, C_RES)
    with torch.no_grad():
        out1 = attn(s, t1, z)
        out2 = attn(s, t2, z)
    assert not torch.allclose(out1, out2)


def test_attn_pair_bias_pair_bias_affects_output(attn, s, t):
    z1 = torch.randn(B, N_RES, N_RES, C_PAIR)
    z2 = torch.randn(B, N_RES, N_RES, C_PAIR)
    with torch.no_grad():
        out1 = attn(s, t, z1)
        out2 = attn(s, t, z2)
    assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# NodeUpdate
# ---------------------------------------------------------------------------

def test_node_update_output_shape(node_update, s, t, z):
    with torch.no_grad():
        out = node_update(s, t, z)
    assert out.shape == (B, N_RES, C_RES)


def test_node_update_output_finite(node_update, s, t, z):
    with torch.no_grad():
        out = node_update(s, t, z)
    assert torch.isfinite(out).all()


def test_node_update_changes_input(node_update, s, t, z):
    with torch.no_grad():
        out = node_update(s, t, z)
    assert not torch.allclose(out, s)


def test_node_update_gradient_flows_to_s(node_update, s, t, z):
    s_g = s.clone().requires_grad_(True)
    scalar_sum(node_update(s_g, t, z)).backward()
    assert s_g.grad is not None
    assert torch.isfinite(s_g.grad).all()


def test_node_update_gradient_flows_to_t(node_update, s, t, z):
    t_g = t.clone().requires_grad_(True)
    scalar_sum(node_update(s, t_g, z)).backward()
    assert t_g.grad is not None
    assert torch.isfinite(t_g.grad).all()


def test_node_update_gradient_flows_to_z(node_update, s, t, z):
    z_g = z.clone().requires_grad_(True)
    scalar_sum(node_update(s, t, z_g)).backward()
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_node_update_eval_is_deterministic(node_update, s, t, z):
    # DropoutRowwise is identity in eval mode, so repeated calls must match exactly.
    with torch.no_grad():
        out1 = node_update(s, t, z)
        out2 = node_update(s, t, z)
    assert torch.equal(out1, out2)


def test_node_update_train_dropout_preserves_shape_and_finite(s, t, z):
    model = NodeUpdate(C_RES, C_PAIR, n_heads=N_HEADS, dropout=0.5).train()
    with torch.no_grad():
        out = model(s, t, z)
    assert out.shape == (B, N_RES, C_RES)
    assert torch.isfinite(out).all()

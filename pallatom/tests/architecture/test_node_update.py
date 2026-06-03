"""Tests for node update modules."""

import pytest
import torch
from architecture.node_update import AdaLN, AttentionPairBias, NodeUpdate
from beartype import beartype
from einops import reduce, repeat
from helpers.useful_objects import manual_seed
from jaxtyping import Float, Int, TypeCheckError, jaxtyped

manual_seed(42)

N_RES = 8
C_RES = 32
C_PAIR = 32
N_HEADS = 4
B = 2

# Sparse regression constants: K < N_RES to reproduce the production mismatch.
# In production the crash was attn (B, h, 640, 640) + pair_bias (B, h, 640, 635).
N_RES_LARGE = 20  # total atom / node count
K_SPARSE = 6  # neighbour count strictly less than N_RES_LARGE

RMS_GAIN = 5.0

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
    (grad,) = torch.autograd.grad(scalar_sum(attn(s_g, t, z)), s_g)
    assert torch.isfinite(grad).all()


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


def test_attn_pair_bias_norm_a_has_no_learnable_params() -> None:
    """norm_a must be non-learnable (elementwise_affine=False).

    All callers that exist in the current model (NodeUpdate, DiffusionTransformer)
    always supply a real conditioning tensor, so the s=None fallback path through
    norm_a is never taken.  If norm_a had learnable parameters they would never
    receive gradients, triggering a DDP "unused parameter" crash at runtime.
    """
    attn = AttentionPairBias(C_RES, C_PAIR, n_heads=N_HEADS)
    assert list(attn.norm_a.parameters()) == [], (
        "norm_a must have no learnable parameters (elementwise_affine=False); "
        "making it learnable causes DDP to crash because norm_a is never called "
        "when s is always provided"
    )


def test_attn_pair_bias_all_params_receive_gradients(
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Every learnable parameter in AttentionPairBias receives a finite gradient.

    Regression test for the DDP unused-parameter bug: ensures no parameter
    silently skips the forward pass when the module is called with real s.
    """
    attn = AttentionPairBias(C_RES, C_PAIR, n_heads=N_HEADS)
    torch.autograd.backward([scalar_sum(attn(s, t, z))])
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"parameter {name!r} has no gradient"
        assert torch.isfinite(param.grad).all(), f"parameter {name!r} has non-finite gradient"


def test_node_update_all_params_receive_gradients(
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Every learnable parameter in NodeUpdate receives a finite gradient.

    Regression test for the DDP unused-parameter bug.
    """
    node = NodeUpdate(C_RES, C_PAIR, n_heads=N_HEADS, dropout=0.0)
    torch.autograd.backward([scalar_sum(node(s, t, z))])
    for name, param in node.named_parameters():
        assert param.grad is not None, f"parameter {name!r} has no gradient"
        assert torch.isfinite(param.grad).all(), f"parameter {name!r} has non-finite gradient"


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
    (grad,) = torch.autograd.grad(scalar_sum(node_update(s_g, t, z)), s_g)
    assert torch.isfinite(grad).all()


def test_node_update_gradient_flows_to_t(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Gradients flow back through NodeUpdate to the time-conditioning input t."""
    t_g = t.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(scalar_sum(node_update(s, t_g, z)), t_g)
    assert torch.isfinite(grad).all()


def test_node_update_gradient_flows_to_z(
    node_update: NodeUpdate,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Gradients flow back through NodeUpdate to the pair embedding z (via the attention bias)."""
    z_g = z.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(scalar_sum(node_update(s, t, z_g)), z_g)
    assert torch.isfinite(grad).all()


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


def test_attn_pair_bias_all_masked_beta_no_nan(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Output is finite when beta suppresses every key position with -1e10.

    Guards against the softmax([-inf, ...]) = NaN failure that arises in float16
    mixed-precision when -1e10 overflows to -inf.
    """
    beta = torch.full((B, N_RES, N_RES), -1e10)
    with torch.no_grad():
        out = attn(s, t, z, beta=beta)
    assert torch.isfinite(out).all()


def test_attn_pair_bias_single_unmasked_neighbor_dominates(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Masking all keys except one materially changes the output from the all-open case."""
    beta_one_open = torch.full((B, N_RES, N_RES), -1e10)
    beta_one_open[:, :, 0] = 0.0
    beta_all_open = torch.zeros(B, N_RES, N_RES)
    with torch.no_grad():
        out_one = attn(s, t, z, beta=beta_one_open)
        out_all = attn(s, t, z, beta=beta_all_open)
    assert not torch.allclose(out_one, out_all)


def test_attn_pair_bias_beta_constant_shift_invariant(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Adding a uniform constant to all beta entries leaves output unchanged.

    Softmax is shift-invariant: exp(x+c)/sum(exp(x+c)) = exp(x)/sum(exp(x)).
    """
    beta_base = torch.randn(B, N_RES, N_RES)
    beta_shifted = beta_base + 5.0
    with torch.no_grad():
        out_base = attn(s, t, z, beta=beta_base)
        out_shifted = attn(s, t, z, beta=beta_shifted)
    assert torch.allclose(out_base, out_shifted, atol=1e-5)


def test_attn_pair_bias_sparse_equals_dense(
    attn: AttentionPairBias,
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Sparse path with neighbor_idx=[0..N-1] per query matches the dense path exactly."""
    # neighbor_idx[b, i, j] = j — each query sees all N_RES keys in order
    neighbor_idx: Int[torch.Tensor, "B N_res N_res"] = repeat(
        torch.arange(N_RES), "k -> b n k", b=B, n=N_RES
    )
    with torch.no_grad():
        out_dense = attn(s, t, z)
        out_sparse = attn(s, t, z, neighbor_idx=neighbor_idx)
    assert torch.allclose(out_dense, out_sparse, atol=1e-5)


def test_attn_pair_bias_zero_pair_bias_weights_no_effect(
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
) -> None:
    """With z_to_b.weight zeroed, different pair embeddings z produce identical output."""
    attn = AttentionPairBias(C_RES, C_PAIR, n_heads=N_HEADS).eval()
    torch.nn.init.zeros_(attn.z_to_b.weight)
    z1 = torch.randn(B, N_RES, N_RES, C_PAIR)
    z2 = torch.randn(B, N_RES, N_RES, C_PAIR) * 10.0
    with torch.no_grad():
        out1 = attn(s, t, z1)
        out2 = attn(s, t, z2)
    assert torch.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# AttentionPairBias — sparse path (K < N_res regression tests)
#
# Production crash: attn (B, n_heads, N, N) + pair_bias (B, n_heads, N, K)
# with N != K.  These tests use K_SPARSE < N_RES_LARGE to guarantee the
# mismatch that caused the RuntimeError.
# ---------------------------------------------------------------------------


@pytest.fixture
def attn_large() -> AttentionPairBias:
    """AttentionPairBias sized for the large-N sparse tests."""
    return AttentionPairBias(C_RES, C_PAIR, n_heads=N_HEADS).eval()


@pytest.fixture
def a_large() -> Float[torch.Tensor, "B N_res_large C_res"]:
    """Node embeddings for the large sparse scenario [B, N_RES_LARGE, C_RES]."""
    return torch.randn(B, N_RES_LARGE, C_RES)


@pytest.fixture
def s_large() -> Float[torch.Tensor, "B N_res_large C_res"]:
    """Conditioning embeddings for the large sparse scenario [B, N_RES_LARGE, C_RES]."""
    return torch.randn(B, N_RES_LARGE, C_RES)


@pytest.fixture
def z_sparse() -> Float[torch.Tensor, "B N_res_large K_sparse C_pair"]:
    """Sparse pair embeddings [B, N_RES_LARGE, K_SPARSE, C_PAIR] with K_SPARSE < N_RES_LARGE."""
    return torch.randn(B, N_RES_LARGE, K_SPARSE, C_PAIR)


@pytest.fixture
def neighbor_idx_sparse() -> Int[torch.Tensor, "B N_res_large K_sparse"]:
    """Neighbour index [B, N_RES_LARGE, K_SPARSE] — each node's K_SPARSE nearest neighbours."""
    idx = torch.zeros(N_RES_LARGE, K_SPARSE, dtype=torch.long)
    for i in range(N_RES_LARGE):
        neighbours = torch.arange(max(0, i - K_SPARSE // 2), max(K_SPARSE, i + K_SPARSE // 2 + 1))[
            :K_SPARSE
        ]
        idx[i] = neighbours.clamp(0, N_RES_LARGE - 1)
    return repeat(idx, "n k -> b n k", b=B)


@pytest.fixture
def beta_sparse() -> Float[torch.Tensor, "B N_res_large K_sparse"]:
    """Sparse attention bias [B, N_RES_LARGE, K_SPARSE] aligned with z_sparse."""
    return torch.zeros(B, N_RES_LARGE, K_SPARSE)


def test_attn_pair_bias_sparse_output_shape(
    attn_large: AttentionPairBias,
    a_large: Float[torch.Tensor, "B N_res_large C_res"],
    s_large: Float[torch.Tensor, "B N_res_large C_res"],
    z_sparse: Float[torch.Tensor, "B N_res_large K_sparse C_pair"],
    neighbor_idx_sparse: Int[torch.Tensor, "B N_res_large K_sparse"],
) -> None:
    """Sparse attention (K < N) returns the correct [B, N, C_res] shape without RuntimeError."""
    with torch.no_grad():
        out = attn_large(a_large, s_large, z_sparse, neighbor_idx=neighbor_idx_sparse)
    assert out.shape == (B, N_RES_LARGE, C_RES)


def test_attn_pair_bias_sparse_output_finite(
    attn_large: AttentionPairBias,
    a_large: Float[torch.Tensor, "B N_res_large C_res"],
    s_large: Float[torch.Tensor, "B N_res_large C_res"],
    z_sparse: Float[torch.Tensor, "B N_res_large K_sparse C_pair"],
    neighbor_idx_sparse: Int[torch.Tensor, "B N_res_large K_sparse"],
) -> None:
    """Sparse attention output contains no NaN or Inf values."""
    with torch.no_grad():
        out = attn_large(a_large, s_large, z_sparse, neighbor_idx=neighbor_idx_sparse)
    assert torch.isfinite(out).all()


def test_attn_pair_bias_sparse_with_beta(
    attn_large: AttentionPairBias,
    a_large: Float[torch.Tensor, "B N_res_large C_res"],
    s_large: Float[torch.Tensor, "B N_res_large C_res"],
    z_sparse: Float[torch.Tensor, "B N_res_large K_sparse C_pair"],
    beta_sparse: Float[torch.Tensor, "B N_res_large K_sparse"],
    neighbor_idx_sparse: Int[torch.Tensor, "B N_res_large K_sparse"],
) -> None:
    """Sparse attention works when both beta and neighbor_idx are provided."""
    with torch.no_grad():
        out = attn_large(
            a_large, s_large, z_sparse, beta=beta_sparse, neighbor_idx=neighbor_idx_sparse
        )
    assert out.shape == (B, N_RES_LARGE, C_RES)
    assert torch.isfinite(out).all()


def test_attn_pair_bias_sparse_gradient_flows(
    attn_large: AttentionPairBias,
    a_large: Float[torch.Tensor, "B N_res_large C_res"],
    s_large: Float[torch.Tensor, "B N_res_large C_res"],
    z_sparse: Float[torch.Tensor, "B N_res_large K_sparse C_pair"],
    neighbor_idx_sparse: Int[torch.Tensor, "B N_res_large K_sparse"],
) -> None:
    """Gradients flow through sparse attention back to the node embedding input."""
    a_g = a_large.clone().requires_grad_(True)
    out = attn_large(a_g, s_large, z_sparse, neighbor_idx=neighbor_idx_sparse)
    (grad,) = torch.autograd.grad(reduce(out, "b n c -> ", "sum"), a_g)
    assert torch.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_adaln_forward_wrong_shape() -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers TypeCheckError."""
    adaln = AdaLN(c_a=C_RES, c_s=C_RES).eval()
    a_bad = torch.zeros(B, N_RES)  # missing c_a dim
    s_good = torch.zeros(B, N_RES, C_RES)
    with pytest.raises(TypeCheckError):
        adaln(a_bad, s_good)


def test_attention_pair_bias_forward_wrong_shape(
    attn: AttentionPairBias,
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers TypeCheckError."""
    a_bad = torch.zeros(B, N_RES)  # missing c_res dim
    with pytest.raises(TypeCheckError):
        attn(a_bad, None, z)


def test_node_update_forward_wrong_shape(
    node_update: NodeUpdate,
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Wrong s ndim (2-D instead of 3-D) triggers TypeCheckError."""
    s_bad = torch.zeros(B, N_RES)  # missing c_res dim
    with pytest.raises(TypeCheckError):
        node_update(s_bad, t, z)


# ---------------------------------------------------------------------------
# AttentionPairBias — stability (residual gain, recycling, gradient scaling)
# ---------------------------------------------------------------------------


def test_attn_pair_bias_residual_gain_contractive(
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """The attention delta is contractive at random init: ||delta|| / ||s|| < 1.0.

    AttentionPairBias.forward returns the additive update that NodeUpdate applies as
    s = s + attn(s, t, z).  A gain >= 1.0 at init predicts compounding norm explosions
    across 8 decoder recycling blocks in late training.
    """
    attn = AttentionPairBias(C_RES, C_PAIR, n_heads=N_HEADS).eval()
    with torch.no_grad():
        delta = attn(s, t, z)
    delta_norm = float(torch.sqrt(reduce(delta**2, "... -> ", "sum")))
    s_norm = float(torch.sqrt(reduce(s**2, "... -> ", "sum")))
    gain = delta_norm / s_norm
    assert gain < 1.0


def test_attn_pair_bias_repeated_application_bounded(
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """RMS stays bounded after 20 rounds of s = s + attn(s, t, z) with z held fixed.

    Simulates the decoder recycling loop.  An intrinsically expansive operator shows
    exponential RMS growth even without training; a contractive one stays near the
    initial scale.  Threshold 5.0 allows healthy growth while catching explosions.
    """
    attn = AttentionPairBias(C_RES, C_PAIR, n_heads=N_HEADS).eval()
    s_cycle = s.clone()
    rms_initial = float(torch.sqrt(reduce(s_cycle**2, "... -> ", "mean")))
    with torch.no_grad():
        for _ in range(20):
            s_cycle = s_cycle + attn(s_cycle, t, z)
    rms_final = float(torch.sqrt(reduce(s_cycle**2, "... -> ", "mean")))
    assert rms_final / rms_initial < RMS_GAIN


def test_attn_pair_bias_param_grads_finite_after_recycling(
    s: Float[torch.Tensor, "B N_res C_res"],
    t: Float[torch.Tensor, "B N_res C_res"],
    z: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """All parameter gradients remain finite after 20 recycling steps.

    An expansive operator causes gradient norms to grow with recycling depth via
    backpropagation-through-time.  Finite gradients after the full 20-step unrolled
    loop is a necessary condition for stable training in an 8-block decoder.
    """
    attn = AttentionPairBias(C_RES, C_PAIR, n_heads=N_HEADS)
    s_cycle = s.clone()
    for _ in range(20):
        s_cycle = s_cycle + attn(s_cycle, t, z)
    loss = reduce(s_cycle**2, "... -> ", "mean")
    torch.autograd.backward([loss])
    for name, param in attn.named_parameters():
        assert param.grad is not None, f"param {name!r} has no gradient after recycling"
        assert torch.isfinite(
            param.grad
        ).all(), f"param {name!r} has non-finite gradient after 20 recycling steps"

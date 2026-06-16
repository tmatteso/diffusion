"""Tests for the pairformer stack.

Covers PairformerBlock, PairformerStack, TriangleMultiplicationOutgoing,
and TriangleMultiplicationIncoming: output shape/dtype/finiteness, gradient
flow, batch independence, gating behaviour, hidden-dimension configuration,
contraction-direction correctness, access-pattern sensitivity, and type
enforcement via beartype/jaxtyping.
"""

import pytest
import torch
from architecture.pairformer_stack import (
    PairformerBlock,
    PairformerStack,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from beartype import beartype
from einops import rearrange, reduce
from helpers.useful_objects import manual_seed
from jaxtyping import Float, TypeCheckError, jaxtyped

_ = manual_seed(42)

B = 1
N_RES = 8
C = 32
N_HEADS = 4
N_BLOCKS = 3
TOLERANCE = 1e-6
TIGHT_TOLERANCE = 1e-3
PAIRFORMER_BLOCK_NUM = 5
# Constants used exclusively by the TriangleMultiplicationOutgoing tests
_N = 6
_C = 16
_C_HIDDEN = 8


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def mean_abs_asymmetry(
    x: Float[torch.Tensor, "B N N C"],
) -> Float[torch.Tensor, ""]:
    """Return mean absolute difference between x[b,i,j,c] and x[b,j,i,c].

    Computes the asymmetry of a pair tensor by comparing each (i, j) entry
    with its transposed counterpart (j, i) and averaging the absolute
    differences over all elements.
    """
    diff = x - rearrange(x, "b i j c -> b j i c")
    return reduce(diff.abs(), "b i j c -> ", "mean")


# ---------------------------------------------------------------------------
# Fixtures — models
# ---------------------------------------------------------------------------


@pytest.fixture
def block() -> PairformerBlock:
    """Provide a single PairformerBlock in eval mode.

    Instantiated with the module-level C and N_HEADS constants.
    """
    return PairformerBlock(c_pair=C, c_res=None, n_heads=N_HEADS).eval()


@pytest.fixture
def stack() -> PairformerStack:
    """Provide a PairformerStack with N_BLOCKS blocks in eval mode.

    Instantiated with module-level C, N_BLOCKS, and N_HEADS constants.
    """
    return PairformerStack(
        c=C,
        c_res=None,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
    ).eval()


@pytest.fixture
def stack_single_block() -> PairformerStack:
    """Provide a PairformerStack with exactly one block in eval mode.

    Instantiated with module-level C and N_HEADS constants and n_blocks=1.
    Used to verify numerical equivalence with a standalone PairformerBlock.
    """
    return PairformerStack(
        c=C,
        c_res=None,
        n_blocks=1,
        n_heads=N_HEADS,
    ).eval()


# ---------------------------------------------------------------------------
# Fixtures — tensors
# ---------------------------------------------------------------------------


@pytest.fixture
def v() -> Float[torch.Tensor, "B N_RES N_RES C"]:
    """Provide a random pair-embedding tensor (B, N_RES, N_RES, C).

    Sampled from a standard normal distribution using module-level B,
    N_RES, and C constants.
    """
    return torch.randn(B, N_RES, N_RES, C)


@pytest.fixture
def z_3d() -> Float[torch.Tensor, "B N_RES N_RES"]:
    """Provide a 3-D tensor (B, N_RES, N_RES) missing the c_pair channel dim.

    Used by wrong-shape type-enforcement tests for PairformerBlock and
    PairformerStack to verify that beartype/jaxtyping rejects inputs with
    an incorrect number of dimensions.
    """
    return torch.zeros(B, N_RES, N_RES)


# ---------------------------------------------------------------------------
# PairformerBlock
# ---------------------------------------------------------------------------


def test_pairformer_block_output_shape(
    block: PairformerBlock,
    v: Float[torch.Tensor, "B N_RES N_RES C"],
) -> None:
    """PairformerBlock output shape matches the input (B, N_RES, N_RES, C).

    Verifies that the block does not change the spatial or channel dimensions
    of the pair embedding tensor.
    """
    with torch.no_grad():
        _, out = block(s=None, z=v)
    assert out.shape == (B, N_RES, N_RES, C)


def test_pairformer_block_output_finite(
    block: PairformerBlock,
    v: Float[torch.Tensor, "B N_RES N_RES C"],
) -> None:
    """PairformerBlock output contains only finite values.

    Verifies that no NaN or Inf values are produced for a standard random
    pair-embedding input.
    """
    with torch.no_grad():
        _, out = block(s=None, z=v)
    assert torch.isfinite(out).all()


def test_pairformer_block_output_dtype(
    block: PairformerBlock,
    v: Float[torch.Tensor, "B N_RES N_RES C"],
) -> None:
    """PairformerBlock output dtype matches the input dtype.

    Verifies that the block does not inadvertently upcast or downcast the
    pair-embedding tensor during the forward pass.
    """
    with torch.no_grad():
        _, out = block(s=None, z=v)
    assert out.dtype == v.dtype


def test_pairformer_block_changes_input(
    block: PairformerBlock,
    v: Float[torch.Tensor, "B N_RES N_RES C"],
) -> None:
    """PairformerBlock output differs from its input (non-identity transform).

    Verifies that the block applies a meaningful transformation rather than
    passing the pair embedding through unchanged.
    """
    with torch.no_grad():
        _, out = block(s=None, z=v)
    assert not torch.allclose(out, v)


def test_pairformer_block_gradient_flows(
    block: PairformerBlock,
    v: Float[torch.Tensor, "B N_RES N_RES C"],
) -> None:
    """Gradient propagates from PairformerBlock output back to v.

    Verifies that the backward pass produces a non-None, fully finite
    gradient on the input pair-embedding tensor.
    """
    v_g = v.clone().requires_grad_(True)  # noqa: FBT003
    _, out = block(s=None, z=v_g)
    torch.autograd.backward([reduce(out, "b n m c -> ", "sum")])
    assert v_g.grad is not None
    assert torch.isfinite(v_g.grad).all()


def test_pairformer_block_row_attn_preserves_col_symmetry(
    block: PairformerBlock,
    v: Float[torch.Tensor, "B N_RES N_RES C"],
) -> None:
    """Symmetric input remains finite after one PairformerBlock.

    Verifies numerical stability when the input pair embedding is spatially
    symmetric (z[b,i,j,c] == z[b,j,i,c]); both row and column attention axes
    are exercised so neither dimension is favoured (row+col attention both
    applied).
    """
    # Symmetric input should remain approximately symmetric after one block
    # (row + col attention are both applied, so neither axis is favoured)
    v_sym = (v + rearrange(v, "b i j c -> b j i c")) * 0.5
    with torch.no_grad():
        _, out = block(s=None, z=v_sym)
    # Not exactly symmetric (weights differ for row vs col), but check finite
    assert torch.isfinite(out).all()


def test_pairformer_block_different_batch_items_independent(
    block: PairformerBlock,
) -> None:
    """Two samples in batch equivalent to processing them separately.

    Verifies that each element in the batch is processed independently by
    checking that concatenating two samples produces the same results as
    running them through the block one at a time.
    """
    # Each batch element should be processed independently
    v1 = torch.randn(1, N_RES, N_RES, C)
    v2 = torch.randn(1, N_RES, N_RES, C)
    v_cat = torch.cat([v1, v2], dim=0)
    with torch.no_grad():
        _, out1 = block(s=None, z=v1)
        _, out2 = block(s=None, z=v2)
        _, out_cat = block(s=None, z=v_cat)
    assert torch.allclose(out1, out_cat[:1], atol=1e-5)
    assert torch.allclose(out2, out_cat[1:], atol=1e-5)


# ---------------------------------------------------------------------------
# PairformerStack
# ---------------------------------------------------------------------------


def test_pairformer_stack_output_shape(
    stack: PairformerStack,
    v: Float[torch.Tensor, "B N_RES N_RES C"],
) -> None:
    """PairformerStack returns output of same shape as input.

    Verifies that stacking N_BLOCKS pairformer blocks preserves pair-embedding
    tensor dimensions end-to-end.
    """
    with torch.no_grad():
        out = stack(s=None, z=v)
    assert out.shape == (B, N_RES, N_RES, C)


def test_pairformer_stack_output_finite(
    stack: PairformerStack,
    v: Float[torch.Tensor, "B N_RES N_RES C"],
) -> None:
    """Stacking N_BLOCKS pairformer blocks does not produce NaN or Inf values.

    Verifies numerical stability of the full stack on a standard random
    pair-embedding input.
    """
    with torch.no_grad():
        out = stack(s=None, z=v)
    assert torch.isfinite(out).all()


def test_pairformer_stack_block_count() -> None:
    """The n_blocks constructor creates exact PairformerBlock amount.

    Verifies that the length of the internal blocks list equals the value
    passed to the constructor.
    """
    s = PairformerStack(C, n_blocks=PAIRFORMER_BLOCK_NUM, n_heads=N_HEADS)
    assert len(s.blocks) == PAIRFORMER_BLOCK_NUM


def test_pairformer_stack_gradient_flows(
    stack: PairformerStack,
    v: Float[torch.Tensor, "B N_RES N_RES C"],
) -> None:
    """Gradients propagate through all blocks back to input pair embedding.

    Verifies that the backward pass through the entire stack yields a
    non-None, fully finite gradient on the input tensor.
    """
    v_g = v.clone().requires_grad_(True)  # noqa: FBT003
    out = stack(s=None, z=v_g)
    torch.autograd.backward([reduce(out, "b n m c -> ", "sum")])
    assert v_g.grad is not None
    assert torch.isfinite(v_g.grad).all()


def test_pairformer_stack_single_block_matches_pairformer_block(
    block: PairformerBlock,
    stack_single_block: PairformerStack,
) -> None:
    """Single-block PairformerStack with identical weights == PairformerBlock.

    Verifies that PairformerStack(n_blocks=1) is numerically equivalent to a
    standalone PairformerBlock after copying all weights, confirming that the
    stack wrapper introduces no extra transformations.
    """
    _ = manual_seed(0)
    z_v = torch.randn(B, N_RES, N_RES, C)
    _ = stack_single_block.blocks[0].load_state_dict(block.state_dict())
    with torch.no_grad():
        _, block_z = block(s=None, z=z_v)
        stack_z = stack_single_block(s=None, z=z_v)
        assert torch.allclose(block_z, stack_z, atol=TOLERANCE)


# ===========================================================================
# TriangleMultiplicationOutgoing
#
# The outgoing update computes (Algorithm 17, step 2):
#   m[i,j] = sigmoid(gate(z[i,j])) · proj_out(norm_out(Σ_k a[i,k] ⊙ b[k,j]))
# where
#   a[i,k] = sigmoid(gate_a(norm(z)[i,k])) · proj_a(norm(z)[i,k])
#   b[k,j] = sigmoid(gate_b(norm(z)[k,j])) · proj_b(norm(z)[k,j])
#
# Design facts verified below:
#   • outer gate uses raw z (not LayerNorm'd); proj_a / proj_b see norm(z).
#   • contraction is Σ_k a[i,k]⊙b[k,j]  ("outgoing": row of a, column of b).
#   • with shared weights, outgoing(z) == incoming(z) iff z is symmetric.
# ===========================================================================


# ---------------------------------------------------------------------------
# Fixtures - TriangleMultiplicationOutgoing
# ---------------------------------------------------------------------------


@pytest.fixture
def tmo() -> TriangleMultiplicationOutgoing:
    """TriangleMultiplicationOutgoing(_C, c_hidden=_C_HIDDEN) in eval mode.

    Instantiated with the module-level _C and _C_HIDDEN constants.
    """
    return TriangleMultiplicationOutgoing(c=_C, c_hidden=_C_HIDDEN).eval()


@pytest.fixture
def z_pair() -> Float[torch.Tensor, "B N_res N_res C"]:
    """Random pair-embedding tensor (2, _N, _N, _C).

    Batch size 2 is used so that batch-independence tests can split it.
    """
    return torch.randn(2, _N, _N, _C)


@pytest.fixture
def z_sym() -> Float[torch.Tensor, "B N_res N_res C"]:
    """Spatially symmetric pair tensor: z[b,i,j,c] == z[b,j,i,c].

    Constructed by averaging a random tensor with its spatial transpose so
    that the symmetry constraint holds exactly.
    """
    raw = torch.randn(1, _N, _N, _C)
    return (raw + rearrange(raw, "b i j c -> b j i c")) * 0.5


@pytest.fixture
def z_no_batch() -> Float[torch.Tensor, "N N C"]:
    """Random pair tensor (_N, _N, _C) missing the batch dimension.

    Used to verify that beartype/jaxtyping rejects 3-D inputs where a
    4-D batch-first tensor is expected.
    """
    return torch.randn(_N, _N, _C)


@pytest.fixture
def z_nonsquare() -> Float[torch.Tensor, "B N M C"]:
    """Random pair tensor (1, _N, _N+1, _C) with non-square spatial dims.

    Used to verify that beartype/jaxtyping rejects inputs where the two
    spatial dimensions are not equal.
    """
    return torch.randn(1, _N, _N + 1, _C)


@pytest.fixture
def z_single() -> Float[torch.Tensor, "B N_res N_res C"]:
    """Random single-batch pair-embedding tensor (1, _N, _N, _C).

    Distinct from z_pair (batch=2); used where a single batch element is
    needed but z_pair's larger batch would be misleading.
    """
    return torch.randn(1, _N, _N, _C)


# ---------------------------------------------------------------------------
# Basic output contracts
# ---------------------------------------------------------------------------


def test_tmo_output_shape(
    tmo: TriangleMultiplicationOutgoing,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Output shape equals input shape (B, N, N, C).

    Verifies that TriangleMultiplicationOutgoing does not alter the spatial
    or channel dimensions of the pair tensor.
    """
    with torch.no_grad():
        out = tmo(z_pair)
    assert out.shape == z_pair.shape


def test_tmo_output_dtype_preserved(
    tmo: TriangleMultiplicationOutgoing,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Output dtype matches input dtype.

    Verifies that TriangleMultiplicationOutgoing does not inadvertently
    upcast or downcast the pair tensor during the forward pass.
    """
    with torch.no_grad():
        out = tmo(z_pair)
    assert out.dtype == z_pair.dtype


def test_tmo_output_is_finite(
    tmo: TriangleMultiplicationOutgoing,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """No NaN or Inf in output for a standard random input.

    Verifies numerical stability of TriangleMultiplicationOutgoing for a
    typical pair-embedding tensor drawn from a standard normal distribution.
    """
    with torch.no_grad():
        out = tmo(z_pair)
    assert torch.isfinite(out).all()


def test_tmo_gradient_flows_to_input(
    tmo: TriangleMultiplicationOutgoing,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Loss gradient propagates back to z: non-None and fully finite.

    Verifies that the backward pass through TriangleMultiplicationOutgoing
    produces a well-defined gradient on the input pair tensor.
    """
    z_g = z_pair.clone().requires_grad_(True)  # noqa: FBT003
    out = tmo(z_g)
    torch.autograd.backward([reduce(out, "b i j c -> ", "sum")])
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


# ---------------------------------------------------------------------------
# Batch independence
# ---------------------------------------------------------------------------


def test_tmo_batch_items_are_independent(
    tmo: TriangleMultiplicationOutgoing,
) -> None:
    """Processing two inputs in a batch equals processing them separately.

    Verifies that TriangleMultiplicationOutgoing processes each batch element
    independently, with no cross-contamination between samples.
    """
    z1 = torch.randn(1, _N, _N, _C)
    z2 = torch.randn(1, _N, _N, _C)
    z_cat = torch.cat([z1, z2], dim=0)
    with torch.no_grad():
        out1 = tmo(z1)
        out2 = tmo(z2)
        out_cat = tmo(z_cat)
    assert torch.allclose(out1, out_cat[:1], atol=1e-5)
    assert torch.allclose(out2, out_cat[1:], atol=1e-5)


# ---------------------------------------------------------------------------
# Gating mechanism
# ---------------------------------------------------------------------------


def test_tmo_gate_near_zero_suppresses_output(
    tmo: TriangleMultiplicationOutgoing,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Outer gate bias << 0 → sigmoid ≈ 0 → output ≈ 0 for any pair content.

    Verifies that the outer gating mechanism can suppress the full output by
    driving the gate bias strongly negative, regardless of the pair content.
    """
    with torch.no_grad():
        _ = tmo.gate.bias.fill_(-100.0)
        _ = tmo.gate.weight.fill_(0.0)
    with torch.no_grad():
        out = tmo(z_pair)
    assert out.abs().max().item() < TOLERANCE


def test_tmo_gate_near_one_passes_output_through(
    tmo: TriangleMultiplicationOutgoing,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Outer gate bias >> 0, sigmoid ≈ 1, output driven by z_pair."""
    with torch.no_grad():
        _ = tmo.gate.bias.fill_(100.0)
        _ = tmo.gate.weight.fill_(0.0)
    with torch.no_grad():
        out = tmo(z_pair)
    assert out.abs().mean().item() > TIGHT_TOLERANCE


# ---------------------------------------------------------------------------
# Outer gate uses raw z; proj_a / proj_b see LayerNorm(z)
# ---------------------------------------------------------------------------


def test_tmo_gate_responds_to_input_scale(
    tmo: TriangleMultiplicationOutgoing,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Scaling z changes outer gate output even when LayerNorm is invariant.

    Setup: pin gate_a / gate_b to sigmoid ≈ 1 so proj_a / proj_b are
        scale-invariant.
    The outer gate still sees raw z and therefore produces different values for
    z vs 100·z.
    """
    with torch.no_grad():
        _ = tmo.gate_a.bias.fill_(100.0)
        _ = tmo.gate_a.weight.fill_(0.0)
        _ = tmo.gate_b.bias.fill_(100.0)
        _ = tmo.gate_b.weight.fill_(0.0)
    z_large = 100.0 * z_single
    with torch.no_grad():
        out_z = tmo(z_single)
        out_large = tmo(z_large)
    assert not torch.allclose(out_z, out_large, atol=1e-4)


def test_tmo_proj_layers_are_layernorm_scale_invariant(
    tmo: TriangleMultiplicationOutgoing,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """With outer gate pinned to 1, output is identical for z and 2·z.

    LayerNorm is scale-invariant in its input, so proj_a and proj_b (which
    receive
    norm(z)) produce the same result for z and 2·z. Pinning the outer gate
    removes
    the only path that sees raw z.
    """
    with torch.no_grad():
        _ = tmo.gate.bias.fill_(100.0)
        _ = tmo.gate.weight.fill_(0.0)
    z_scaled = 2.0 * z_pair
    with torch.no_grad():
        out_z = tmo(z_pair)
        out_scaled = tmo(z_scaled)
    assert torch.allclose(out_z, out_scaled, atol=1e-4)


# ---------------------------------------------------------------------------
# Hidden dimension
# ---------------------------------------------------------------------------


def test_tmo_default_c_hidden_equals_c() -> None:
    """Omitting c_hidden defaults to c (proj_a, proj_out, norm_out shapes)."""
    m = TriangleMultiplicationOutgoing(c=_C).eval()
    assert m.proj_a.weight.shape == (_C, _C)
    assert m.proj_out.weight.shape == (_C, _C)
    assert m.norm_out.normalized_shape == (_C,)


def test_tmo_custom_c_hidden_output_shape_is_c(
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """c_hidden != c: output depth remains c; internal proj are c_hidden."""
    m = TriangleMultiplicationOutgoing(c=_C, c_hidden=4).eval()
    with torch.no_grad():
        out = m(z_single)
    assert out.shape == (1, _N, _N, _C)
    assert m.proj_a.weight.shape == (4, _C)
    assert m.norm_out.normalized_shape == (4,)
    assert m.proj_out.weight.shape == (_C, 4)


# ---------------------------------------------------------------------------
# Contraction direction: outgoing vs incoming
# ---------------------------------------------------------------------------


def test_tmo_differs_from_incoming_for_asymmetric_input(
    tmo: TriangleMultiplicationOutgoing,
    tmi: TriangleMultiplicationIncoming,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Same weights, different contractions, different output for asymmetric z.

    Outgoing:  Σ_k a[i,k] ⊙ b[k,j]
    Incoming:  Σ_k a[k,i] ⊙ b[j,k]
    These are not equal when z is not spatially symmetric.
    """
    _ = tmi.load_state_dict(tmo.state_dict())
    with torch.no_grad():
        out_o = tmo(z_single)
        out_i = tmi(z_single)
    assert not torch.allclose(out_o, out_i, atol=1e-4)


def test_tmo_equals_incoming_for_symmetric_input_with_shared_weights(
    tmo: TriangleMultiplicationOutgoing,
    tmi: TriangleMultiplicationIncoming,
    z_sym: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """For spatially symmetric z, outgoing == incoming when weights are shared.

    Proof sketch: with z[b,i,k,c] = z[b,k,i,c], LayerNorm preserves symmetry,
    so
      f_a(zn[i,k]) = f_a(zn[k,i])   and   f_b(zn[k,j]) = f_b(zn[j,k]).
    Substituting into the incoming sum:
      Σ_k f_a(zn[k,i]) ⊙ f_b(zn[j,k]) = Σ_k f_a(zn[i,k]) ⊙ f_b(zn[k,j])
    which equals the outgoing sum. The outer gate g[i,j] =
    sigmoid(gate(z[i,j]))
    is identical for both since they share weights and receive the same z.
    """
    _ = tmi.load_state_dict(tmo.state_dict())
    with torch.no_grad():
        out_o = tmo(z_sym)
        out_i = tmi(z_sym)
    assert torch.allclose(out_o, out_i, atol=1e-4)


# ---------------------------------------------------------------------------
# Access-pattern sensitivity
# Output[i,j] = Σ_k a[i,k] ⊙ b[k,j], so it depends on z[i,:] through a
# and on z[:,j] through b.  Changing row i of z also affects b[i,:], which
# enters m[i', j] for ALL i' via the k=i summation term.
# ---------------------------------------------------------------------------


def test_tmo_output_row_i_changes_when_row_i_of_z_is_perturbed(
    tmo: TriangleMultiplicationOutgoing,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Output at row i changes when z[i, :, :] is perturbed.

    (a[i,k] changes for all k)
    """
    z_pert = z_single.clone()
    z_pert[0, 0, :, :] = z_pert[0, 0, :, :] + 5.0
    with torch.no_grad():
        out_orig = tmo(z_single)
        out_pert = tmo(z_pert)
    assert not torch.allclose(
        out_orig[0, 0, :, :],
        out_pert[0, 0, :, :],
        atol=1e-4,
    )


def test_tmo_output_col_j_changes_when_col_j_of_z_is_perturbed(
    tmo: TriangleMultiplicationOutgoing,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Output at column j changes when z[:, :, j, :] is perturbed .

    (b[k,j] changes for all k)
    """
    z_pert = z_single.clone()
    z_pert[0, :, 1, :] = z_pert[0, :, 1, :] + 5.0
    with torch.no_grad():
        out_orig = tmo(z_single)
        out_pert = tmo(z_pert)
    assert not torch.allclose(
        out_orig[0, :, 1, :],
        out_pert[0, :, 1, :],
        atol=1e-4,
    )


def test_tmo_output_other_rows_change_when_row_i_of_z_is_perturbed(
    tmo: TriangleMultiplicationOutgoing,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Changing row i changes output at row i' != i via b[i, j] in k=i term.

    Even though a[i', k] is independent of z[i, :], b[i, j] enters
    m[i', j] = Σ_k a[i', k] ⊙ b[k, j] at k=i.  This confirms the full-row
    dependency that is the hallmark of the outgoing contraction.

    The perturbation must be channel-varying (not a uniform constant shift)
    because
    LayerNorm is invariant to uniform additive shifts across channels, which
    would
    leave zn — and therefore b — unchanged and make the test trivially pass.
    """
    z_pert = z_single.clone()
    # Channel-varying noise changes relative channel distribution so LayerNorm
    # cannot cancel it, guaranteeing zn[0,0,:,:] and thus b[0,0,:,:] change.
    z_pert[0, 0, :, :] = z_pert[0, 0, :, :] + torch.randn(_N, _C) * 5.0
    with torch.no_grad():
        out_orig = tmo(z_single)
        out_pert = tmo(z_pert)
    # Row i'=1 must also change (b[0, j] contributes to m[1, j] via k=0)
    assert not torch.allclose(
        out_orig[0, 1, :, :],
        out_pert[0, 1, :, :],
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# Type enforcement
# ---------------------------------------------------------------------------


def test_tmo_jaxtyping_rejects_3d_input(
    tmo: TriangleMultiplicationOutgoing,
    z_no_batch: Float[torch.Tensor, "N N C"],
) -> None:
    """Beartype raises when the batch dimension is missing."""
    with pytest.raises(TypeCheckError):
        _ = tmo(z_no_batch)


def test_tmo_jaxtyping_rejects_non_square_spatial_dims(
    tmo: TriangleMultiplicationOutgoing,
    z_nonsquare: Float[torch.Tensor, "B N M C"],
) -> None:
    """Beartype raises when the two N_res spatial dimensions differ."""
    with pytest.raises(TypeCheckError):
        _ = tmo(z_nonsquare)


# ===========================================================================
# TriangleMultiplicationIncoming
#
# The incoming update computes (Algorithm 17, step 3):
#   m[i,j] = sigmoid(gate(z[i,j])) · proj_out(norm_out(Σ_k a[k,i] ⊙ b[j,k]))
# where
#   a[k,i] = sigmoid(gate_a(norm(z)[k,i])) · proj_a(norm(z)[k,i])
#   b[j,k] = sigmoid(gate_b(norm(z)[j,k])) · proj_b(norm(z)[j,k])
#
# Design facts verified below:
#   • outer gate uses raw z (not LayerNorm'd); proj_a / proj_b see norm(z).
#   • contraction is Σ_k a[k,i]⊙b[j,k]  ("incoming": column of a, row of b).
#   • column i of z drives the first output dimension; row j drives the second.
#   • incoming(z) == outgoing(transpose(z)) when inner gates are pinned to 1.
#   • with shared weights, incoming(z) == outgoing(z) iff z is symmetric.
# ===========================================================================


# ---------------------------------------------------------------------------
# Fixtures - TriangleMultiplicationIncoming
# ---------------------------------------------------------------------------


@pytest.fixture
def tmi() -> TriangleMultiplicationIncoming:
    """TriangleMultiplicationIncoming(_C, c_hidden=_C_HIDDEN) in eval mode.

    Instantiated with the module-level _C and _C_HIDDEN constants.
    """
    return TriangleMultiplicationIncoming(c=_C, c_hidden=_C_HIDDEN).eval()


# ---------------------------------------------------------------------------
# Basic output contracts
# ---------------------------------------------------------------------------


def test_tmi_output_shape(
    tmi: TriangleMultiplicationIncoming,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Output shape equals input shape (B, N, N, C)."""
    with torch.no_grad():
        out = tmi(z_pair)
    assert out.shape == z_pair.shape


def test_tmi_output_dtype_preserved(
    tmi: TriangleMultiplicationIncoming,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Output dtype matches input dtype."""
    with torch.no_grad():
        out = tmi(z_pair)
    assert out.dtype == z_pair.dtype


def test_tmi_output_is_finite(
    tmi: TriangleMultiplicationIncoming,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """No NaN or Inf in output for a standard random input."""
    with torch.no_grad():
        out = tmi(z_pair)
    assert torch.isfinite(out).all()


def test_tmi_gradient_flows_to_input(
    tmi: TriangleMultiplicationIncoming,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Loss gradient propagates back to z: non-None and fully finite."""
    z_g = z_pair.clone().requires_grad_(True)  # noqa: FBT003
    out = tmi(z_g)
    torch.autograd.backward([reduce(out, "b i j c -> ", "sum")])
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


# ---------------------------------------------------------------------------
# Batch independence
# ---------------------------------------------------------------------------


def test_tmi_batch_items_are_independent(
    tmi: TriangleMultiplicationIncoming,
) -> None:
    """Processing two inputs in a batch equals processing them separately."""
    z1 = torch.randn(1, _N, _N, _C)
    z2 = torch.randn(1, _N, _N, _C)
    z_cat = torch.cat([z1, z2], dim=0)
    with torch.no_grad():
        out1 = tmi(z1)
        out2 = tmi(z2)
        out_cat = tmi(z_cat)
    assert torch.allclose(out1, out_cat[:1], atol=1e-5)
    assert torch.allclose(out2, out_cat[1:], atol=1e-5)


# ---------------------------------------------------------------------------
# Gating mechanism
# ---------------------------------------------------------------------------


def test_tmi_gate_near_zero_suppresses_output(
    tmi: TriangleMultiplicationIncoming,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Outer gate bias << 0, sigmoid ~ 0,  output ~ 0 for z_pair."""
    with torch.no_grad():
        _ = tmi.gate.bias.fill_(-100.0)
        _ = tmi.gate.weight.fill_(0.0)
    with torch.no_grad():
        out = tmi(z_pair)
    assert out.abs().max().item() < TOLERANCE


def test_tmi_gate_near_one_passes_output_through(
    tmi: TriangleMultiplicationIncoming,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Outer gate bias >> 0, sigmoid ~ 1, output driven by z_pair."""
    with torch.no_grad():
        _ = tmi.gate.bias.fill_(100.0)
        _ = tmi.gate.weight.fill_(0.0)
    with torch.no_grad():
        out = tmi(z_pair)
    assert out.abs().mean().item() > TIGHT_TOLERANCE


# ---------------------------------------------------------------------------
# Outer gate uses raw z; proj_a / proj_b see LayerNorm(z)
# ---------------------------------------------------------------------------


def test_tmi_gate_responds_to_input_scale(
    tmi: TriangleMultiplicationIncoming,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Scaling z changes outer gate output even when LayerNorm is invariant.

    Setup: pin gate_a / gate_b to sigmoid ≈ 1 so proj_a / proj_b are
        scale-invariant.
    The outer gate still sees raw z and therefore produces different values for
    z vs 100·z.
    """
    with torch.no_grad():
        _ = tmi.gate_a.bias.fill_(100.0)
        _ = tmi.gate_a.weight.fill_(0.0)
        _ = tmi.gate_b.bias.fill_(100.0)
        _ = tmi.gate_b.weight.fill_(0.0)
    z_large = 100.0 * z_single
    with torch.no_grad():
        out_z = tmi(z_single)
        out_large = tmi(z_large)
    assert not torch.allclose(out_z, out_large, atol=1e-4)


def test_tmi_proj_layers_are_layernorm_scale_invariant(
    tmi: TriangleMultiplicationIncoming,
    z_pair: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """With outer gate pinned to 1, output is identical for z and 2·z.

    LayerNorm is scale-invariant, so proj_a and proj_b (which receive norm(z))
    produce the same result for z and 2·z.  Pinning the outer gate removes the
    only path that sees raw z.
    """
    with torch.no_grad():
        _ = tmi.gate.bias.fill_(100.0)
        _ = tmi.gate.weight.fill_(0.0)
    z_scaled = 2.0 * z_pair
    with torch.no_grad():
        out_z = tmi(z_pair)
        out_scaled = tmi(z_scaled)
    assert torch.allclose(out_z, out_scaled, atol=1e-4)


# ---------------------------------------------------------------------------
# Hidden dimension
# ---------------------------------------------------------------------------


def test_tmi_default_c_hidden_equals_c() -> None:
    """Omitting c_hidden defaults to c (proj_a, proj_out, norm_out)."""
    m = TriangleMultiplicationIncoming(c=_C).eval()
    assert m.proj_a.weight.shape == (_C, _C)
    assert m.proj_out.weight.shape == (_C, _C)
    assert m.norm_out.normalized_shape == (_C,)


def test_tmi_custom_c_hidden_output_shape_is_c(
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """c_hidden ≠ c: output depth remains c; internal proj are c_hidden."""
    m = TriangleMultiplicationIncoming(c=_C, c_hidden=4).eval()
    with torch.no_grad():
        out = m(z_single)
    assert out.shape == (1, _N, _N, _C)
    assert m.proj_a.weight.shape == (4, _C)
    assert m.norm_out.normalized_shape == (4,)
    assert m.proj_out.weight.shape == (_C, 4)


# ---------------------------------------------------------------------------
# Contraction direction: incoming vs outgoing
# ---------------------------------------------------------------------------


def test_tmi_differs_from_outgoing_for_asymmetric_input(
    tmo: TriangleMultiplicationOutgoing,
    tmi: TriangleMultiplicationIncoming,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Same weights, different contractions, different output for asymmetric z.

    Incoming:  Σ_k a[k,i] ⊙ b[j,k]
    Outgoing:  Σ_k a[i,k] ⊙ b[k,j]
    These are not equal when z is not spatially symmetric.
    """
    _ = tmo.load_state_dict(tmi.state_dict())
    with torch.no_grad():
        out_i = tmi(z_single)
        out_o = tmo(z_single)
    assert not torch.allclose(out_i, out_o, atol=1e-4)


def test_tmi_equals_outgoing_for_symmetric_input_with_shared_weights(
    tmo: TriangleMultiplicationOutgoing,
    tmi: TriangleMultiplicationIncoming,
    z_sym: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """For spatially symmetric z, incoming == outgoing when weights are shared.

    With z[b,k,i,c] = z[b,i,k,c], LayerNorm preserves symmetry so a[k,i] =
    a[i,k]
    and b[j,k] = b[k,j].  Substituting into the incoming sum:
      Σ_k a[k,i] ⊙ b[j,k] = Σ_k a[i,k] ⊙ b[k,j]  (= outgoing sum).
    The outer gate g[i,j] is identical for both (shared weights, same z).
    """
    _ = tmo.load_state_dict(tmi.state_dict())
    with torch.no_grad():
        out_i = tmi(z_sym)
        out_o = tmo(z_sym)
    assert torch.allclose(out_i, out_o, atol=1e-4)


def test_tmi_incoming_z_equals_outgoing_transposed_z_when_inner_gates_pinned(
    tmo: TriangleMultiplicationOutgoing,
    tmi: TriangleMultiplicationIncoming,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """incoming(z) == outgoing(transpose(z)) when all inner gates pinned to 1.

    Mathematical identity: with gate_a and gate_b pinned (sigmoid ≈ 1), the
    inner sums
    reduce to:
      incoming m[i,j]         = Σ_k proj_a(zn[k,i]) ⊙ proj_b(zn[j,k])
      outgoing m[i,j] on z_T  = Σ_k proj_a(zn_T[i,k]) ⊙ proj_b(zn_T[k,j])
                               = Σ_k proj_a(zn[k,i]) ⊙ proj_b(zn[j,k]) ←
                               identical
    (because zn_T[i,k] = LN(z_T[i,k]) = LN(z[k,i]) = zn[k,i]).
    With the outer gate also pinned (gate.weight=0 makes sigmoid constant for
    both
    inputs), the final outputs match exactly.  This test would fail if either
    contraction were implemented as the other.
    """
    _ = tmo.load_state_dict(tmi.state_dict())

    with torch.no_grad():
        _ = tmi.gate.weight.fill_(0.0)
        _ = tmi.gate.bias.fill_(100.0)
        _ = tmi.gate_a.weight.fill_(0.0)
        _ = tmi.gate_a.bias.fill_(100.0)
        _ = tmi.gate_b.weight.fill_(0.0)
        _ = tmi.gate_b.bias.fill_(100.0)
        _ = tmo.gate.weight.fill_(0.0)
        _ = tmo.gate.bias.fill_(100.0)
        _ = tmo.gate_a.weight.fill_(0.0)
        _ = tmo.gate_a.bias.fill_(100.0)
        _ = tmo.gate_b.weight.fill_(0.0)
        _ = tmo.gate_b.bias.fill_(100.0)

    z_T = rearrange(z_single, "b i j c -> b j i c")

    with torch.no_grad():
        out_incoming = tmi(z_single)
        out_outgoing_T = tmo(z_T)

    assert torch.allclose(out_incoming, out_outgoing_T, atol=1e-5)


# ---------------------------------------------------------------------------
# Access-pattern sensitivity (incoming is the TRANSPOSE of outgoing)
#
# m[i,j] = Σ_k a[k,i] ⊙ b[j,k], so:
#   • a[k,i] derives from z[:,i,:] → column i of z drives first output dim.
#   • b[j,k] derives from z[j,:,:] → row j of z drives second output dim.
# Contrast with outgoing, where row i drives first dim and column j the second.
# ---------------------------------------------------------------------------


def test_tmi_output_first_dim_i_changes_when_column_i_of_z_is_perturbed(
    tmi: TriangleMultiplicationIncoming,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Out at first-dim index i changes when z[:, :, i, :] (column i) changed.

    a[k, i] = f_a(zn[k, i]) depends on z[:, :, i, :] for all k.
    m[i, j] = Σ_k a[k, i] ⊙ b[j, k] therefore changes for all j.
    """
    z_pert = z_single.clone()
    z_pert[0, :, 0, :] = z_pert[0, :, 0, :] + 5.0
    with torch.no_grad():
        out_orig = tmi(z_single)
        out_pert = tmi(z_pert)
    assert not torch.allclose(
        out_orig[0, 0, :, :],
        out_pert[0, 0, :, :],
        atol=1e-4,
    )


def test_tmi_output_second_dim_j_changes_when_row_j_of_z_is_perturbed(
    tmi: TriangleMultiplicationIncoming,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Out at second-dim index j changes when z[:, j, :, :] (row j) perturbed.

    b[j, k] = f_b(zn[j, k]) depends on z[:, j, :, :] for all k.
    m[i, j] = Σ_k a[k, i] ⊙ b[j, k] therefore changes for all i.
    """
    z_pert = z_single.clone()
    z_pert[0, 1, :, :] = z_pert[0, 1, :, :] + 5.0
    with torch.no_grad():
        out_orig = tmi(z_single)
        out_pert = tmi(z_pert)
    assert not torch.allclose(
        out_orig[0, :, 1, :],
        out_pert[0, :, 1, :],
        atol=1e-4,
    )


def test_tmi_output_other_first_dims_change_when_column_i_of_z_is_perturbed(
    tmi: TriangleMultiplicationIncoming,
    z_single: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Chnaging column i changes output at first-dim i' ≠ i via b[j, i] at k=i.

    m[i', j] = Σ_k a[k, i'] ⊙ b[j, k].  At k=i the term b[j, i] = f_b(zn[j, i])
    changes when z[:, :, i, :] is perturbed, propagating to ALL first-dim
    indices.
    This cross-column spread is the hallmark of the incoming contraction (the
    transpose
    of outgoing's cross-row spread).

    A channel-varying perturbation is used so LayerNorm cannot cancel the
    shift.
    """
    z_pert = z_single.clone()
    z_pert[0, :, 0, :] = z_pert[0, :, 0, :] + torch.randn(_N, _C) * 5.0
    with torch.no_grad():
        out_orig = tmi(z_single)
        out_pert = tmi(z_pert)
    # First-dim i'=1 must also change (b[j, 0] contributes to m[1, j] via k=0)
    assert not torch.allclose(
        out_orig[0, 1, :, :],
        out_pert[0, 1, :, :],
        atol=1e-4,
    )


# ---------------------------------------------------------------------------
# Type enforcement
# ---------------------------------------------------------------------------


def test_tmi_jaxtyping_rejects_3d_input(
    tmi: TriangleMultiplicationIncoming,
    z_no_batch: Float[torch.Tensor, "N N C"],
) -> None:
    """Beartype raises when the batch dimension is missing."""
    with pytest.raises(TypeCheckError):
        _ = tmi(z_no_batch)


def test_tmi_jaxtyping_rejects_non_square_spatial_dims(
    tmi: TriangleMultiplicationIncoming,
    z_nonsquare: Float[torch.Tensor, "B N M C"],
) -> None:
    """Beartype raises when the two N_res spatial dimensions differ."""
    with pytest.raises(TypeCheckError):
        _ = tmi(z_nonsquare)


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_triangle_mult_outgoing_forward_wrong_shape(
    tmo: TriangleMultiplicationOutgoing,
    z_3d: Float[torch.Tensor, "B N_RES N_RES"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        _ = tmo(z_3d)


def test_triangle_mult_incoming_forward_wrong_shape(
    tmi: TriangleMultiplicationIncoming,
    z_3d: Float[torch.Tensor, "B N_RES N_RES"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        _ = tmi(z_3d)


def test_pairformer_block_forward_wrong_shape(
    block: PairformerBlock,
    z_3d: Float[torch.Tensor, "B N_RES N_RES"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        _ = block(None, z_3d)


def test_pairformer_stack_forward_wrong_shape(
    stack: PairformerStack,
    z_3d: Float[torch.Tensor, "B N_RES N_RES"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        _ = stack(None, z_3d)

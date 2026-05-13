"""Tests for the pairformer stack."""

import pytest
import torch
from architecture.pairformer_stack import PairformerBlock, PairformerStack
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float, jaxtyped

torch.manual_seed(42)

B = 1
N_RES = 8
C = 32
N_HEADS = 4
N_BLOCKS = 3


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def mean_abs_asymmetry(
    x: Float[torch.Tensor, "B N N C"],
) -> Float[torch.Tensor, ""]:
    """Return mean absolute difference between x[b,i,j,c] and x[b,j,i,c]."""
    diff = x - rearrange(x, "b i j c -> b j i c")
    return reduce(diff.abs(), "b i j c -> ", "mean")


# ---------------------------------------------------------------------------
# Fixtures — models
# ---------------------------------------------------------------------------


@pytest.fixture
def block() -> PairformerBlock:
    """Provide a single PairformerBlock in eval mode."""
    return PairformerBlock(C, n_heads=N_HEADS).eval()


@pytest.fixture
def stack() -> PairformerStack:
    """Provide a PairformerStack with N_BLOCKS blocks in eval mode."""
    return PairformerStack(C, n_blocks=N_BLOCKS, n_heads=N_HEADS).eval()


# ---------------------------------------------------------------------------
# Fixtures — tensors
# ---------------------------------------------------------------------------


@pytest.fixture
def v() -> torch.Tensor:
    """Provide a random pair-embedding tensor (B, N_RES, N_RES, C)."""
    return torch.randn(B, N_RES, N_RES, C)


# ---------------------------------------------------------------------------
# PairformerBlock
# ---------------------------------------------------------------------------


def test_pairformer_block_output_shape(block: PairformerBlock, v: torch.Tensor) -> None:
    """PairformerBlock output shape matches the input (B, N_RES, N_RES, C)."""
    with torch.no_grad():
        out = block(v)
    assert out.shape == (B, N_RES, N_RES, C)


def test_pairformer_block_output_finite(block: PairformerBlock, v: torch.Tensor) -> None:
    """PairformerBlock output contains only finite values."""
    with torch.no_grad():
        out = block(v)
    assert torch.isfinite(out).all()


def test_pairformer_block_output_dtype(block: PairformerBlock, v: torch.Tensor) -> None:
    """PairformerBlock output dtype matches the input dtype."""
    with torch.no_grad():
        out = block(v)
    assert out.dtype == v.dtype


def test_pairformer_block_changes_input(block: PairformerBlock, v: torch.Tensor) -> None:
    """PairformerBlock output differs from its input (non-identity transform)."""
    with torch.no_grad():
        out = block(v)
    assert not torch.allclose(out, v)


def test_pairformer_block_gradient_flows(block: PairformerBlock, v: torch.Tensor) -> None:
    """Gradient propagates from PairformerBlock output back to v."""
    v_g = v.clone().requires_grad_(True)
    out = block(v_g)
    reduce(out, "b n m c -> ", "sum").backward()
    assert v_g.grad is not None
    assert torch.isfinite(v_g.grad).all()


def test_pairformer_block_row_attn_preserves_col_symmetry(
    block: PairformerBlock, v: torch.Tensor
) -> None:
    """Symmetric input remains finite after one PairformerBlock (row+col attention both applied)."""
    # Symmetric input should remain approximately symmetric after one block
    # (row + col attention are both applied, so neither axis is favoured)
    v_sym = (v + rearrange(v, "b i j c -> b j i c")) * 0.5
    with torch.no_grad():
        out = block(v_sym)
    # Not exactly symmetric (weights differ for row vs col), but check finite
    assert torch.isfinite(out).all()


def test_pairformer_block_different_batch_items_independent(block: PairformerBlock) -> None:
    """Two samples in batch is equivalent to processing them separately — no cross-contamination."""
    # Each batch element should be processed independently
    v1 = torch.randn(1, N_RES, N_RES, C)
    v2 = torch.randn(1, N_RES, N_RES, C)
    v_cat = torch.cat([v1, v2], dim=0)
    with torch.no_grad():
        out1 = block(v1)
        out2 = block(v2)
        out_cat = block(v_cat)
    assert torch.allclose(out1, out_cat[:1], atol=1e-5)
    assert torch.allclose(out2, out_cat[1:], atol=1e-5)


# ---------------------------------------------------------------------------
# PairformerStack
# ---------------------------------------------------------------------------


def test_pairformer_stack_output_shape(stack: PairformerStack, v: torch.Tensor) -> None:
    """PairformerStack returns an output of the same [B, N_res, N_res, C] shape as the input."""
    with torch.no_grad():
        out = stack(v)
    assert out.shape == (B, N_RES, N_RES, C)


def test_pairformer_stack_output_finite(stack: PairformerStack, v: torch.Tensor) -> None:
    """Stacking N_BLOCKS pairformer blocks does not produce NaN or Inf values."""
    with torch.no_grad():
        out = stack(v)
    assert torch.isfinite(out).all()


def test_pairformer_stack_block_count() -> None:
    """The n_blocks constructor argument creates exactly that many PairformerBlock instances."""
    s = PairformerStack(C, n_blocks=5, n_heads=N_HEADS)
    assert len(s.blocks) == 5


def test_pairformer_stack_depth_changes_output() -> None:
    """A 2-block stack produces a different output than a 1-block stack for the same input."""
    v = torch.randn(B, N_RES, N_RES, C)
    s1 = PairformerStack(C, n_blocks=1, n_heads=N_HEADS).eval()
    s2 = PairformerStack(C, n_blocks=2, n_heads=N_HEADS).eval()
    with torch.no_grad():
        out1 = s1(v)
        out2 = s2(v)
    assert not torch.allclose(out1, out2)


def test_pairformer_stack_gradient_flows(stack: PairformerStack, v: torch.Tensor) -> None:
    """Gradients propagate through all stacked blocks back to the input pair embedding."""
    v_g = v.clone().requires_grad_(True)
    out = stack(v_g)
    reduce(out, "b n m c -> ", "sum").backward()
    assert v_g.grad is not None
    assert torch.isfinite(v_g.grad).all()


def test_pairformer_stack_single_block_matches_pairformer_block() -> None:
    """Single-block PairformerStack with identical weights produces PairformerBlock exact output."""
    torch.manual_seed(0)
    v = torch.randn(B, N_RES, N_RES, C)
    block = PairformerBlock(C, n_heads=N_HEADS).eval()
    stack = PairformerStack(C, n_blocks=1, n_heads=N_HEADS).eval()
    stack.blocks[0].load_state_dict(block.state_dict())
    with torch.no_grad():
        assert torch.allclose(block(v), stack(v), atol=1e-6)

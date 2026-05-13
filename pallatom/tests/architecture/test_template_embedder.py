"""Tests for the template embedder."""

import pytest
import torch
import torch.nn.functional as F
from architecture.pairformer_stack import PairformerStack
from architecture.template_embedder import TemplateEmbedder
from beartype import beartype
from einops import einsum, rearrange, reduce
from jaxtyping import Float, jaxtyped

torch.manual_seed(42)

B = 2
N_RES = 6
N_BINS = 8
C_Z = 16  # input pair dim
C = 8  # internal pair dim  (must divide N_HEADS)
D = 16  # output dim
N_HEADS = 2
N_BLOCKS = 1


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def outer_product(
    v: Float[torch.Tensor, "B N"],
) -> Float[torch.Tensor, "B N N"]:
    """b_ij = v_i · v_j, the same outer product used for b_mask inside TemplateEmbedder."""
    return einsum(v, v, "b i, b j -> b i j")


@jaxtyped(typechecker=beartype)
def mean_sq_diff(
    a: Float[torch.Tensor, "B N N D"],
    b: Float[torch.Tensor, "B N N D"],
) -> Float[torch.Tensor, ""]:
    """Return mean squared element-wise difference between a and b."""
    diff = a - b
    return reduce(einsum(diff, diff, "b n m d, b n m d -> b n m"), "b n m -> ", "mean")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder() -> TemplateEmbedder:
    """Provide a small TemplateEmbedder in eval mode."""
    return TemplateEmbedder(
        n_bins=N_BINS,
        c_z=C_Z,
        c=C,
        d=D,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
    ).eval()


@pytest.fixture
def f_distogram() -> Float[torch.Tensor, "B N_res N_res N_bins"]:
    """Provide a random one-hot distogram tensor (B, N_RES, N_RES, N_BINS)."""
    return F.one_hot(torch.randint(0, N_BINS, (B, N_RES, N_RES)), N_BINS).float()


@pytest.fixture
def f_pseudo_beta_mask() -> Float[torch.Tensor, "B N_res"]:
    """Provide an all-ones pseudo-beta mask (B, N_RES)."""
    return torch.ones(B, N_RES)


@pytest.fixture
def zero_mask() -> Float[torch.Tensor, "B N_res"]:
    """Provide an all-zeros pseudo-beta mask (B, N_RES)."""
    return torch.zeros(B, N_RES)


@pytest.fixture
def z_ij() -> Float[torch.Tensor, "B N_res N_res C_z"]:
    """Provide a random pair-embedding tensor (B, N_RES, N_RES, C_Z)."""
    return torch.randn(B, N_RES, N_RES, C_Z)


# ---------------------------------------------------------------------------
# Typed helper self-tests
# ---------------------------------------------------------------------------


def test_outer_product_known_values():
    """outer_product produces the correct values for a known input."""
    v = torch.tensor([[1.0, 0.0, 1.0]])  # B=1, N=3
    expected = torch.tensor([[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]])
    assert torch.allclose(outer_product(v), expected)


def test_outer_product_diagonal_is_elementwise_square(
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
):
    """outer_product diagonal equals element-wise square of the input vector."""
    op = outer_product(f_pseudo_beta_mask)  # (B, N, N)
    diag = torch.diagonal(op, dim1=-2, dim2=-1)  # (B, N)
    assert torch.allclose(diag, f_pseudo_beta_mask**2)


# ---------------------------------------------------------------------------
# PairformerStack
# ---------------------------------------------------------------------------


def test_pairformer_stack_preserves_shape():
    """PairformerStack used inside template embedder preserves the [B, N_res, N_res, C] shape."""
    stack = PairformerStack(c=C, n_blocks=N_BLOCKS, n_heads=N_HEADS)
    v = torch.randn(B, N_RES, N_RES, C)
    out = stack(v)
    assert out.shape == (B, N_RES, N_RES, C)
    assert torch.isfinite(out).all()


def test_pairformer_stack_output_differs_from_input():
    """PairformerStack applies a non-trivial transform — output is distinct from the input."""
    stack = PairformerStack(c=C, n_blocks=N_BLOCKS, n_heads=N_HEADS)
    v = torch.randn(B, N_RES, N_RES, C)
    assert not torch.allclose(stack(v), v)


def test_pairformer_stack_gradient_flows():
    """Gradients flow through the PairformerStack back to its pair embedding input."""
    stack = PairformerStack(c=C, n_blocks=N_BLOCKS, n_heads=N_HEADS)
    v = torch.randn(B, N_RES, N_RES, C, requires_grad=True)
    reduce(stack(v), "b n m c -> ", "sum").backward()
    assert v.grad is not None
    assert torch.isfinite(v.grad).all()


# ---------------------------------------------------------------------------
# TemplateEmbedder — output shape and values
# ---------------------------------------------------------------------------


def test_time_modulates_output(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
):
    """Changing diffusion time t changes template embedding, confirms time conditioning is wired."""
    with torch.no_grad():
        out_t0 = embedder(f_distogram, f_pseudo_beta_mask, z_ij, t=0.0)
        out_t5 = embedder(f_distogram, f_pseudo_beta_mask, z_ij, t=0.5)
    assert mean_sq_diff(out_t0, out_t5).item() > 0


def test_mask_zeros_modulates_output(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    zero_mask: Float[torch.Tensor, "B N_res"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
):
    """All-zero pseudo-beta mask (no valid template) produces diff output than all-ones mask."""
    with torch.no_grad():
        out_ones = embedder(f_distogram, torch.ones(B, N_RES), z_ij, t=0.5)
        out_zeros = embedder(f_distogram, zero_mask, z_ij, t=0.5)
    assert mean_sq_diff(out_ones, out_zeros).item() > 0


def test_distogram_modulates_output(
    embedder: TemplateEmbedder,
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
):
    """Different distogram templates produce different embeddings."""
    f_dist_a = F.one_hot(torch.randint(0, N_BINS, (B, N_RES, N_RES)), N_BINS).float()
    f_dist_b = F.one_hot(torch.randint(0, N_BINS, (B, N_RES, N_RES)), N_BINS).float()
    with torch.no_grad():
        out_a = embedder(f_dist_a, f_pseudo_beta_mask, z_ij, t=0.5)
        out_b = embedder(f_dist_b, f_pseudo_beta_mask, z_ij, t=0.5)
    assert mean_sq_diff(out_a, out_b).item() > 0


def test_z_ij_modulates_output(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
):
    """Different trunk pair embeddings z_ij produce different template embedder outputs."""
    z_a = torch.randn(B, N_RES, N_RES, C_Z)
    z_b = torch.randn(B, N_RES, N_RES, C_Z)
    with torch.no_grad():
        out_a = embedder(f_distogram, f_pseudo_beta_mask, z_a, t=0.5)
        out_b = embedder(f_distogram, f_pseudo_beta_mask, z_b, t=0.5)
    assert mean_sq_diff(out_a, out_b).item() > 0


def test_batched_consistency(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
):
    """Processing single sample in batch produces same result as extracting from a larger batch."""
    with torch.no_grad():
        out_batch = embedder(f_distogram, f_pseudo_beta_mask, z_ij, t=0.5)
        out_single = embedder(f_distogram[:1], f_pseudo_beta_mask[:1], z_ij[:1], t=0.5)
    assert torch.allclose(
        rearrange(out_batch[0], "n m d -> 1 n m d"),
        out_single,
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# TemplateEmbedder — gradient flow
# ---------------------------------------------------------------------------


def test_gradient_flows_to_z_ij(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
):
    """Template embedder output is differentiable with respect to the input pair embedding z_ij."""
    z_g = torch.randn(B, N_RES, N_RES, C_Z, requires_grad=True)
    out = embedder(f_distogram, f_pseudo_beta_mask, z_g, t=0.3)
    reduce(out, "b n m d -> ", "sum").backward()
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_gradient_flows_to_f_distogram(
    embedder: TemplateEmbedder,
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
):
    """Template embedder output is differentiable with respect to distogram template features."""
    f_dist_g = torch.randn(B, N_RES, N_RES, N_BINS, requires_grad=True)
    out = embedder(f_dist_g, f_pseudo_beta_mask, z_ij, t=0.3)
    reduce(out, "b n m d -> ", "sum").backward()
    assert f_dist_g.grad is not None
    assert torch.isfinite(f_dist_g.grad).all()


def test_gradient_flows_to_f_pseudo_beta_mask(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
):
    """The template embedder output is differentiable with respect to the pseudo-beta mask."""
    mask_g = torch.ones(B, N_RES, requires_grad=True)
    out = embedder(f_distogram, mask_g, z_ij, t=0.3)
    reduce(out, "b n m d -> ", "sum").backward()
    assert mask_g.grad is not None
    assert torch.isfinite(mask_g.grad).all()

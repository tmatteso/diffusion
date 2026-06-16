"""Tests for the template embedder.

Covers TemplateEmbedder forward pass correctness, output shape, gradient flow,
time/mask/distogram modulation, batched consistency, and shape-contract
enforcement. Also tests the typed helper utilities (outer_product,
mean_sq_diff) and the underlying PairformerStack used inside TemplateEmbedder.
"""

import pytest
import torch
import torch.nn.functional as F
from architecture.pairformer_stack import PairformerStack
from architecture.template_embedder import TemplateEmbedder
from beartype import beartype
from einops import einsum, rearrange, reduce
from helpers.useful_objects import manual_seed
from jaxtyping import Float, TypeCheckError, jaxtyped

_ = manual_seed(42)

B = 2
N_RES = 6
N_BINS = 8
C_Z = 16  # input pair dim
C = 8  # internal pair dim  (must divide N_HEADS)
D = 16  # output dim
N_HEADS = 4
N_BLOCKS = 1


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def outer_product(
    v: Float[torch.Tensor, "B N"],
) -> Float[torch.Tensor, "B N N"]:
    """Compute the outer product b_ij = v_i · v_j.

    Mirrors the mask outer product used to build b_mask inside
    TemplateEmbedder.
    """
    return einsum(v, v, "b i, b j -> b i j")


@jaxtyped(typechecker=beartype)
def mean_sq_diff(
    a: Float[torch.Tensor, "B N N D"],
    b: Float[torch.Tensor, "B N N D"],
) -> Float[torch.Tensor, ""]:
    """Return mean squared element-wise difference between a and b.

    Reduces over all batch, spatial, and channel dimensions.
    """
    diff = a - b
    return reduce(
        einsum(diff, diff, "b n m d, b n m d -> b n m"),
        "b n m -> ",
        "mean",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder() -> TemplateEmbedder:
    """Provide a small TemplateEmbedder in eval mode.

    Constructed with test-sized hyperparameters (N_BINS, C_Z, C, D, N_BLOCKS,
    N_HEADS).
    """
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
    """Provide a random one-hot distogram tensor (B, N_RES, N_RES, N_BINS).

    Each pairwise position is assigned a random bin via one-hot encoding.
    """
    return F.one_hot(
        torch.randint(0, N_BINS, (B, N_RES, N_RES)),
        N_BINS,
    ).float()


@pytest.fixture
def f_pseudo_beta_mask() -> Float[torch.Tensor, "B N_res"]:
    """Provide an all-ones pseudo-beta mask (B, N_RES).

    Indicates every residue has a valid pseudo-beta carbon in the template.
    """
    return torch.ones(B, N_RES)


@pytest.fixture
def zero_mask() -> Float[torch.Tensor, "B N_res"]:
    """Provide an all-zeros pseudo-beta mask (B, N_RES).

    Simulates a template with no valid pseudo-beta carbons for any residue.
    """
    return torch.zeros(B, N_RES)


@pytest.fixture
def z_ij() -> Float[torch.Tensor, "B N_res N_res C_z"]:
    """Provide a random pair-embedding tensor (B, N_RES, N_RES, C_Z).

    Used as the trunk pair embedding input to TemplateEmbedder.
    """
    return torch.randn(B, N_RES, N_RES, C_Z)


# ---------------------------------------------------------------------------
# Typed helper self-tests
# ---------------------------------------------------------------------------


def test_outer_product_known_values() -> None:
    """Verify outer_product produces the correct values for a known input.

    Uses hand-crafted vector where the expected outer product matrix is exact.
    """
    v = torch.tensor([[1.0, 0.0, 1.0]])  # B=1, N=3
    expected = torch.tensor(
        [[[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]],
    )
    assert torch.allclose(outer_product(v), expected)


def test_outer_product_diagonal_is_elementwise_square(
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
) -> None:
    """Outer_product diagonal equals element-wise square of the input vector.

    Checks identity diag(v ⊗ v) = v² for the all-ones pseudo-beta mask fixture.
    """
    op = outer_product(f_pseudo_beta_mask)  # (B, N, N)
    diag = torch.diagonal(op, dim1=-2, dim2=-1)  # (B, N)
    assert torch.allclose(diag, f_pseudo_beta_mask**2)


# ---------------------------------------------------------------------------
# PairformerStack
# ---------------------------------------------------------------------------


def test_pairformer_stack_preserves_shape() -> None:
    """PairformerStack preserves the [B, N_res, N_res, C] pair embedding shape.

    Checks both shape equality and that all output values are finite.
    """
    stack = PairformerStack(c=C, n_blocks=N_BLOCKS, n_heads=N_HEADS)
    v = torch.randn(B, N_RES, N_RES, C)
    out = stack(s=None, z=v)
    assert out.shape == (B, N_RES, N_RES, C)
    assert torch.isfinite(out).all()


def test_pairformer_stack_gradient_flows() -> None:
    """Gradients flow through PairformerStack back to pair embedding input.

    Confirms that the backward pass populates v.grad with finite values.
    """
    stack = PairformerStack(c=C, n_blocks=N_BLOCKS, n_heads=N_HEADS)
    v = torch.randn(B, N_RES, N_RES, C, requires_grad=True)
    torch.autograd.backward([reduce(stack(s=None, z=v), "b n m c -> ", "sum")])
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
) -> None:
    """Verify changing diffusion time t changes the template embedding output.

    Confirms time conditioning is correctly wired by comparing outputs at t=0
    and t=0.5.
    """
    with torch.no_grad():
        out_t0 = embedder(
            f_distogram,
            f_pseudo_beta_mask,
            z_ij,
            t=torch.zeros(B, N_RES, N_RES),
        )
        out_t5 = embedder(
            f_distogram,
            f_pseudo_beta_mask,
            z_ij,
            t=torch.full((B, N_RES, N_RES), 0.5),
        )
    assert mean_sq_diff(out_t0, out_t5).item() > 0


def test_mask_zeros_modulates_output(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    zero_mask: Float[torch.Tensor, "B N_res"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
) -> None:
    """All-zero pseudo-beta mask produces different output than all-ones mask.

    Checks mask (indicating valid template residues) meaningfully influences
    the embedding, simulating the no-valid-template case.
    """
    t = torch.full((B, N_RES, N_RES), 0.5)
    with torch.no_grad():
        out_ones = embedder(f_distogram, torch.ones(B, N_RES), z_ij, t=t)
        out_zeros = embedder(f_distogram, zero_mask, z_ij, t=t)
    assert mean_sq_diff(out_ones, out_zeros).item() > 0


def test_distogram_modulates_output(
    embedder: TemplateEmbedder,
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
) -> None:
    """Verify that different distogram templates produce different embeddings.

    Generates two independent random one-hot distograms and checks their
    outputs differ.
    """
    f_dist_a = F.one_hot(
        torch.randint(0, N_BINS, (B, N_RES, N_RES)),
        N_BINS,
    ).float()
    f_dist_b = F.one_hot(
        torch.randint(0, N_BINS, (B, N_RES, N_RES)),
        N_BINS,
    ).float()
    t = torch.full((B, N_RES, N_RES), 0.5)
    with torch.no_grad():
        out_a = embedder(f_dist_a, f_pseudo_beta_mask, z_ij, t=t)
        out_b = embedder(f_dist_b, f_pseudo_beta_mask, z_ij, t=t)
    assert mean_sq_diff(out_a, out_b).item() > 0


def test_z_ij_modulates_output(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
) -> None:
    """Different pair embeddings z_ij produce different template embedder out.

    Checks trunk pair embedding is active conditioning signal in forward pass.
    """
    z_a = torch.randn(B, N_RES, N_RES, C_Z)
    z_b = torch.randn(B, N_RES, N_RES, C_Z)
    t = torch.full((B, N_RES, N_RES), 0.5)
    with torch.no_grad():
        out_a = embedder(f_distogram, f_pseudo_beta_mask, z_a, t=t)
        out_b = embedder(f_distogram, f_pseudo_beta_mask, z_b, t=t)
    assert mean_sq_diff(out_a, out_b).item() > 0


def test_batched_consistency(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
) -> None:
    """Processing one sample gives same result as slicing from larger batch.

    Confirms embedder has no cross-batch interactions or batch-norm artefacts.
    """
    with torch.no_grad():
        out_batch = embedder(
            f_distogram,
            f_pseudo_beta_mask,
            z_ij,
            t=torch.full((B, N_RES, N_RES), 0.5),
        )
        out_single = embedder(
            f_distogram[:1],
            f_pseudo_beta_mask[:1],
            z_ij[:1],
            t=torch.full((1, N_RES, N_RES), 0.5),
        )
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
) -> None:
    """Verify template embedder output is differentiable with respect to z_ij.

    Checks that the backward pass populates z_g.grad with finite values.
    """
    z_g = torch.randn(B, N_RES, N_RES, C_Z, requires_grad=True)
    t = torch.full((B, N_RES, N_RES), 0.3)
    out = embedder(f_distogram, f_pseudo_beta_mask, z_g, t=t)
    torch.autograd.backward([reduce(out, "b n m d -> ", "sum")])
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_gradient_flows_to_f_distogram(
    embedder: TemplateEmbedder,
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
) -> None:
    """Template embedder output is differentiable w.r.t distogram features.

    Checks that the backward pass populates f_dist_g.grad with finite values.
    """
    f_dist_g = torch.randn(B, N_RES, N_RES, N_BINS, requires_grad=True)
    t = torch.full((B, N_RES, N_RES), 0.3)
    out = embedder(f_dist_g, f_pseudo_beta_mask, z_ij, t=t)
    torch.autograd.backward([reduce(out, "b n m d -> ", "sum")])
    assert f_dist_g.grad is not None
    assert torch.isfinite(f_dist_g.grad).all()


def test_gradient_flows_to_f_pseudo_beta_mask(
    embedder: TemplateEmbedder,
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    z_ij: Float[torch.Tensor, "B N_res N_res C_z"],
) -> None:
    """Template embedder output is differentiable w.r.t. pseudo-beta mask.

    Checks that the backward pass populates mask_g.grad with finite values.
    """
    mask_g = torch.ones(B, N_RES, requires_grad=True)
    t = torch.full((B, N_RES, N_RES), 0.3)
    out = embedder(f_distogram, mask_g, z_ij, t=t)
    torch.autograd.backward([reduce(out, "b n m d -> ", "sum")])
    assert mask_g.grad is not None
    assert torch.isfinite(mask_g.grad).all()


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_template_embedder_forward_wrong_shape(
    embedder: TemplateEmbedder,
) -> None:
    """Verify wrong f_distogram ndim triggers TypeCheckError.

    Passes a 3-D tensor (missing the n_bins dimension) to confirm shape
    contracts fire.
    """
    f_distogram_bad = torch.zeros(B, N_RES, N_RES)  # missing n_bins dim
    bad_mask = torch.zeros(B, N_RES)
    bad_z_ij = torch.zeros(B, N_RES, N_RES, C_Z)
    t = torch.full((B, N_RES, N_RES), 0.5)
    with pytest.raises(TypeCheckError):
        _ = embedder(f_distogram_bad, bad_mask, bad_z_ij, t)

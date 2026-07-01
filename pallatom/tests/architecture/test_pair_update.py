"""Tests for the pair_update architecture module.

Covers TransformRBF, TriangleAttentionStartingNodeWithBias,
TriangleAttentionEndingNodeWithBias, Transition, DropoutRowwise,
DropoutColumnwise, and PairUpdate including output shapes, finite-value
checks, gradient flow, dropout semantics, and SE(3) geometric invariance.
"""

import pytest
import torch
from architecture.pair_update import (
    DropoutColumnwise,
    DropoutRowwise,
    PairUpdate,
    ParamsForRBF,
    TransformRBF,
    Transition,
    TriangleAttentionEndingNodeWithBias,
    TriangleAttentionStartingNodeWithBias,
)
from beartype import beartype
from einops import einsum, rearrange, reduce
from helpers.useful_objects import manual_seed
from jaxtyping import Float, TypeCheckError, jaxtyped

_ = manual_seed(42)

N_RES = 8
C = 32
N_HEADS = 4
B = 2

TOLERANCE = 1e-5
# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def mean_abs_asymmetry(
    x: Float[torch.Tensor, "B N N C"],
) -> Float[torch.Tensor, ""]:
    """Return mean absolute difference between x[b,i,j,c] and x[b,j,i,c].

    Used to quantify how far a pair tensor is from being symmetric in its
    spatial indices.
    """
    return reduce(
        (x - rearrange(x, "b i j c -> b j i c")).abs(),
        "b i j c -> ",
        "mean",
    )


@jaxtyped(typechecker=beartype)
def compute_dij(
    r: Float[torch.Tensor, "B N_res 3"],
) -> Float[torch.Tensor, "B N_res N_res"]:
    """Compute the (B, N_res, N_res) Euclidean pairwise distance matrix.

    For each batch item and each pair of residue positions (i, j), computes
    the L2 distance between the two 3-D coordinates.
    """
    diff = rearrange(r, "b n d -> b n 1 d") - rearrange(r, "b n d -> b 1 n d")
    return torch.sqrt(
        reduce(diff**2, "b n m d -> b n m", "sum").clamp(min=1e-8),
    )


@jaxtyped(typechecker=beartype)
def random_rotation() -> Float[torch.Tensor, "3 3"]:
    """Generate a uniformly random proper rotation matrix (det = +1).

    Uses QR decomposition of a Gaussian random matrix and corrects the sign
    so the determinant is exactly +1.
    """
    Q, _ = torch.qr(torch.randn(3, 3))
    if Q.det() < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@jaxtyped(typechecker=beartype)
def apply_rotation(
    r: Float[torch.Tensor, "B N_res 3"],
    R: Float[torch.Tensor, "3 3"],
) -> Float[torch.Tensor, "B N_res 3"]:
    """Apply rotation matrix R to every atom coordinate in r.

    Each coordinate vector in the (B, N_res) batch is rotated by the same
    global rotation R via a batched matrix-vector product.
    """
    return einsum(r, R, "b n d, d e -> b n e")


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rbf() -> TransformRBF:
    """Provide a TransformRBF module in eval mode.

    Returns a freshly constructed TransformRBF built from default ParamsForRBF,
    switched to evaluation mode so dropout and batch-norm are deterministic.
    """
    return TransformRBF(ParamsForRBF()).eval()


@pytest.fixture
def tri_start() -> TriangleAttentionStartingNodeWithBias:
    """Provide a TriangleAttentionStartingNodeWithBias module in eval mode.

    Returns the module constructed with channel width C and N_HEADS attention
    heads, switched to evaluation mode.
    """
    return TriangleAttentionStartingNodeWithBias(C, n_heads=N_HEADS).eval()


@pytest.fixture
def tri_end() -> TriangleAttentionEndingNodeWithBias:
    """Provide a TriangleAttentionEndingNodeWithBias module in eval mode.

    Returns the module constructed with channel width C and N_HEADS attention
    heads, switched to evaluation mode.
    """
    return TriangleAttentionEndingNodeWithBias(C, n_heads=N_HEADS).eval()


@pytest.fixture
def transition() -> Transition:
    """Provide a Transition module in eval mode.

    Returns a freshly constructed Transition MLP with channel width C,
    switched to evaluation mode.
    """
    return Transition(C).eval()


@pytest.fixture
def pair_update() -> PairUpdate:
    """Provide a PairUpdate module (no dropout) in eval mode.

    Returns a PairUpdate built with N_HEADS attention heads and default
    ParamsForRBF, with dropout disabled (p=0.0) and switched to evaluation
    mode.
    """
    return PairUpdate(C, n_heads=N_HEADS, dropout=0.0).eval()


# ---------------------------------------------------------------------------
# Input tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def z() -> Float[torch.Tensor, "B N_res N_res C"]:
    """Provide a random pair-embedding tensor (B, N_RES, N_RES, C).

    The tensor is sampled from a standard normal distribution and is used as
    the primary pair embedding input across multiple tests.
    """
    return torch.randn(B, N_RES, N_RES, C)


@pytest.fixture
def b() -> Float[torch.Tensor, "B N_res N_res C"]:
    """Provide a random pair bias tensor [B, N_RES, N_RES, C].

    The tensor is sampled from a standard normal distribution and serves as
    the attention bias input to triangle attention modules, which internally
    project it down to per-head width.
    """
    return torch.randn(B, N_RES, N_RES, C)


@pytest.fixture
def d() -> Float[torch.Tensor, "B N_res N_res"]:
    """Provide a Euclidean pairwise distance matrix [B, N_RES, N_RES].

    Distances are derived from random residue positions so the matrix is
    symmetric with a zero diagonal, matching the expected input contract of
    TransformRBF.
    """
    pos = torch.randn(B, N_RES, 3)
    diff = rearrange(pos, "b n c -> b n 1 c") - rearrange(
        pos,
        "b n c -> b 1 n c",
    )
    return torch.sqrt(
        reduce(diff**2, "b n m c -> b n m", "sum").clamp(min=1e-8),
    )


@pytest.fixture
def r_center() -> Float[torch.Tensor, "B N_res 3"]:
    """Provide random center-atom positions [B, N_RES, 3].

    Used as the geometric coordinate input to PairUpdate; sampled from a
    standard normal distribution.
    """
    return torch.randn(B, N_RES, 3)


# ---------------------------------------------------------------------------
# TransformRBF
# ---------------------------------------------------------------------------


def test_transform_rbf_output(
    rbf: TransformRBF,
    d: Float[torch.Tensor, "B N_res N_res"],
) -> None:
    """TransformRBF expands [B, N, N] dist matrix to [B, N, N, n_rbf] features.

    Verifies that the output shape is (B, N_RES, N_RES, n_rbf) and that all
    values are finite for a valid distance matrix.
    """
    with torch.no_grad():
        out = rbf(d)
    assert out.shape == (B, N_RES, N_RES, ParamsForRBF().n_rbf)
    assert torch.isfinite(out).all()


def test_transform_rbf_symmetric_distance_gives_symmetric_output(
    rbf: TransformRBF,
    d: Float[torch.Tensor, "B N_res N_res"],
) -> None:
    """A symmetric distance matrix produces a symmetric RBF output tensor.

    Because d[i,j] == d[j,i], the RBF output must also satisfy symmetry in
    the spatial dimensions; mean absolute asymmetry must be below TOLERANCE.
    """
    # d is a Euclidean dist matrix, so d[b,i,j] == d[b,j,i]; output must match
    with torch.no_grad():
        out = rbf(d)
    assert mean_abs_asymmetry(out).item() < TOLERANCE


# ---------------------------------------------------------------------------
# TriangleAttentionStartingNodeWithBias
# ---------------------------------------------------------------------------


def test_tri_start_output(
    tri_start: TriangleAttentionStartingNodeWithBias,
    z: Float[torch.Tensor, "B N_res N_res C"],
    b: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """TriangleAttentionStartingNode output shape and finite for random inputs.

    Verifies that the output shape is (B, N_res, N_res, C) and that no NaN or
    Inf values are produced for a standard-normal pair embedding and bias.
    """
    with torch.no_grad():
        out = tri_start(z, b)
    assert out.shape == (B, N_RES, N_RES, C)
    assert torch.isfinite(out).all()


def test_tri_start_gradient_flows(
    tri_start: TriangleAttentionStartingNodeWithBias,
    z: Float[torch.Tensor, "B N_res N_res C"],
    b: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Gradients flow through TriangleAttentionStartingNode to pair embedding.

    Verifies that z.grad is non-None and contains only finite values after a
    backward pass through the module.
    """
    z_g = z.clone().requires_grad_(True)  # noqa: FBT003
    out = tri_start(z_g, b)
    torch.autograd.backward([reduce(out, "b n m c -> ", "sum")])
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_tri_start_row_independence(
    tri_start: TriangleAttentionStartingNodeWithBias,
    z: Float[torch.Tensor, "B N_res N_res C"],
    b: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Modifying row 0 of z does not change output for rows 1..N_RES-1.

    Starting-node attention fixes row i and attends over columns j. Q, K, V, G
    for row i all come from z[:, i, :], so modifying row 0 must leave output
    rows 1..N_RES-1 completely unchanged.
    """
    # Starting-node attention fixes row i and attends over columns j.
    # Q, K, V, G for row i all come from z[:, i, :], so modifying row 0 must
    # leave output rows 1..N_RES-1 completely unchanged.
    z_mod = z.clone()
    z_mod[:, 0] = torch.randn_like(z_mod[:, 0])
    with torch.no_grad():
        out_orig = tri_start(z, b)
        out_mod = tri_start(z_mod, b)
    assert torch.allclose(out_orig[:, 1:], out_mod[:, 1:], atol=TOLERANCE)


# ---------------------------------------------------------------------------
# TriangleAttentionEndingNodeWithBias
# ---------------------------------------------------------------------------


def test_tri_end_output(
    tri_end: TriangleAttentionEndingNodeWithBias,
    z: Float[torch.Tensor, "B N_res N_res C"],
    b: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """TriangleAttentionEndingNode output shape and finite for random inputs.

    Verifies that the output shape is (B, N_res, N_res, C) and that no NaN or
    Inf values are produced for a standard-normal pair embedding and bias.
    """
    with torch.no_grad():
        out = tri_end(z, b)
    assert out.shape == (B, N_RES, N_RES, C)
    assert torch.isfinite(out).all()


def test_tri_end_gradient_flows(
    tri_end: TriangleAttentionEndingNodeWithBias,
    z: Float[torch.Tensor, "B N_res N_res C"],
    b: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Gradients flow through TriangleAttentionEndingNode to pair embedding z.

    Verifies that z.grad is non-None and contains only finite values after a
    backward pass through the module.
    """
    z_g = z.clone().requires_grad_(True)  # noqa: FBT003
    out = tri_end(z_g, b)
    torch.autograd.backward([reduce(out, "b n m c -> ", "sum")])
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_tri_end_col_independence(
    tri_end: TriangleAttentionEndingNodeWithBias,
    z: Float[torch.Tensor, "B N_res N_res C"],
    b: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Modifying column 0 of z does not change output for columns 1..N_RES-1.

    Ending-node attention fixes column j and attends over rows i. Q, K, V, G
    for column j all come from z[:, :, j], so modifying column 0 must leave
    output columns 1..N_RES-1 completely unchanged.
    """
    # Ending-node attention fixes column j and attends over rows i.
    # Q, K, V, G for column j all come from z[:, :, j], so modifying column 0
    # must leave output columns 1..N_RES-1 completely unchanged.
    z_mod = z.clone()
    z_mod[:, :, 0, :] = torch.randn_like(z_mod[:, :, 0, :])
    with torch.no_grad():
        out_orig = tri_end(z, b)
        out_mod = tri_end(z_mod, b)
    assert torch.allclose(
        out_orig[:, :, 1:, :],
        out_mod[:, :, 1:, :],
        atol=TOLERANCE,
    )


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


def test_transition_output_3d(
    transition: Transition,
    z: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Transition applied to pair tensor returns same shape and finite values.

    Verifies that the MLP preserves the leading batch and spatial dimensions
    while keeping channel width C, and produces no NaN or Inf values.
    """
    with torch.no_grad():
        out = transition(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_transition_output_shape_2d(transition: Transition) -> None:
    """Transition applied to a 2-D tensor [N, C] also returns the same shape.

    Verifies that the MLP is broadcast-safe and can handle inputs without a
    batch or pair dimension.
    """
    x = torch.randn(N_RES, C)
    with torch.no_grad():
        out = transition(x)
    assert out.shape == (N_RES, C)


def test_transition_gradient_flows(
    transition: Transition,
    z: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Gradients flow back through the Transition MLP to its input.

    Verifies that z.grad is non-None and contains only finite values after a
    backward pass through the module.
    """
    z_g = z.clone().requires_grad_(True)  # noqa: FBT003
    out = transition(z_g)
    torch.autograd.backward([reduce(out, "b n m c -> ", "sum")])
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


# ---------------------------------------------------------------------------
# DropoutRowwise
# ---------------------------------------------------------------------------


def test_dropout_rowwise_eval_is_identity(
    z: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """In eval mode DropoutRowwise passes its input through unchanged.

    Verifies that the module acts as an identity function during inference,
    producing output that is exactly equal to the input tensor.
    """
    drop = DropoutRowwise(p=0.5)
    _ = drop.eval()
    assert torch.equal(drop(z), z)


def test_dropout_rowwise_train_preserves_shape(
    z: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """DropoutRowwise in train mode returns a finite tensor of correct shape.

    Verifies that stochastic row dropping does not alter the output tensor
    shape or introduce non-finite values.
    """
    drop = DropoutRowwise(p=0.5)
    _ = drop.train()
    out = drop(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_dropout_rowwise_train_zeroes_entire_rows() -> None:
    """DropoutRowwise drops entire rows at once with inverted-dropout scaling.

    Uses a ones input so a dropped row becomes all-zero and a kept row becomes
    all 1/(1-p), verifying that the mask is applied uniformly across the entire
    row rather than element-wise.
    """
    _ = manual_seed(0)
    p = 0.5
    x = torch.ones(B, N_RES, N_RES, C)
    drop = DropoutRowwise(p=p)
    _ = drop.train()
    out = drop(x)
    scale = 1.0 / (1.0 - p)
    for i in range(N_RES):
        row = out[0, i]  # shape (N_RES, C) — check first batch item
        assert torch.allclose(row, torch.zeros_like(row)) or torch.allclose(
            row,
            torch.full_like(row, scale),
        )


# ---------------------------------------------------------------------------
# DropoutColumnwise
# ---------------------------------------------------------------------------


def test_dropout_columnwise_eval_is_identity(
    z: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """In eval mode DropoutColumnwise passes its input through unchanged.

    Verifies that the module acts as an identity function during inference,
    producing output that is exactly equal to the input tensor.
    """
    drop = DropoutColumnwise(p=0.5)
    _ = drop.eval()
    assert torch.equal(drop(z), z)


def test_dropout_columnwise_train_preserves_shape(
    z: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """DropoutColumnwise in train returns a finite tensor of correct shape.

    Verifies that stochastic column dropping does not alter the output tensor
    shape or introduce non-finite values.
    """
    drop = DropoutColumnwise(p=0.5)
    _ = drop.train()
    out = drop(z)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_dropout_columnwise_train_zeroes_entire_cols() -> None:
    """DropoutColumnwise drops entire columns with inverted-dropout scaling.

    Uses a ones input so a dropped column becomes all-zero and a kept column
    becomes all 1/(1-p), verifying that the mask is applied uniformly across
    the entire column rather than element-wise.
    """
    _ = manual_seed(0)
    p = 0.5
    x = torch.ones(B, N_RES, N_RES, C)
    drop = DropoutColumnwise(p=p)
    _ = drop.train()
    out = drop(x)
    scale = 1.0 / (1.0 - p)
    for j in range(N_RES):
        col = out[0, :, j, :]  # shape (N_RES, C) — check first batch item
        assert torch.allclose(col, torch.zeros_like(col)) or torch.allclose(
            col,
            torch.full_like(col, scale),
        )


# ---------------------------------------------------------------------------
# PairUpdate
# ---------------------------------------------------------------------------


def test_pair_update_changes_input(
    pair_update: PairUpdate,
    z: Float[torch.Tensor, "B N_res N_res C"],
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """PairUpdate is not an identity, output pair embedding differs from input.

    Verifies that the module actually transforms z rather than passing it
    through unchanged.
    """
    with torch.no_grad():
        out = pair_update(z, r_center)
    assert not torch.allclose(out, z)


def test_pair_update_gradient_flows_to_z(
    pair_update: PairUpdate,
    z: Float[torch.Tensor, "B N_res N_res C"],
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """Gradients flow back through PairUpdate to the input pair embedding z.

    Verifies that z.grad is non-None and contains only finite values after a
    backward pass through the full PairUpdate module.
    """
    z_g = z.clone().requires_grad_(True)  # noqa: FBT003
    out = pair_update(z_g, r_center)
    torch.autograd.backward([reduce(out, "b n m c -> ", "sum")])
    assert z_g.grad is not None
    assert torch.isfinite(z_g.grad).all()


def test_pair_update_gradient_flows_to_r_center(
    pair_update: PairUpdate,
    z: Float[torch.Tensor, "B N_res N_res C"],
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """Gradients flow back through PairUpdate to center-atom coords r_center.

    Verifies that r_center.grad is non-None and contains only finite values
    after a backward pass, confirming end-to-end differentiability through
    the distance computation.
    """
    r_g = r_center.clone().requires_grad_(True)  # noqa: FBT003
    out = pair_update(z, r_g)
    torch.autograd.backward([reduce(out, "b n m c -> ", "sum")])
    assert r_g.grad is not None
    assert torch.isfinite(r_g.grad).all()


def test_pair_update_no_nan_grad_from_zero_diagonal_distance(
    pair_update: PairUpdate,
) -> None:
    """Gradient w.r.t. r_center is finite at zero-distance diagonal (d[i,i]=0).

    d_ij[b, i, i] = 0 always; verifies that the backward pass clamps the
    denominator so the gradient is finite rather than NaN at the zero-distance
    diagonal.
    """
    # d_ij[b, i, i] = 0 always; torch.norm's backward clamps the denominator so
    # the gradient must be finite rather than nan at the zero-distance diagonal.
    r_g = torch.randn(B, N_RES, 3, requires_grad=True)
    out = pair_update(torch.randn(B, N_RES, N_RES, C), r_g)
    torch.autograd.backward([reduce(out, "b n m c -> ", "sum")])
    assert r_g.grad is not None
    assert torch.isfinite(r_g.grad).all()


def test_tri_start_changes_with_pair_bias(
    tri_start: TriangleAttentionStartingNodeWithBias,
    z: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Pair bias tensors produce triangle attention starting-node outputs.

    Verifies that the pair bias b is actually used in the attention computation
    and that varying it causes the output to change.
    """
    b1 = torch.randn(B, N_RES, N_RES, C)
    b2 = torch.randn(B, N_RES, N_RES, C)
    with torch.no_grad():
        out1 = tri_start(z, b1)
        out2 = tri_start(z, b2)
    assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Geometric Invariance
# Layer 1: distance computation invariant to translation and rotation.
# Layer 2: TransformRBF output invariant (invariant input -> invariant output).
# Layer 3: PairUpdate end-to-end output invariant.
# ---------------------------------------------------------------------------


def test_distance_translation_invariant(
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """Pairwise Euclidean distances unchanged by global translation of coords.

    Verifies that adding a constant offset to every coordinate in r_center
    leaves the distance matrix identical up to TOLERANCE.
    """
    t = torch.randn(1, 1, 3)
    with torch.no_grad():
        d_orig = compute_dij(r_center)
        d_shift = compute_dij(r_center + t)
    assert torch.allclose(d_orig, d_shift, atol=TOLERANCE)


def test_distance_rotation_invariant(
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """Pairwise Euclidean distances unchanged by global rotation of all coords.

    Verifies that applying a proper orthogonal rotation to every coordinate in
    r_center leaves the distance matrix identical up to TOLERANCE.
    """
    R = random_rotation()
    with torch.no_grad():
        d_orig = compute_dij(r_center)
        d_rot = compute_dij(apply_rotation(r_center, R))
    assert torch.allclose(d_orig, d_rot, atol=TOLERANCE)


def test_rbf_translation_invariant(
    rbf: TransformRBF,
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """TransformRBF output is unchanged by a global translation of coordinates.

    Verifies the composition: translation-invariant distances feed into a
    deterministic RBF projection, so final output is also
    translation-invariant.
    """
    t = torch.randn(1, 1, 3)
    with torch.no_grad():
        b_orig = rbf(compute_dij(r_center))
        b_shift = rbf(compute_dij(r_center + t))
    assert torch.allclose(b_orig, b_shift, atol=TOLERANCE)


def test_rbf_rotation_invariant(
    rbf: TransformRBF,
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """TransformRBF output is unchanged by a global rotation of coordinates.

    Verifies the composition: rotation-invariant distances feed into a
    deterministic RBF projection, so final output is also rotation-invariant.
    """
    R = random_rotation()
    with torch.no_grad():
        b_orig = rbf(compute_dij(r_center))
        b_rot = rbf(compute_dij(apply_rotation(r_center, R)))
    assert torch.allclose(b_orig, b_rot, atol=TOLERANCE)


def test_pair_update_translation_invariant(
    pair_update: PairUpdate,
    z: Float[torch.Tensor, "B N_res N_res C"],
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """PairUpdate output unchanged by global translation of center-atom coords.

    Verifies end-to-end translational invariance: coordinates enter PairUpdate
    only through the pairwise distance b_ij, so a global offset must not change
    the output.
    """
    t = torch.randn(1, 1, 3)
    with torch.no_grad():
        out_orig = pair_update(z, r_center)
        out_shift = pair_update(z, r_center + t)
    assert torch.allclose(out_orig, out_shift, atol=TOLERANCE)


def test_pair_update_rotation_invariant(
    pair_update: PairUpdate,
    z: Float[torch.Tensor, "B N_res N_res C"],
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """PairUpdate output unchanged by global rotation of center-atom coords.

    Verifies end-to-end rotational invariance: coordinates enter PairUpdate
    only through the pairwise distance b_ij, so a global rotation must not
    change the output.
    """
    R = random_rotation()
    with torch.no_grad():
        out_orig = pair_update(z, r_center)
        out_rot = pair_update(z, apply_rotation(r_center, R))
    assert torch.allclose(out_orig, out_rot, atol=TOLERANCE)


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_transform_rbf_forward_wrong_shape(rbf: TransformRBF) -> None:
    """Wrong d ndim (2-D instead of 3-D) triggers TypeCheckError.

    Verifies that the jaxtyping shape contract on TransformRBF.forward
    rejects a distance tensor that is missing its last N_res dimension.
    """
    d_bad = torch.zeros(B, N_RES)  # missing last N_res dim
    with pytest.raises(TypeCheckError):
        _ = rbf(d_bad)


def test_triangle_attn_starting_node_forward_wrong_shape(
    tri_start: TriangleAttentionStartingNodeWithBias,
    b: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers TypeCheckError.

    Verifies that the jaxtyping shape contract on
    TriangleAttentionStartingNodeWithBias.forward rejects a pair tensor that
    is missing its channel dimension.
    """
    z_bad = torch.zeros(B, N_RES, N_RES)  # missing c_pair dim
    with pytest.raises(TypeCheckError):
        _ = tri_start(z_bad, b)


def test_triangle_attn_ending_node_forward_wrong_shape(
    tri_end: TriangleAttentionEndingNodeWithBias,
    b: Float[torch.Tensor, "B N_res N_res C"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers TypeCheckError.

    Verifies that the jaxtyping shape contract on
    TriangleAttentionEndingNodeWithBias.forward rejects a pair tensor that
    is missing its channel dimension.
    """
    z_bad = torch.zeros(B, N_RES, N_RES)  # missing c_pair dim
    with pytest.raises(TypeCheckError):
        _ = tri_end(z_bad, b)


def test_pair_update_forward_wrong_shape(
    pair_update: PairUpdate,
    r_center: Float[torch.Tensor, "B N_res 3"],
) -> None:
    """Wrong z ndim (3-D instead of 4-D) triggers TypeCheckError.

    Verifies that the jaxtyping shape contract on PairUpdate.forward rejects
    a pair tensor that is missing its channel dimension.
    """
    z_bad = torch.zeros(B, N_RES, N_RES)  # missing c_pair dim
    with pytest.raises(TypeCheckError):
        _ = pair_update(z_bad, r_center)

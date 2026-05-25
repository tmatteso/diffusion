"""Tests for atom-level transformer layers."""

import pytest
import torch
import torch.nn.functional as F
from architecture.atom_transformers import (
    AtomAttentionDecoder,
    AtomFeatureEncoder,
    AtomTransformer,
    ConditionedTransitionBlock,
    DiffusionTransformer,
    build_sparse_pairs,
)
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Int, TypeCheckError, jaxtyped

manual_seed(42)

N_RES = 8
ATOMS_PER_RES = 3
N_ATOM = N_RES * ATOMS_PER_RES  # 24
E = 4  # element one-hot dim
C_ATOM = 32
C_ATOMPAIR = 16
C_TOKEN = 32
C_PAIR = 32
F_REF_DIM = ATOMS_PER_RES * (3 + E)
B = 1
N_HEADS = 4

# K is dynamic — pre-compute once from the canonical tok_idx
_base_tok = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
_base_nbrs, _ = build_sparse_pairs(_base_tok)
K = _base_nbrs.size(1)

# Sparse-mismatch regression constants.
# With a small window_size, K < N_ATOM, reproducing the production crash where
# attn (B, n_heads, N, N) and pair_bias (B, n_heads, N, K) had different last dims.
_SPARSE_WINDOW = 4  # small window so K << N_ATOM
_SPARSE_RES = 20  # number of residues — gives N_ATOM_SPARSE = 20 * 2 = 40
_ATOMS_PER_RES_SPARSE = 2
N_ATOM_SPARSE = _SPARSE_RES * _ATOMS_PER_RES_SPARSE
_sparse_tok = torch.repeat_interleave(torch.arange(_SPARSE_RES), _ATOMS_PER_RES_SPARSE)
_sparse_nbrs, _sparse_mask_raw = build_sparse_pairs(_sparse_tok, window_size=_SPARSE_WINDOW)
K_SPARSE = _sparse_nbrs.size(1)  # guaranteed < N_ATOM_SPARSE for _SPARSE_WINDOW small enough


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def pairwise_sq_dist(
    x: Float[torch.Tensor, "N D"],
) -> Float[torch.Tensor, "N N"]:
    """Compute pairwise squared Euclidean distances between N vectors."""
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    return einsum(diff, diff, "n m d, n m d -> n m")


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conditioned_transition_block() -> ConditionedTransitionBlock:
    """Provide a ConditionedTransitionBlock with c_a=c_s=C_ATOM in eval mode."""
    return ConditionedTransitionBlock(c_a=C_ATOM, c_s=C_ATOM, expansion=2).eval()


@pytest.fixture
def diffusion_transformer() -> DiffusionTransformer:
    """Provide a DiffusionTransformer with 2 blocks in eval mode."""
    return DiffusionTransformer(
        c_a=C_ATOM, c_s=C_ATOM, c_pair=C_PAIR, N_block=2, N_head=N_HEADS
    ).eval()


@pytest.fixture
def transformer() -> AtomTransformer:
    """Provide an AtomTransformer with 2 blocks in eval mode."""
    return AtomTransformer(C_ATOM, C_ATOMPAIR, n_blocks=2, n_heads=4).eval()


@pytest.fixture
def encoder() -> AtomFeatureEncoder:
    """Provide an AtomFeatureEncoder in eval mode."""
    return AtomFeatureEncoder(
        f_ref_dim=F_REF_DIM,
        c_token=C_TOKEN,
        c_pair=C_PAIR,
        c=C_TOKEN,
        d=C_ATOMPAIR,
        m=C_ATOM,
        n_blocks=1,
        n_heads=4,
    ).eval()


@pytest.fixture
def decoder() -> AtomAttentionDecoder:
    """Provide an AtomAttentionDecoder in eval mode."""
    return AtomAttentionDecoder(
        c_token=C_TOKEN,
        c_pair=C_PAIR,
        c_atom=C_ATOM,
        c_atompair=C_ATOMPAIR,
        n_blocks=1,
        n_heads=4,
    ).eval()


# ---------------------------------------------------------------------------
# Input tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tok_idx() -> Int[torch.Tensor, "N_atom"]:
    """Unbatched token index — used for build_sparse_pairs and deriving batched version."""
    return torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)


@pytest.fixture
def tok_idx_batched(tok_idx: Int[torch.Tensor, "N_atom"]) -> Int[torch.Tensor, "B N_atom"]:
    """Batched token index [B, N_atom] — passed to encoder/decoder."""
    return repeat(tok_idx, "n -> b n", b=B).contiguous()


@pytest.fixture
def neighbor_idx(tok_idx: Int[torch.Tensor, "N_atom"]) -> Int[torch.Tensor, "N_atom K"]:
    """Sparse neighbor index tensor [N_atom, K] computed from the canonical tok_idx."""
    nbrs, _ = build_sparse_pairs(tok_idx)
    return nbrs


@pytest.fixture
def valid_mask(tok_idx: Int[torch.Tensor, "N_atom"]) -> Bool[torch.Tensor, "B N_atom K"]:
    """Bool validity mask [B=1, N_atom, K] — True where neighbor slot is real atom, not padding."""
    _, mask = build_sparse_pairs(tok_idx)
    return rearrange(mask, "n k -> 1 n k")  # [B=1, N_atom, K]


@pytest.fixture
def ref_pos() -> Float[torch.Tensor, "B N_atom 3"]:
    """Random reference atom positions [B, N_atom, 3] build atom-pair features in encoder."""
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def ref_element() -> Float[torch.Tensor, "B N_atom E"]:
    """Random one-hot element features [B, N_atom, E] representing C/N/O/UNK per atom."""
    return F.one_hot(torch.randint(0, E, (B, N_ATOM)), num_classes=E).float()


@pytest.fixture
def ref_space_uid() -> Int[torch.Tensor, "B N_atom"]:
    """All-zero space UIDs [B, N_atom] placing every atom in the same covalent chain/space."""
    return torch.zeros(B, N_ATOM, dtype=torch.long)


@pytest.fixture
def s_input() -> Float[torch.Tensor, "B N_res C_token"]:
    """Random residue single embedding [B, N_res, C_token] serving as trunk context for encoder."""
    return torch.randn(B, N_RES, C_TOKEN)


@pytest.fixture
def z_input() -> Float[torch.Tensor, "B N_res N_res C_pair"]:
    """Random trunk pair embedding [B, N_res, N_res, C_pair] injected into encoder and decoder."""
    return torch.randn(B, N_RES, N_RES, C_PAIR)


@pytest.fixture
def a_res() -> Float[torch.Tensor, "B N_res C_atom"]:
    """Random residue-level atom embeddings [B, N_res, C_atom] for transition/transformer tests."""
    return torch.randn(B, N_RES, C_ATOM)


@pytest.fixture
def beta_dense() -> Float[torch.Tensor, "B N_res N_res"]:
    """Dense window-bias mask [B, N_res, N_res] with 0 in-window and -1e10 out-of-window."""
    return torch.zeros(B, N_RES, N_RES)


@pytest.fixture
def r_scaled() -> Float[torch.Tensor, "B N_atom 3"]:
    """Random atom positions [B, N_atom, 3] passed as conditioning signal to encoder."""
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def q_batched() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Random atom single (query) embedding [B, N_atom, C_atom] for transformer block tests."""
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def c_batched() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Random atom context embedding [B, N_atom, C_atom] used as key/value in transformer block."""
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def p_batched() -> Float[torch.Tensor, "B N_atom K C_atompair"]:
    """Random sparse atom-pair embedding [B, N_atom, K, C_atompair] for transformer block tests."""
    return torch.randn(B, N_ATOM, K, C_ATOMPAIR)


@pytest.fixture
def q_skip() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Random atom skip-connection query embed [B, N_atom, C_atom] passed to every decoder unit."""
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def c_skip() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Random atom skip-connection context embed[B, N_atom, C_atom] passed to every decoder unit."""
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def p_skip() -> Float[torch.Tensor, "B N_atom K C_atompair"]:
    """Random sparse pair skip embed [B, N_atom, K, C_atompair] passed to every decoder unit."""
    return torch.randn(B, N_ATOM, K, C_ATOMPAIR)


@pytest.fixture
def c_l() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Random atom-level latent embedding [B, N_atom, C_atom] that the decoder refines each unit."""
    return torch.randn(B, N_ATOM, C_ATOM)


# ---------------------------------------------------------------------------
# build_sparse_pairs
# ---------------------------------------------------------------------------


def test_build_sparse_pairs_output_shapes(tok_idx: Int[torch.Tensor, "N_atom"]):
    """Nbrs has shape [N_atom, K] and mask has matching shape with bool dtype."""
    nbrs, mask = build_sparse_pairs(tok_idx)
    assert nbrs.shape == (N_ATOM, K)
    assert mask.shape == (N_ATOM, K)
    assert mask.dtype == torch.bool


def test_build_sparse_pairs_all_valid_within_window(tok_idx: Int[torch.Tensor, "N_atom"]):
    """When N_res is smaller than half-window, every atom neighbours every other atom, no pad."""
    # N_RES=8 < WINDOW_SIZE//2=16, so every atom neighbours every other atom
    _, mask = build_sparse_pairs(tok_idx)
    assert mask.all()


def test_build_sparse_pairs_self_is_neighbour(tok_idx: Int[torch.Tensor, "N_atom"]):
    """Every atom index appears in own neighbour row, ensuring self-attention always possible."""
    nbrs, _ = build_sparse_pairs(tok_idx)
    for i in range(N_ATOM):
        assert (nbrs[i] == i).any()


def test_build_sparse_pairs_padding_indices_in_range(tok_idx: Int[torch.Tensor, "N_atom"]):
    """Padding slots re-use valid atom indices rather than out-of-bounds sentinels."""
    nbrs, _ = build_sparse_pairs(tok_idx)
    assert (nbrs >= 0).all()
    assert (nbrs < N_ATOM).all()


def test_build_sparse_pairs_window_excludes_far_atoms():
    """Atoms beyond half-window radius are excluded: boundary atoms only see local neighbourhood."""
    # 64 residues, 1 atom each, window_size=8 → half=4
    n_res = 64
    tok = torch.arange(n_res, dtype=torch.long)
    nbrs, mask = build_sparse_pairs(tok, window_size=8)
    half = 4
    first_valid = nbrs[0][mask[0]]
    last_valid = nbrs[-1][mask[-1]]
    assert (first_valid < half).all()
    assert (last_valid >= n_res - half).all()


# ---------------------------------------------------------------------------
# ConditionedTransitionBlock
# ---------------------------------------------------------------------------


def test_conditioned_transition_block_output_shape(
    conditioned_transition_block: ConditionedTransitionBlock,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
):
    """The block returns an updated embedding of the same [B, N_res, C_atom] shape as input."""
    with torch.no_grad():
        out = conditioned_transition_block(a_res, s_input)
    assert out.shape == (B, N_RES, C_ATOM)


def test_conditioned_transition_block_output_finite(
    conditioned_transition_block: ConditionedTransitionBlock,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
):
    """The SwiGLU gated output contains no NaN or Inf for random valid inputs."""
    with torch.no_grad():
        out = conditioned_transition_block(a_res, s_input)
    assert torch.isfinite(out).all()


def test_conditioned_transition_block_gradient_flows(
    conditioned_transition_block: ConditionedTransitionBlock,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
):
    """Gradients flow back through the AdaLN and SwiGLU gate to the atom embedding input."""
    a_g = a_res.clone().requires_grad_(True)
    out = conditioned_transition_block(a_g, s_input)
    reduce(out, "b n c -> ", "sum").backward()
    assert a_g.grad is not None
    assert torch.isfinite(a_g.grad).all()


# ---------------------------------------------------------------------------
# DiffusionTransformer
# ---------------------------------------------------------------------------


def test_diffusion_transformer_output_shape(
    diffusion_transformer: DiffusionTransformer,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    beta_dense: Float[torch.Tensor, "B N_res N_res"],
):
    """Running N_block rounds leaves the atom embedding shape [B, N_res, C_atom] unchanged."""
    with torch.no_grad():
        out = diffusion_transformer(a_res, s_input, z_input, beta_dense)
    assert out.shape == (B, N_RES, C_ATOM)


def test_diffusion_transformer_output_finite(
    diffusion_transformer: DiffusionTransformer,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    beta_dense: Float[torch.Tensor, "B N_res N_res"],
):
    """Stacked attention-plus-transition rounds produce no NaN or Inf for random valid inputs."""
    with torch.no_grad():
        out = diffusion_transformer(a_res, s_input, z_input, beta_dense)
    assert torch.isfinite(out).all()


def test_diffusion_transformer_n_block_stored():
    """The N_block constructor argument is stored and determines iteration depth."""
    dt = DiffusionTransformer(c_a=C_ATOM, c_s=C_ATOM, c_pair=C_PAIR, N_block=3, N_head=N_HEADS)
    assert dt.N_block == 3


def test_diffusion_transformer_gradient_flows(
    diffusion_transformer: DiffusionTransformer,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    beta_dense: Float[torch.Tensor, "B N_res N_res"],
):
    """Gradients propagate through all blocks back to the atom embedding input."""
    a_g = a_res.clone().requires_grad_(True)
    out = diffusion_transformer(a_g, s_input, z_input, beta_dense)
    reduce(out, "b n c -> ", "sum").backward()
    assert a_g.grad is not None
    assert torch.isfinite(a_g.grad).all()


# ---------------------------------------------------------------------------
# AtomTransformer
# ---------------------------------------------------------------------------


def test_atom_transformer_output_shape(
    transformer: AtomTransformer,
    q_batched: Float[torch.Tensor, "B N_atom C_atom"],
    c_batched: Float[torch.Tensor, "B N_atom C_atom"],
    p_batched: Float[torch.Tensor, "B N_atom K C_atompair"],
    neighbor_idx: Int[torch.Tensor, "N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
):
    """Stacking multiple blocks leaves the output shape [B, N_atom, C_atom] unchanged."""
    with torch.no_grad():
        out = transformer(q_batched, c_batched, p_batched, neighbor_idx, valid_mask)
    assert out.shape == (B, N_ATOM, C_ATOM)


def test_atom_transformer_output_finite(
    transformer: AtomTransformer,
    q_batched: Float[torch.Tensor, "B N_atom C_atom"],
    c_batched: Float[torch.Tensor, "B N_atom C_atom"],
    p_batched: Float[torch.Tensor, "B N_atom K C_atompair"],
    neighbor_idx: Int[torch.Tensor, "N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
):
    """Running two stacked blocks in sequence does not accumulate NaN or Inf values."""
    with torch.no_grad():
        out = transformer(q_batched, c_batched, p_batched, neighbor_idx, valid_mask)
    assert torch.isfinite(out).all()


def test_atom_transformer_block_count():
    """The n_blocks constructor argument is forwarded to DiffusionTransformer.N_block."""
    t = AtomTransformer(C_ATOM, C_ATOMPAIR, n_blocks=3, n_heads=4)
    assert t.blocks.N_block == 3


def test_atom_transformer_gradient_flows(
    transformer: AtomTransformer,
    q_batched: Float[torch.Tensor, "B N_atom C_atom"],
    c_batched: Float[torch.Tensor, "B N_atom C_atom"],
    p_batched: Float[torch.Tensor, "B N_atom K C_atompair"],
    neighbor_idx: Int[torch.Tensor, "N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
):
    """Gradients propagate through all stacked blocks back to the query input without vanishing."""
    q_g = q_batched.clone().requires_grad_(True)
    out = transformer(q_g, c_batched, p_batched, neighbor_idx, valid_mask)
    reduce(out, "b n c -> ", "sum").backward()
    assert q_g.grad is not None
    assert torch.isfinite(q_g.grad).all()


# ---------------------------------------------------------------------------
# AtomFeatureEncoder
# ---------------------------------------------------------------------------


def test_atom_feature_encoder_output_shapes(
    encoder: AtomFeatureEncoder,
    ref_pos: Float[torch.Tensor, "B N_atom 3"],
    ref_element: Float[torch.Tensor, "B N_atom E"],
    ref_space_uid: Int[torch.Tensor, "B N_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    r_scaled: Float[torch.Tensor, "B N_atom 3"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """The encoder returns five tensors with correct residue, atom, and sparse-pair shapes."""
    with torch.no_grad():
        s_i, q_skip, c_skip, p_skip, c_l = encoder(
            ref_pos,
            ref_element,
            ref_space_uid,
            s_input,
            z_input,
            r_scaled,
            tok_idx_batched,
        )
    assert s_i.shape == (B, N_RES, C_TOKEN)
    assert q_skip.shape == (B, N_ATOM, C_ATOM)
    assert c_skip.shape == (B, N_ATOM, C_ATOM)
    assert p_skip.shape == (B, N_ATOM, K, C_ATOMPAIR)
    assert c_l.shape == (B, N_ATOM, C_ATOM)


def test_atom_feature_encoder_outputs_finite(
    encoder: AtomFeatureEncoder,
    ref_pos: Float[torch.Tensor, "B N_atom 3"],
    ref_element: Float[torch.Tensor, "B N_atom E"],
    ref_space_uid: Int[torch.Tensor, "B N_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    r_scaled: Float[torch.Tensor, "B N_atom 3"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """None of the five encoder outputs contain NaN or Inf for random valid inputs."""
    with torch.no_grad():
        outputs = encoder(
            ref_pos,
            ref_element,
            ref_space_uid,
            s_input,
            z_input,
            r_scaled,
            tok_idx_batched,
        )
    for t in outputs:
        assert torch.isfinite(t).all()


def test_atom_feature_encoder_s_i_varies_across_residues(
    encoder: AtomFeatureEncoder,
    ref_pos: Float[torch.Tensor, "B N_atom 3"],
    ref_element: Float[torch.Tensor, "B N_atom E"],
    ref_space_uid: Int[torch.Tensor, "B N_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    r_scaled: Float[torch.Tensor, "B N_atom 3"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """Updated residue embeddings distinct — encoder does not collapse residues to same vector."""
    with torch.no_grad():
        s_i, *_ = encoder(
            ref_pos,
            ref_element,
            ref_space_uid,
            s_input,
            z_input,
            r_scaled,
            tok_idx_batched,
        )
    off_diag = pairwise_sq_dist(s_i[0]) + torch.eye(N_RES) * 1e10
    assert off_diag.min().item() > 0


def test_atom_feature_encoder_p_skip_k_matches_sparse_index(
    encoder: AtomFeatureEncoder,
    ref_pos: Float[torch.Tensor, "B N_atom 3"],
    ref_element: Float[torch.Tensor, "B N_atom E"],
    ref_space_uid: Int[torch.Tensor, "B N_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    r_scaled: Float[torch.Tensor, "B N_atom 3"],
    tok_idx: Int[torch.Tensor, "N_atom"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """K dimension of p_skip equals number of sparse neighbours produced by build_sparse_pairs."""
    with torch.no_grad():
        _, _, _, p_skip, _ = encoder(
            ref_pos,
            ref_element,
            ref_space_uid,
            s_input,
            z_input,
            r_scaled,
            tok_idx_batched,
        )
    nbrs, _ = build_sparse_pairs(tok_idx)
    assert p_skip.shape[2] == nbrs.size(1)  # K dim is index 2 in (B, N_atom, K, c_atompair)


def test_atom_feature_encoder_gradient_flows_to_r_scaled(
    encoder: AtomFeatureEncoder,
    ref_pos: Float[torch.Tensor, "B N_atom 3"],
    ref_element: Float[torch.Tensor, "B N_atom E"],
    ref_space_uid: Int[torch.Tensor, "B N_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """The encoder is differentiable with respect to the noisy input positions r_scaled."""
    r_scaled_g = torch.randn(B, N_ATOM, 3, requires_grad=True)
    _, q_skip, *_ = encoder(
        ref_pos,
        ref_element,
        ref_space_uid,
        s_input,
        z_input,
        r_scaled_g,
        tok_idx_batched,
    )
    reduce(q_skip, "b n c -> ", "sum").backward()
    assert r_scaled_g.grad is not None
    assert torch.isfinite(r_scaled_g.grad).all()


# ---------------------------------------------------------------------------
# AtomAttentionDecoder
# ---------------------------------------------------------------------------


def test_atom_attention_decoder_r_update_shape(
    decoder: AtomAttentionDecoder,
    q_skip: Float[torch.Tensor, "B N_atom C_atom"],
    p_skip: Float[torch.Tensor, "B N_atom K C_atompair"],
    c_skip: Float[torch.Tensor, "B N_atom C_atom"],
    c_l: Float[torch.Tensor, "B N_atom C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """The decoder produces a 3D position update tensor of shape [B, N_atom, 3] per decoder unit."""
    with torch.no_grad():
        _, _, r_update, _ = decoder(q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    assert r_update.shape == (B, N_ATOM, 3)


def test_atom_attention_decoder_r_update_finite(
    decoder: AtomAttentionDecoder,
    q_skip: Float[torch.Tensor, "B N_atom C_atom"],
    p_skip: Float[torch.Tensor, "B N_atom K C_atompair"],
    c_skip: Float[torch.Tensor, "B N_atom C_atom"],
    c_l: Float[torch.Tensor, "B N_atom C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """The position update output of the decoder contains no NaN or Inf for random valid inputs."""
    with torch.no_grad():
        _, _, r_update, _ = decoder(q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    assert torch.isfinite(r_update).all()


def test_atom_attention_decoder_c_out_shape(
    decoder: AtomAttentionDecoder,
    q_skip: Float[torch.Tensor, "B N_atom C_atom"],
    p_skip: Float[torch.Tensor, "B N_atom K C_atompair"],
    c_skip: Float[torch.Tensor, "B N_atom C_atom"],
    c_l: Float[torch.Tensor, "B N_atom C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """The updated atom embedding output of the decoder has shape [B, N_atom, C_atom]."""
    with torch.no_grad():
        _, _, _, c_out = decoder(q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    assert c_out.shape == (B, N_ATOM, C_ATOM)


def test_atom_attention_decoder_c_out_finite(
    decoder: AtomAttentionDecoder,
    q_skip: Float[torch.Tensor, "B N_atom C_atom"],
    p_skip: Float[torch.Tensor, "B N_atom K C_atompair"],
    c_skip: Float[torch.Tensor, "B N_atom C_atom"],
    c_l: Float[torch.Tensor, "B N_atom C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """The updated atom embedding output of the decoder contains no NaN or Inf values."""
    with torch.no_grad():
        _, _, _, c_out = decoder(q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    assert torch.isfinite(c_out).all()


def test_atom_attention_decoder_gradient_flows_to_q_skip(
    decoder: AtomAttentionDecoder,
    p_skip: Float[torch.Tensor, "B N_atom K C_atompair"],
    c_skip: Float[torch.Tensor, "B N_atom C_atom"],
    c_l: Float[torch.Tensor, "B N_atom C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    tok_idx_batched: Int[torch.Tensor, "B N_atom"],
):
    """The position update is differentiable with respect to the skip-connection query embedding."""
    q_skip_g = torch.randn(B, N_ATOM, C_ATOM, requires_grad=True)
    _, _, r_update, _ = decoder(q_skip_g, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    reduce(r_update, "b n d -> ", "sum").backward()
    assert q_skip_g.grad is not None
    assert torch.isfinite(q_skip_g.grad).all()


# ---------------------------------------------------------------------------
# Sparse-mismatch regression: K_SPARSE < N_ATOM_SPARSE
#
# Before the fix, DiffusionTransformer (and thus AtomTransformer) computed dense
# attention logits of shape (B, n_heads, N, N) then tried to add a sparse pair
# bias of shape (B, n_heads, N, K) with K != N, producing:
#   RuntimeError: The size of tensor a (N) must match the size of tensor b (K)
# These tests exercise the exact shape regime that triggered that crash.
# ---------------------------------------------------------------------------


@pytest.fixture
def sparse_neighbor_idx() -> Int[torch.Tensor, "N_atom_sparse K_sparse"]:
    """Sparse neighbour index [N_ATOM_SPARSE, K_SPARSE] with K_SPARSE < N_ATOM_SPARSE."""
    return _sparse_nbrs


@pytest.fixture
def sparse_valid_mask() -> Bool[torch.Tensor, "B N_atom_sparse K_sparse"]:
    """Validity mask [B, N_ATOM_SPARSE, K_SPARSE] for the sparse regression scenario."""
    return repeat(_sparse_mask_raw, "n k -> b n k", b=B)


@pytest.fixture
def q_sparse() -> Float[torch.Tensor, "B N_atom_sparse C_atom"]:
    """Query atom embeddings [B, N_ATOM_SPARSE, C_ATOM] for sparse regression."""
    return torch.randn(B, N_ATOM_SPARSE, C_ATOM)


@pytest.fixture
def p_sparse() -> Float[torch.Tensor, "B N_atom_sparse K_sparse C_atompair"]:
    """Sparse pair embeddings [B, N_ATOM_SPARSE, K_SPARSE, C_ATOMPAIR] with K < N."""
    return torch.randn(B, N_ATOM_SPARSE, K_SPARSE, C_ATOMPAIR)


@pytest.fixture
def z_sparse_dt() -> Float[torch.Tensor, "B N_atom_sparse K_sparse C_atom"]:
    """Sparse pair embeddings for DiffusionTransformer [B, N, K, C_ATOM] with K < N."""
    return torch.randn(B, N_ATOM_SPARSE, K_SPARSE, C_ATOM)


@pytest.fixture
def beta_sparse_dt() -> Float[torch.Tensor, "B N_atom_sparse K_sparse"]:
    """Sparse attention bias [B, N, K] aligned with z_sparse_dt."""
    return torch.zeros(B, N_ATOM_SPARSE, K_SPARSE)


def test_diffusion_transformer_sparse_no_shape_mismatch(
    z_sparse_dt: Float[torch.Tensor, "B N_atom_sparse K_sparse C_atom"],
    beta_sparse_dt: Float[torch.Tensor, "B N_atom_sparse K_sparse"],
    sparse_neighbor_idx: Int[torch.Tensor, "N_atom_sparse K_sparse"],
) -> None:
    """DiffusionTransformer with sparse z (K<N) and neighbor_idx raises no RuntimeError."""
    model = DiffusionTransformer(
        c_a=C_ATOM, c_s=C_ATOM, c_pair=C_ATOM, N_block=1, N_head=N_HEADS
    ).eval()
    a = torch.randn(B, N_ATOM_SPARSE, C_ATOM)
    s = torch.randn(B, N_ATOM_SPARSE, C_ATOM)
    with torch.no_grad():
        out = model(a, s, z_sparse_dt, beta_sparse_dt, neighbor_idx=sparse_neighbor_idx)
    assert out.shape == (B, N_ATOM_SPARSE, C_ATOM)
    assert torch.isfinite(out).all()


def test_atom_transformer_sparse_k_lt_n_no_shape_mismatch(
    q_sparse: Float[torch.Tensor, "B N_atom_sparse C_atom"],
    p_sparse: Float[torch.Tensor, "B N_atom_sparse K_sparse C_atompair"],
    sparse_neighbor_idx: Int[torch.Tensor, "N_atom_sparse K_sparse"],
    sparse_valid_mask: Bool[torch.Tensor, "B N_atom_sparse K_sparse"],
) -> None:
    """AtomTransformer with K_SPARSE < N_ATOM_SPARSE raises no RuntimeError.

    This is the exact shape regime that caused the production crash:
        attn (B, n_heads, N, N) + pair_bias (B, n_heads, N, K) with N != K.
    """
    model = AtomTransformer(C_ATOM, C_ATOMPAIR, n_blocks=1, n_heads=N_HEADS).eval()
    c = torch.randn(B, N_ATOM_SPARSE, C_ATOM)
    with torch.no_grad():
        out = model(q_sparse, c, p_sparse, sparse_neighbor_idx, sparse_valid_mask)
    assert out.shape == (B, N_ATOM_SPARSE, C_ATOM)
    assert torch.isfinite(out).all()


def test_atom_transformer_sparse_gradient_flows(
    p_sparse: Float[torch.Tensor, "B N_atom_sparse K_sparse C_atompair"],
    sparse_neighbor_idx: Int[torch.Tensor, "N_atom_sparse K_sparse"],
    sparse_valid_mask: Bool[torch.Tensor, "B N_atom_sparse K_sparse"],
) -> None:
    """Gradients flow through the sparse attention path when K < N."""
    model = AtomTransformer(C_ATOM, C_ATOMPAIR, n_blocks=1, n_heads=N_HEADS).eval()
    q_g = torch.randn(B, N_ATOM_SPARSE, C_ATOM, requires_grad=True)
    c = torch.randn(B, N_ATOM_SPARSE, C_ATOM)
    out = model(q_g, c, p_sparse, sparse_neighbor_idx, sparse_valid_mask)
    reduce(out, "b n c -> ", "sum").backward()
    assert q_g.grad is not None
    assert torch.isfinite(q_g.grad).all()


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_conditioned_transition_block_forward_wrong_shape(
    conditioned_transition_block: ConditionedTransitionBlock,
) -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers TypeCheckError."""
    a_bad = torch.zeros(B, N_RES)  # missing c_a dim
    s = torch.zeros(B, N_RES, C_ATOM)
    with pytest.raises(TypeCheckError):
        conditioned_transition_block(a_bad, s)


def test_diffusion_transformer_forward_wrong_shape(
    diffusion_transformer: DiffusionTransformer,
) -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers TypeCheckError."""
    a_bad = torch.zeros(B, N_RES)  # missing c_a dim
    s = torch.zeros(B, N_RES, C_ATOM)
    z = torch.zeros(B, N_RES, N_RES, C_PAIR)
    with pytest.raises(TypeCheckError):
        diffusion_transformer(a_bad, s, z)


def test_atom_transformer_forward_wrong_shape(transformer: AtomTransformer) -> None:
    """Wrong q ndim (2-D instead of 3-D) triggers TypeCheckError."""
    tok_idx = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
    nbrs, mask = build_sparse_pairs(tok_idx)
    q_bad = torch.zeros(B, N_ATOM)  # missing c_atom dim
    c = torch.zeros(B, N_ATOM, C_ATOM)
    p = torch.zeros(B, N_ATOM, K, C_ATOMPAIR)
    with pytest.raises(TypeCheckError):
        transformer(q_bad, c, p, nbrs, repeat(mask, "n k -> b n k", b=B))


def test_atom_feature_encoder_forward_wrong_shape(encoder: AtomFeatureEncoder) -> None:
    """Wrong ref_pos ndim (2-D instead of 3-D) triggers TypeCheckError."""
    tok_idx = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
    ref_pos_bad = torch.zeros(B, N_ATOM)  # missing coordinate dim
    ref_element = torch.zeros(B, N_ATOM, E)
    ref_space_uid = torch.zeros(B, N_ATOM, dtype=torch.long)
    s_input = torch.zeros(B, N_RES, C_TOKEN)
    z_input = torch.zeros(B, N_RES, N_RES, C_PAIR)
    r_scaled = torch.zeros(B, N_ATOM, 3)
    tok = repeat(tok_idx, "n -> b n", b=B)
    with pytest.raises(TypeCheckError):
        encoder(ref_pos_bad, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok)


def test_atom_attention_decoder_forward_wrong_shape(decoder: AtomAttentionDecoder) -> None:
    """Wrong q_skip ndim (2-D instead of 3-D) triggers TypeCheckError."""
    tok_idx = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
    tok = repeat(tok_idx, "n -> b n", b=B)
    q_skip_bad = torch.zeros(B, N_ATOM)  # missing c_atom dim
    p_skip = torch.zeros(B, N_ATOM, K, C_ATOMPAIR)
    c_skip = torch.zeros(B, N_ATOM, C_ATOM)
    c = torch.zeros(B, N_ATOM, C_ATOM)
    s = torch.zeros(B, N_RES, C_TOKEN)
    z = torch.zeros(B, N_RES, N_RES, C_PAIR)
    with pytest.raises(TypeCheckError):
        decoder(q_skip_bad, p_skip, c_skip, c, s, z, tok)

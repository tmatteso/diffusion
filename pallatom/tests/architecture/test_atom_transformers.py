import pytest
import torch
import torch.nn.functional as F
from beartype import beartype
from einops import einsum, rearrange, reduce
from jaxtyping import Float, jaxtyped

from architecture.atom_transformers import (
    AtomAttentionDecoder,
    AtomFeatureEncoder,
    AtomTransformer,
    AtomTransformerBlock,
    build_sparse_pairs,
)

torch.manual_seed(42)

N_RES = 8
ATOMS_PER_RES = 3
N_ATOM = N_RES * ATOMS_PER_RES   # 24
E = 4                             # element one-hot dim
C_ATOM = 32
C_ATOMPAIR = 16
C_TOKEN = 32
C_PAIR = 32
F_REF_DIM = ATOMS_PER_RES * (3 + E)
B = 1

# K is dynamic — pre-compute once from the canonical tok_idx
_base_tok = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
_base_nbrs, _ = build_sparse_pairs(_base_tok)
K = _base_nbrs.size(1)


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------

@jaxtyped(typechecker=beartype)
def pairwise_sq_dist(
    x: Float[torch.Tensor, "N D"],
) -> Float[torch.Tensor, "N N"]:
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    return einsum(diff, diff, "n m d, n m d -> n m")


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def block():
    return AtomTransformerBlock(C_ATOM, C_ATOMPAIR, n_heads=4).eval()


@pytest.fixture
def transformer():
    return AtomTransformer(C_ATOM, C_ATOMPAIR, n_blocks=2, n_heads=4).eval()


@pytest.fixture
def encoder():
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
def decoder():
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
def tok_idx():
    """Unbatched token index — used for build_sparse_pairs and deriving batched version."""
    return torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)


@pytest.fixture
def tok_idx_batched(tok_idx):
    """Batched token index [B, N_atom] — passed to encoder/decoder."""
    return tok_idx.unsqueeze(0).expand(B, -1).contiguous()


@pytest.fixture
def neighbor_idx(tok_idx):
    nbrs, _ = build_sparse_pairs(tok_idx)
    return nbrs


@pytest.fixture
def valid_mask(tok_idx):
    _, mask = build_sparse_pairs(tok_idx)
    return mask.unsqueeze(0)   # [B=1, N_atom, K]


@pytest.fixture
def ref_pos():
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def ref_element():
    return F.one_hot(torch.randint(0, E, (B, N_ATOM)), num_classes=E).float()


@pytest.fixture
def ref_space_uid():
    return torch.zeros(B, N_ATOM, dtype=torch.long)


@pytest.fixture
def s_input():
    return torch.randn(B, N_RES, C_TOKEN)


@pytest.fixture
def z_input():
    return torch.randn(B, N_RES, N_RES, C_PAIR)


@pytest.fixture
def r_scaled():
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def q_batched():
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def c_batched():
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def p_batched():
    return torch.randn(B, N_ATOM, K, C_ATOMPAIR)


@pytest.fixture
def q_skip():
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def c_skip():
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def p_skip():
    return torch.randn(B, N_ATOM, K, C_ATOMPAIR)


@pytest.fixture
def c_l():
    return torch.randn(B, N_ATOM, C_ATOM)


# ---------------------------------------------------------------------------
# build_sparse_pairs
# ---------------------------------------------------------------------------

def test_build_sparse_pairs_output_shapes(tok_idx):
    nbrs, mask = build_sparse_pairs(tok_idx)
    assert nbrs.shape == (N_ATOM, K)
    assert mask.shape == (N_ATOM, K)
    assert mask.dtype == torch.bool


def test_build_sparse_pairs_all_valid_within_window(tok_idx):
    # N_RES=8 < WINDOW_SIZE//2=16, so every atom neighbours every other atom
    _, mask = build_sparse_pairs(tok_idx)
    assert mask.all()


def test_build_sparse_pairs_self_is_neighbour(tok_idx):
    nbrs, _ = build_sparse_pairs(tok_idx)
    for i in range(N_ATOM):
        assert (nbrs[i] == i).any()


def test_build_sparse_pairs_padding_indices_in_range(tok_idx):
    nbrs, _ = build_sparse_pairs(tok_idx)
    assert (nbrs >= 0).all()
    assert (nbrs < N_ATOM).all()


def test_build_sparse_pairs_window_excludes_far_atoms():
    # 64 residues, 1 atom each, window_size=8 → half=4
    n_res = 64
    tok = torch.arange(n_res, dtype=torch.long)
    nbrs, mask = build_sparse_pairs(tok, window_size=8)
    half = 4
    first_valid = nbrs[0][mask[0]]
    last_valid  = nbrs[-1][mask[-1]]
    assert (first_valid < half).all()
    assert (last_valid >= n_res - half).all()


# ---------------------------------------------------------------------------
# AtomTransformerBlock
# ---------------------------------------------------------------------------

def test_atom_transformer_block_output_shape(block, q_batched, c_batched, p_batched, neighbor_idx, valid_mask):
    with torch.no_grad():
        out = block(q_batched, c_batched, p_batched, neighbor_idx, valid_mask)
    assert out.shape == (B, N_ATOM, C_ATOM)


def test_atom_transformer_block_output_finite(block, q_batched, c_batched, p_batched, neighbor_idx, valid_mask):
    with torch.no_grad():
        out = block(q_batched, c_batched, p_batched, neighbor_idx, valid_mask)
    assert torch.isfinite(out).all()


def test_atom_transformer_block_output_dtype(block, q_batched, c_batched, p_batched, neighbor_idx, valid_mask):
    with torch.no_grad():
        out = block(q_batched, c_batched, p_batched, neighbor_idx, valid_mask)
    assert out.dtype == q_batched.dtype


def test_atom_transformer_block_gradient_flows(block, q_batched, c_batched, p_batched, neighbor_idx, valid_mask):
    q_g = q_batched.clone().requires_grad_(True)
    out = block(q_g, c_batched, p_batched, neighbor_idx, valid_mask)
    reduce(out, "b n c -> ", "sum").backward()
    assert q_g.grad is not None
    assert torch.isfinite(q_g.grad).all()


# ---------------------------------------------------------------------------
# AtomTransformer
# ---------------------------------------------------------------------------

def test_atom_transformer_output_shape(transformer, q_batched, c_batched, p_batched, neighbor_idx, valid_mask):
    with torch.no_grad():
        out = transformer(q_batched, c_batched, p_batched, neighbor_idx, valid_mask)
    assert out.shape == (B, N_ATOM, C_ATOM)


def test_atom_transformer_output_finite(transformer, q_batched, c_batched, p_batched, neighbor_idx, valid_mask):
    with torch.no_grad():
        out = transformer(q_batched, c_batched, p_batched, neighbor_idx, valid_mask)
    assert torch.isfinite(out).all()


def test_atom_transformer_block_count():
    t = AtomTransformer(C_ATOM, C_ATOMPAIR, n_blocks=3, n_heads=4)
    assert len(t.blocks) == 3


def test_atom_transformer_gradient_flows(transformer, q_batched, c_batched, p_batched, neighbor_idx, valid_mask):
    q_g = q_batched.clone().requires_grad_(True)
    out = transformer(q_g, c_batched, p_batched, neighbor_idx, valid_mask)
    reduce(out, "b n c -> ", "sum").backward()
    assert q_g.grad is not None
    assert torch.isfinite(q_g.grad).all()


# ---------------------------------------------------------------------------
# AtomFeatureEncoder
# ---------------------------------------------------------------------------

def test_atom_feature_encoder_output_shapes(encoder, ref_pos, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok_idx_batched):
    with torch.no_grad():
        s_i, q_skip, c_skip, p_skip, c_l = encoder(
            ref_pos, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok_idx_batched,
        )
    assert s_i.shape    == (B, N_RES,  C_TOKEN)
    assert q_skip.shape == (B, N_ATOM, C_ATOM)
    assert c_skip.shape == (B, N_ATOM, C_ATOM)
    assert p_skip.shape == (B, N_ATOM, K, C_ATOMPAIR)
    assert c_l.shape    == (B, N_ATOM, C_ATOM)


def test_atom_feature_encoder_outputs_finite(encoder, ref_pos, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok_idx_batched):
    with torch.no_grad():
        outputs = encoder(
            ref_pos, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok_idx_batched,
        )
    for t in outputs:
        assert torch.isfinite(t).all()


def test_atom_feature_encoder_s_i_varies_across_residues(encoder, ref_pos, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok_idx_batched):
    with torch.no_grad():
        s_i, *_ = encoder(
            ref_pos, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok_idx_batched,
        )
    off_diag = pairwise_sq_dist(s_i[0]) + torch.eye(N_RES) * 1e10
    assert off_diag.min().item() > 0


def test_atom_feature_encoder_p_skip_k_matches_sparse_index(encoder, ref_pos, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok_idx, tok_idx_batched):
    with torch.no_grad():
        _, _, _, p_skip, _ = encoder(
            ref_pos, ref_element, ref_space_uid, s_input, z_input, r_scaled, tok_idx_batched,
        )
    nbrs, _ = build_sparse_pairs(tok_idx)
    assert p_skip.shape[2] == nbrs.size(1)  # K dim is index 2 in (B, N_atom, K, c_atompair)


def test_atom_feature_encoder_gradient_flows_to_r_scaled(encoder, ref_pos, ref_element, ref_space_uid, s_input, z_input, tok_idx_batched):
    r_scaled_g = torch.randn(B, N_ATOM, 3, requires_grad=True)
    _, q_skip, *_ = encoder(
        ref_pos, ref_element, ref_space_uid, s_input, z_input, r_scaled_g, tok_idx_batched,
    )
    reduce(q_skip, "b n c -> ", "sum").backward()
    assert r_scaled_g.grad is not None
    assert torch.isfinite(r_scaled_g.grad).all()


# ---------------------------------------------------------------------------
# AtomAttentionDecoder
# ---------------------------------------------------------------------------

def test_atom_attention_decoder_r_update_shape(decoder, q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched):
    with torch.no_grad():
        r_update, _ = decoder(q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    assert r_update.shape == (B, N_ATOM, 3)


def test_atom_attention_decoder_r_update_finite(decoder, q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched):
    with torch.no_grad():
        r_update, _ = decoder(q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    assert torch.isfinite(r_update).all()


def test_atom_attention_decoder_c_out_shape(decoder, q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched):
    with torch.no_grad():
        _, c_out = decoder(q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    assert c_out.shape == (B, N_ATOM, C_ATOM)


def test_atom_attention_decoder_c_out_finite(decoder, q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched):
    with torch.no_grad():
        _, c_out = decoder(q_skip, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    assert torch.isfinite(c_out).all()


def test_atom_attention_decoder_gradient_flows_to_q_skip(decoder, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched):
    q_skip_g = torch.randn(B, N_ATOM, C_ATOM, requires_grad=True)
    r_update, _ = decoder(q_skip_g, p_skip, c_skip, c_l, s_input, z_input, tok_idx_batched)
    reduce(r_update, "b n d -> ", "sum").backward()
    assert q_skip_g.grad is not None
    assert torch.isfinite(q_skip_g.grad).all()

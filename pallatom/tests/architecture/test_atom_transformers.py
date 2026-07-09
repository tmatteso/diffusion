"""Tests for atom-level transformer layers.

Covers build_sparse_pairs, compute_beta, scatter_mean,
ConditionedTransitionBlock, DiffusionTransformer, AtomTransformer,
AtomFeatureEncoder, and AtomAttentionDecoder, including shape contracts,
numerical stability, gradient flow, and sparse regression checks for the
K < N attention-bias shape mismatch.
"""

import dataclasses

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
    compute_beta,
    scatter_mean,
)
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Int, TypeCheckError, jaxtyped
from train.train_config import ModelParams

_ = manual_seed(42)

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
NON_ZERO_BETA = -1e10
ATOM_TRANSFORMER_BLOCKS = 3
ATOM_TRANSFORMER_HEADS = 4
DIFFUSION_TRANSFORMER_BLOCKS = 3

# K is dynamic — pre-compute once from the canonical tok_idx
_base_tok = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
_base_nbrs, _ = build_sparse_pairs(_base_tok)
K = _base_nbrs.size(1)

# Sparse-mismatch regression constants.
# With a small window_size, K < N_ATOM, reproducing the production crash where
# attn (B, n_heads, N, N) and pair_bias (B, n_heads, N, K) had mismatched
# last dims.
_SPARSE_WINDOW = 4  # small window so K << N_ATOM
_SPARSE_RES = 20  # number of residues — gives N_ATOM_SPARSE = 20 * 2 = 40
_ATOMS_PER_RES_SPARSE = 2
N_ATOM_SPARSE = _SPARSE_RES * _ATOMS_PER_RES_SPARSE
_sparse_tok = torch.repeat_interleave(
    torch.arange(_SPARSE_RES),
    _ATOMS_PER_RES_SPARSE,
)
_sparse_nbrs, _sparse_mask_raw = build_sparse_pairs(
    _sparse_tok,
    window_size=_SPARSE_WINDOW,
)
K_SPARSE = _sparse_nbrs.size(
    1,
)  # guaranteed < N_ATOM_SPARSE for _SPARSE_WINDOW small enough


# ---------------------------------------------------------------------------
# Input bundles — collapse wide fixture lists for encoder/decoder tests
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EncoderInputs:
    """All AtomFeatureEncoder forward-pass inputs in one bundle."""

    ref_pos: Float[torch.Tensor, "B N_atom 3"]
    ref_element: Float[torch.Tensor, "B N_atom E"]
    ref_space_uid: Int[torch.Tensor, "B N_atom"]
    s_input: Float[torch.Tensor, "B N_res C_token"]
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"]
    r_scaled: Float[torch.Tensor, "B N_atom 3"]
    tok_idx_batched: Int[torch.Tensor, "B N_atom"]


@dataclasses.dataclass(frozen=True)
class DecoderInputs:
    """All AtomAttentionDecoder forward-pass inputs in one bundle."""

    q_skip: Float[torch.Tensor, "B N_atom C_atom"]
    p_skip: Float[torch.Tensor, "B N_atom K C_atompair"]
    c_skip: Float[torch.Tensor, "B N_atom C_atom"]
    c_l: Float[torch.Tensor, "B N_atom C_atom"]
    s_input: Float[torch.Tensor, "B N_res C_token"]
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"]
    tok_idx_batched: Int[torch.Tensor, "B N_atom"]


# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def pairwise_sq_dist(
    x: Float[torch.Tensor, "N D"],
) -> Float[torch.Tensor, "N N"]:
    """Compute pairwise squared Euclidean distances between N vectors.

    Used in tests to verify that output embeddings differ across positions.
    """
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    return einsum(diff, diff, "n m d, n m d -> n m")


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conditioned_transition_block() -> ConditionedTransitionBlock:
    """Provide a ConditionedTransitionBlock with c_a=c_s=C_ATOM in eval mode.

    Returns:
        A ConditionedTransitionBlock instance ready for inference.
    """
    return ConditionedTransitionBlock(
        c_a=C_ATOM,
        c_s=C_ATOM,
        expansion=2,
    ).eval()


@pytest.fixture
def diffusion_transformer() -> DiffusionTransformer:
    """Provide a DiffusionTransformer with 2 blocks in eval mode.

    Returns:
        A DiffusionTransformer instance ready for inference.
    """
    return DiffusionTransformer(
        c_a=C_ATOM,
        c_s=C_ATOM,
        c_pair=C_PAIR,
        N_block=2,
        N_head=N_HEADS,
    ).eval()


@pytest.fixture
def transformer() -> AtomTransformer:
    """Provide an AtomTransformer with 2 blocks in eval mode.

    Returns:
        An AtomTransformer instance ready for inference.
    """
    return AtomTransformer(C_ATOM, C_ATOMPAIR, n_blocks=2, n_heads=4).eval()


@pytest.fixture
def model_params() -> ModelParams:
    """Provide ModelParams sized to the small test dimensions.

    Returns:
        A ModelParams instance with test-scaled channel dims and 1 block each.
    """
    return ModelParams(
        f_ref_dim=F_REF_DIM,
        c_res=C_TOKEN,
        c_pair=C_PAIR,
        c_atom=C_ATOM,
        c_atompair=C_ATOMPAIR,
        n_blocks_atom_transformer_encoder=1,
        n_heads_atom_transformer_encoder=4,
        n_blocks_atom_transformer_decoder=1,
        n_heads_atom_transformer_decoder=4,
    )


@pytest.fixture
def encoder(model_params: ModelParams) -> AtomFeatureEncoder:
    """Provide an AtomFeatureEncoder in eval mode.

    Args:
        model_params: Fixture providing the shared ModelParams configuration.

    Returns:
        An AtomFeatureEncoder instance ready for inference.
    """
    return AtomFeatureEncoder(
        c=C_TOKEN,
        d=C_ATOMPAIR,
        m=C_ATOM,
        model_params=model_params,
    ).eval()


@pytest.fixture
def decoder(model_params: ModelParams) -> AtomAttentionDecoder:
    """Provide an AtomAttentionDecoder in eval mode.

    Args:
        model_params: Fixture providing the shared ModelParams configuration.

    Returns:
        An AtomAttentionDecoder instance ready for inference.
    """
    return AtomAttentionDecoder(model_params=model_params).eval()


# ---------------------------------------------------------------------------
# Input tensor fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tok_idx() -> Int[torch.Tensor, "N_atom"]:
    """Unbatched token index for build_sparse_pairs and the batched version.

    Returns:
        Int tensor of shape [N_atom] mapping each atom to its parent residue.
    """
    return torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)


@pytest.fixture
def tok_idx_batched(
    tok_idx: Int[torch.Tensor, "N_atom"],
) -> Int[torch.Tensor, "B N_atom"]:
    """Batched token index [B, N_atom] — passed to encoder/decoder.

    Args:
        tok_idx: Fixture providing the unbatched [N_atom] token index.

    Returns:
        Int tensor of shape [B, N_atom] repeating tok_idx along batch.
    """
    return repeat(tok_idx, "n -> b n", b=B).contiguous()


@pytest.fixture
def neighbor_idx(
    tok_idx: Int[torch.Tensor, "N_atom"],
) -> Int[torch.Tensor, "B N_atom K"]:
    """Sparse neighbor index [B=1, N_atom, K] from the canonical tok_idx.

    Args:
        tok_idx: Fixture providing the unbatched [N_atom] token index.

    Returns:
        Int tensor of shape [B, N_atom, K] with neighbour indices per atom.
    """
    nbrs, _ = build_sparse_pairs(tok_idx)
    return rearrange(nbrs, "n k -> 1 n k")


@pytest.fixture
def valid_mask(
    tok_idx: Int[torch.Tensor, "N_atom"],
) -> Bool[torch.Tensor, "B N_atom K"]:
    """Bool mask [B=1, N_atom, K] — True for real atoms, False for padding.

    Args:
        tok_idx: Fixture providing the unbatched [N_atom] token index.

    Returns:
        Bool tensor of shape [B, N_atom, K] for valid sparse neighbour slots.
    """
    _, mask = build_sparse_pairs(tok_idx)
    return rearrange(mask, "n k -> 1 n k")  # [B=1, N_atom, K]


@pytest.fixture
def ref_pos() -> Float[torch.Tensor, "B N_atom 3"]:
    """Random reference atom positions [B, N_atom, 3] for the encoder.

    Returns:
        Float tensor of shape [B, N_atom, 3] with i.i.d. normal entries.
    """
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def ref_element() -> Float[torch.Tensor, "B N_atom E"]:
    """Random one-hot element features [B, N_atom, E] (C/N/O/UNK per atom).

    Returns:
        Float tensor of shape [B, N_atom, E] with one-hot rows.
    """
    return F.one_hot(torch.randint(0, E, (B, N_ATOM)), num_classes=E).float()


@pytest.fixture
def ref_space_uid() -> Int[torch.Tensor, "B N_atom"]:
    """All-zero space UIDs [B, N_atom] — every atom in one chain/space.

    Returns:
        Int tensor of shape [B, N_atom] filled with zeros.
    """
    return torch.zeros(B, N_ATOM, dtype=torch.long)


@pytest.fixture
def s_input() -> Float[torch.Tensor, "B N_res C_token"]:
    """Random residue single embedding [B, N_res, C_token] for encoder.

    Returns:
        Float tensor of shape [B, N_res, C_token] with i.i.d. normal entries.
    """
    return torch.randn(B, N_RES, C_TOKEN)


@pytest.fixture
def z_input() -> Float[torch.Tensor, "B N_res N_res C_pair"]:
    """Trunk pair embedding [B, N_res, N_res, C_pair] for encoder/decoder.

    Returns:
        Float tensor [B, N_res, N_res, C_pair] with i.i.d. normal entries.
    """
    return torch.randn(B, N_RES, N_RES, C_PAIR)


@pytest.fixture
def a_res() -> Float[torch.Tensor, "B N_res C_atom"]:
    """Atom embeddings [B, N_res, C_atom] for transition/transformer tests.

    Returns:
        Float tensor of shape [B, N_res, C_atom] with i.i.d. normal entries.
    """
    return torch.randn(B, N_RES, C_ATOM)


@pytest.fixture
def beta_dense() -> Float[torch.Tensor, "B N_res N_res"]:
    """Dense window-bias mask [B, N_res, N_res]; 0 in-window, -1e10 outside.

    Returns:
        Float tensor of shape [B, N_res, N_res] filled with zeros.
    """
    return torch.zeros(B, N_RES, N_RES)


@pytest.fixture
def r_scaled() -> Float[torch.Tensor, "B N_atom 3"]:
    """Random atom positions [B, N_atom, 3] as encoder conditioning signal.

    Returns:
        Float tensor of shape [B, N_atom, 3] with i.i.d. normal entries.
    """
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def q_batched() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Random atom query embedding [B, N_atom, C_atom] for transformer tests.

    Returns:
        Float tensor of shape [B, N_atom, C_atom] with i.i.d. normal entries.
    """
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def c_batched() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Random atom context embedding [B, N_atom, C_atom] for key/value.

    Returns:
        Float tensor of shape [B, N_atom, C_atom] with i.i.d. normal entries.
    """
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def p_batched() -> Float[torch.Tensor, "B N_atom K C_atompair"]:
    """Sparse atom-pair embedding [B, N_atom, K, C_atompair] for tests.

    Returns:
        Float tensor [B, N_atom, K, C_atompair] with i.i.d. normal entries.
    """
    return torch.randn(B, N_ATOM, K, C_ATOMPAIR)


@pytest.fixture
def q_skip() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Random atom skip-connection query [B, N_atom, C_atom] per decoder unit.

    Returns:
        Float tensor of shape [B, N_atom, C_atom] with i.i.d. normal entries.
    """
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def c_skip() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Random atom skip-connection context [B, N_atom, C_atom] per decoder unit.

    Returns:
        Float tensor of shape [B, N_atom, C_atom] with i.i.d. normal entries.
    """
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def p_skip() -> Float[torch.Tensor, "B N_atom K C_atompair"]:
    """Random sparse pair skip embed [B, N_atom, K, C_atompair] per decoder.

    Returns:
        Float tensor [B, N_atom, K, C_atompair] with i.i.d. normal entries.
    """
    return torch.randn(B, N_ATOM, K, C_ATOMPAIR)


@pytest.fixture
def c_l() -> Float[torch.Tensor, "B N_atom C_atom"]:
    """Atom-level latent embedding [B, N_atom, C_atom] refined by the decoder.

    Returns:
        Float tensor of shape [B, N_atom, C_atom] with i.i.d. normal entries.
    """
    return torch.randn(B, N_ATOM, C_ATOM)


@pytest.fixture
def a_res_bad() -> Float[torch.Tensor, "B N_res"]:
    """Wrong-shape residue embedding [B, N_res] missing the channel dim.

    Returns:
        Float tensor of shape [B, N_res], 2-D instead of the expected 3-D.
    """
    return torch.zeros(B, N_RES)


@pytest.fixture
def q_batched_bad() -> Float[torch.Tensor, "B N_atom"]:
    """Wrong-shape atom query embedding [B, N_atom] missing the channel dim.

    Returns:
        Float tensor of shape [B, N_atom], 2-D instead of the expected 3-D.
    """
    return torch.zeros(B, N_ATOM)


@pytest.fixture
def ref_pos_bad() -> Float[torch.Tensor, "B N_atom"]:
    """Wrong-shape reference positions [B, N_atom] missing the coordinate dim.

    Returns:
        Float tensor of shape [B, N_atom], 2-D instead of the expected 3-D.
    """
    return torch.zeros(B, N_ATOM)


@pytest.fixture
def q_skip_bad() -> Float[torch.Tensor, "B N_atom"]:
    """Wrong-shape skip query embedding [B, N_atom] missing the channel dim.

    Returns:
        Float tensor of shape [B, N_atom], 2-D instead of the expected 3-D.
    """
    return torch.zeros(B, N_ATOM)


@pytest.fixture
def encoder_inputs() -> EncoderInputs:
    """All AtomFeatureEncoder inputs bundled; builds tensors directly.

    Returns:
        EncoderInputs with random ref_pos, ref_element, ref_space_uid,
        s_input, z_input, r_scaled, and tok_idx_batched.
    """
    return EncoderInputs(
        ref_pos=torch.randn(B, N_ATOM, 3),
        ref_element=F.one_hot(
            torch.randint(0, E, (B, N_ATOM)),
            num_classes=E,
        ).float(),
        ref_space_uid=torch.zeros(B, N_ATOM, dtype=torch.long),
        s_input=torch.randn(B, N_RES, C_TOKEN),
        z_input=torch.randn(B, N_RES, N_RES, C_PAIR),
        r_scaled=torch.randn(B, N_ATOM, 3),
        tok_idx_batched=repeat(_base_tok, "n -> b n", b=B).contiguous(),
    )


@pytest.fixture
def decoder_inputs() -> DecoderInputs:
    """All AtomAttentionDecoder inputs bundled; builds tensors directly.

    Returns:
        DecoderInputs with random q_skip, p_skip, c_skip, c_l, s_input,
        z_input, and tok_idx_batched.
    """
    return DecoderInputs(
        q_skip=torch.randn(B, N_ATOM, C_ATOM),
        p_skip=torch.randn(B, N_ATOM, K, C_ATOMPAIR),
        c_skip=torch.randn(B, N_ATOM, C_ATOM),
        c_l=torch.randn(B, N_ATOM, C_ATOM),
        s_input=torch.randn(B, N_RES, C_TOKEN),
        z_input=torch.randn(B, N_RES, N_RES, C_PAIR),
        tok_idx_batched=repeat(_base_tok, "n -> b n", b=B).contiguous(),
    )


# ---------------------------------------------------------------------------
# build_sparse_pairs
# ---------------------------------------------------------------------------


def test_build_sparse_pairs_output_shapes(
    tok_idx: Int[torch.Tensor, "N_atom"],
) -> None:
    """Nbrs has shape [N_atom, K]; mask has same shape with bool dtype."""
    nbrs, mask = build_sparse_pairs(tok_idx)
    assert nbrs.shape == (N_ATOM, K)
    assert mask.shape == (N_ATOM, K)
    assert mask.dtype == torch.bool


def test_build_sparse_pairs_all_valid_within_window(
    tok_idx: Int[torch.Tensor, "N_atom"],
) -> None:
    """When N_res < half-window, every atom is neighbour to every other atom.

    Verifies that the validity mask is all-True when the structure fits entirely
    within a single attention window.
    """
    # N_RES=8 < WINDOW_SIZE//2=16, so every atom neighbours every other atom
    _, mask = build_sparse_pairs(tok_idx)
    assert mask.all()


def test_build_sparse_pairs_self_is_neighbour(
    tok_idx: Int[torch.Tensor, "N_atom"],
) -> None:
    """Every atom index appears in its own neighbour row (self-attention).

    Each atom's own index is present in the neighbour list returned
    by build_sparse_pairs.
    """
    nbrs, _ = build_sparse_pairs(tok_idx)
    for i in range(N_ATOM):
        assert (nbrs[i] == i).any()


def test_build_sparse_pairs_padding_indices_in_range(
    tok_idx: Int[torch.Tensor, "N_atom"],
) -> None:
    """Padding slots re-use valid indices, not out-of-bounds sentinels.

    Verifies that every entry in the neighbor index tensor is a valid atom index
    in [0, N_ATOM), preventing out-of-bounds access during gather operations.
    """
    nbrs, _ = build_sparse_pairs(tok_idx)
    assert (nbrs >= 0).all()
    assert (nbrs < N_ATOM).all()


def test_build_sparse_pairs_window_excludes_far_atoms() -> None:
    """Atoms beyond the half-window are excluded from sparse pairs."""
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
# compute_beta
# ---------------------------------------------------------------------------


def test_compute_beta_output_shape(
    neighbor_idx: Int[torch.Tensor, "B N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """compute_beta returns float [B, N_atom, K] matching the ref dtype."""
    ref = torch.randn(B, N_ATOM, C_ATOM)
    beta = compute_beta(
        neighbor_idx,
        valid_mask,
        n_queries=32,
        n_keys=128,
        ref=ref,
    )
    assert beta.shape == (B, N_ATOM, K)
    assert beta.dtype == ref.dtype


def test_compute_beta_values_are_binary(
    neighbor_idx: Int[torch.Tensor, "B N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """Every element of beta is exactly 0.0 or -1e10; no intermediate values."""
    ref = torch.randn(B, N_ATOM, C_ATOM)
    beta = compute_beta(
        neighbor_idx,
        valid_mask,
        n_queries=32,
        n_keys=128,
        ref=ref,
    )
    assert ((beta == 0.0) | (beta == NON_ZERO_BETA)).all()


def test_compute_beta_zero_for_near_atoms() -> None:
    """An atom pair sharing a window centre receives beta=0.0.

    n_queries=4, n_keys=8: centre 0 at 1.5 (half_q=2.0, half_k=4.0).
    l=0: |0-1.5|=1.5 < 2.0 → query; m=1: |1-1.5|=0.5 < 4.0 → key → beta=0.0.
    """
    N_t, n_q, n_k, B_t = 8, 4, 8, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> b n k", b=B_t, n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(
        neighbor_idx_t,
        valid_mask_t,
        n_queries=n_q,
        n_keys=n_k,
        ref=ref,
    )
    assert beta[0, 0, 1].item() == 0.0


def test_compute_beta_large_neg_for_far_atoms() -> None:
    """An atom pair from disjoint windows receives beta=-1e10.

    n_queries=n_keys=4: centres at 1.5 and 5.5 (half_q=half_k=2.0).
    l=0 is a query for centre 0 (|0-1.5|=1.5<2.0).
    m=7: |7-1.5|=5.5 > 2.0 → not a key for centre 0.
    l=0: |0-5.5|=5.5 > 2.0 → not a query for centre 1.
    No centre admits this pair → beta=-1e10.
    """
    N_t, n_q, n_k, B_t = 8, 4, 4, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> b n k", b=B_t, n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(
        neighbor_idx_t,
        valid_mask_t,
        n_queries=n_q,
        n_keys=n_k,
        ref=ref,
    )
    assert beta[0, 0, 7].item() == NON_ZERO_BETA


def test_compute_beta_valid_mask_forces_neg() -> None:
    """valid_mask=False forces -1e10 even for geometrically in-window pairs."""
    N_t, n_q, n_k, B_t = 8, 4, 8, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> b n k", b=B_t, n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    valid_mask_t[0, 0, 1] = False  # pair (l=0, m=1) is in-window but masked
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(
        neighbor_idx_t,
        valid_mask_t,
        n_queries=n_q,
        n_keys=n_k,
        ref=ref,
    )
    assert beta[0, 0, 1].item() == NON_ZERO_BETA  # mask wins over geometry
    assert beta[0, 0, 0].item() == 0.0  # unmasked in-window pair is unaffected


def test_compute_beta_boundary_exclusive() -> None:
    """The window boundary is strict (<): the atom just outside is excluded.

    With n_queries=n_keys=4, centres are at 1.5 and 5.5 (half_q=half_k=2.0).
    Atom 3 is the last query for centre 0; atom 4 is the first query for centre
    1.
    Pair (l=3, m=4):
      centre 0: l=3 is query (|3-1.5|=1.5<2), m=4 is NOT key (|4-1.5|=2.5>2)
      centre 1: l=3 is NOT query (|3-5.5|=2.5>2)
    → beta=-1e10.
    Pair (l=3, m=3): centre 0 admits both → beta=0.0.
    """
    N_t, n_q, n_k, B_t = 8, 4, 4, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> b n k", b=B_t, n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(
        neighbor_idx_t,
        valid_mask_t,
        n_queries=n_q,
        n_keys=n_k,
        ref=ref,
    )
    assert beta[0, 3, 4].item() == NON_ZERO_BETA
    assert beta[0, 3, 3].item() == 0.0


def test_compute_beta_cross_window_centre_admits_pair() -> None:
    """A pair admitted by any centre (not just its own) gets 0.0.

    n_queries=4, n_keys=8 (half_q=2.0, half_k=4.0). Centres at 1.5 and 5.5.
    Pair (l=4, m=3):
      centre 0: l=4 is NOT a query (|4-1.5|=2.5>2.0)
      centre 1: l=4 IS a query (|4-5.5|=1.5<2.0), m=3 IS a key
      (|3-5.5|=2.5<4.0)
    → beta=0.0 because centre 1 admits both. Tests multi-centre reduce logic.
    """
    N_t, n_q, n_k, B_t = 8, 4, 8, 1
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> b n k", b=B_t, n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(
        neighbor_idx_t,
        valid_mask_t,
        n_queries=n_q,
        n_keys=n_k,
        ref=ref,
    )
    assert beta[0, 4, 3].item() == 0.0


def test_compute_beta_all_zero_when_n_leq_n_queries(
    neighbor_idx: Int[torch.Tensor, "B N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """When N <= n_queries, every valid pair is in-window — no -1e10 entries.

    N_ATOM=24 < n_queries=32: single centre at 15.5 (half_q=16.0) covers all
    atoms. n_keys=128, half_k=64.0: all atoms are keys too. Every valid pair
    gets beta=0.0.
    """
    ref = torch.randn(B, N_ATOM, C_ATOM)
    beta = compute_beta(
        neighbor_idx,
        valid_mask,
        n_queries=32,
        n_keys=128,
        ref=ref,
    )
    assert (beta[valid_mask] == 0.0).all()


def test_compute_beta_wrong_shape_raises() -> None:
    """A 1-D neighbor_idx (missing K dimension) triggers TypeCheckError."""
    neighbor_idx_bad = torch.zeros(N_ATOM, dtype=torch.long)  # 1-D, not [N, K]
    valid_mask_t = torch.ones(B, N_ATOM, K, dtype=torch.bool)
    ref = torch.zeros(B, N_ATOM, C_ATOM)
    with pytest.raises(TypeCheckError):
        _ = compute_beta(neighbor_idx_bad, valid_mask_t, 32, 128, ref)


def test_compute_beta_no_inf_in_output(
    neighbor_idx: Int[torch.Tensor, "B N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """compute_beta output contains no -inf values when ref dtype is float16.

    Regression guard for the float16 fill-value fix in compute_beta: the
    original -1e10 constant raises RuntimeError when cast to float16
    (overflow). After the fix, a dtype-safe value is used and the output
    contains no -inf.
    """
    ref = torch.zeros(B, N_ATOM, C_ATOM, dtype=torch.float16)
    beta = compute_beta(
        neighbor_idx,
        valid_mask,
        n_queries=32,
        n_keys=128,
        ref=ref,
    )
    assert not beta.isinf().any()


def test_compute_beta_asymmetric() -> None:
    """Beta is directional: (l=0, m=4) admitted while (l=4, m=0) is blocked.

    n_queries=2, n_keys=8, N=8: half_q=1.0, half_k=4.0.
    Centres at 0.5, 2.5, 4.5, 6.5. l=0 queries centre 0 (|0-0.5|=0.5 < 1.0);
    m=4 is a key (|4-0.5|=3.5 < 4.0) -> 0.0. l=4 queries centre 2
    (|4-4.5|=0.5 < 1.0); m=0 NOT a key (|0-4.5|=4.5 > 4.0) -> -1e10.
    """
    N_t, n_q, n_k, B_t = 8, 2, 8, 1
    neighbor_idx_t: Int[torch.Tensor, "B_t N_t K_t"] = repeat(
        torch.arange(N_t),
        "k -> b n k",
        b=B_t,
        n=N_t,
    )
    valid_mask_t: Bool[torch.Tensor, "B_t N_t N_t"] = torch.ones(
        B_t,
        N_t,
        N_t,
        dtype=torch.bool,
    )
    ref = torch.zeros(B_t, N_t, C_ATOM)
    beta = compute_beta(
        neighbor_idx_t,
        valid_mask_t,
        n_queries=n_q,
        n_keys=n_k,
        ref=ref,
    )
    assert (
        beta[0, 0, 4].item() == 0.0
    ), "pair (0,4) should be admitted by centre 0"
    assert (
        beta[0, 4, 0].item() == NON_ZERO_BETA
    ), "pair (4,0) blocked: m=0 outside key window of centre 2"


def test_compute_beta_all_neighbors_masked_row(
    neighbor_idx: Int[torch.Tensor, "B N_atom K"],
) -> None:
    """A fully-masked query row gets all -1e10; adjacent rows are unaffected."""
    mask = torch.ones(B, N_ATOM, K, dtype=torch.bool)
    mask[0, 0, :] = False
    ref = torch.randn(B, N_ATOM, C_ATOM)
    beta = compute_beta(
        neighbor_idx,
        mask,
        n_queries=32,
        n_keys=128,
        ref=ref,
    )
    assert (beta[0, 0, :] == NON_ZERO_BETA).all()
    assert (beta[0, 1, :][mask[0, 1, :]] == 0.0).all()


def test_compute_beta_batch_independence(
    neighbor_idx: Int[torch.Tensor, "B N_atom K"],
) -> None:
    """Different valid_masks per batch element produce independent betas."""
    valid_mask_two: Bool[torch.Tensor, "2 N_atom K"] = torch.zeros(
        2,
        N_ATOM,
        K,
        dtype=torch.bool,
    )
    valid_mask_two[0] = True  # batch 0: all valid; batch 1: all masked
    ref = torch.randn(2, N_ATOM, C_ATOM)
    neighbor_idx_two: Int[torch.Tensor, "2 N_atom K"] = repeat(
        neighbor_idx,
        "1 n k -> b n k",
        b=2,
    )
    # n_queries=32 > N_ATOM=24: single window covers all; all valid pairs 0.0
    beta = compute_beta(
        neighbor_idx_two,
        valid_mask_two,
        n_queries=32,
        n_keys=128,
        ref=ref,
    )
    assert (beta[1] == NON_ZERO_BETA).all()
    assert (beta[0][valid_mask_two[0]] == 0.0).all()


# ---------------------------------------------------------------------------
# ConditionedTransitionBlock
# ---------------------------------------------------------------------------


def test_conditioned_transition_block_output(
    conditioned_transition_block: ConditionedTransitionBlock,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
) -> None:
    """The block returns [B, N_res, C_atom] with no NaN or Inf."""
    with torch.no_grad():
        out = conditioned_transition_block(a_res, s_input)
    assert out.shape == (B, N_RES, C_ATOM)
    assert torch.isfinite(out).all()


def test_conditioned_transition_block_gradient_flows(
    conditioned_transition_block: ConditionedTransitionBlock,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
) -> None:
    """Gradients flow through AdaLN and SwiGLU gate to the atom embedding."""
    a_g = a_res.clone().requires_grad_(True)  # noqa:FBT003
    out = conditioned_transition_block(a_g, s_input)
    torch.autograd.backward([reduce(out, "b n c -> ", "sum")])
    assert a_g.grad is not None
    assert torch.isfinite(a_g.grad).all()


# ---------------------------------------------------------------------------
# DiffusionTransformer
# ---------------------------------------------------------------------------


def test_diffusion_transformer_output(
    diffusion_transformer: DiffusionTransformer,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    beta_dense: Float[torch.Tensor, "B N_res N_res"],
) -> None:
    """N_block rounds leave [B, N_res, C_atom] unchanged with no NaN or Inf."""
    with torch.no_grad():
        out = diffusion_transformer(a_res, s_input, z_input, beta_dense)
    assert out.shape == (B, N_RES, C_ATOM)
    assert torch.isfinite(out).all()


def test_diffusion_transformer_n_block_stored() -> None:
    """N_block constructor arg is stored and determines iteration depth."""
    dt = DiffusionTransformer(
        c_a=C_ATOM,
        c_s=C_ATOM,
        c_pair=C_PAIR,
        N_block=DIFFUSION_TRANSFORMER_BLOCKS,
        N_head=N_HEADS,
    )
    assert dt.N_block == DIFFUSION_TRANSFORMER_BLOCKS


def test_diffusion_transformer_gradient_flows(
    diffusion_transformer: DiffusionTransformer,
    a_res: Float[torch.Tensor, "B N_res C_atom"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
    beta_dense: Float[torch.Tensor, "B N_res N_res"],
) -> None:
    """Gradients propagate through all blocks to the atom embedding input."""
    a_g = a_res.clone().requires_grad_(True)  # noqa:FBT003
    out = diffusion_transformer(a_g, s_input, z_input, beta_dense)
    torch.autograd.backward([reduce(out, "b n c -> ", "sum")])
    assert a_g.grad is not None
    assert torch.isfinite(a_g.grad).all()


# ---------------------------------------------------------------------------
# AtomTransformer
# ---------------------------------------------------------------------------


def test_atom_transformer_output(
    transformer: AtomTransformer,
    q_batched: Float[torch.Tensor, "B N_atom C_atom"],
    c_batched: Float[torch.Tensor, "B N_atom C_atom"],
    p_batched: Float[torch.Tensor, "B N_atom K C_atompair"],
    neighbor_idx: Int[torch.Tensor, "B N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """Multiple stacked blocks leave [B, N_atom, C_atom] with no NaN or Inf."""
    with torch.no_grad():
        out = transformer(
            q_batched,
            c_batched,
            p_batched,
            neighbor_idx,
            valid_mask,
        )
    assert out.shape == (B, N_ATOM, C_ATOM)
    assert torch.isfinite(out).all()


def test_atom_transformer_block_count() -> None:
    """n_blocks is stored and forwarded to DiffusionTransformer.N_block."""
    t = AtomTransformer(
        C_ATOM,
        C_ATOMPAIR,
        n_blocks=ATOM_TRANSFORMER_BLOCKS,
        n_heads=ATOM_TRANSFORMER_HEADS,
    )
    assert t.blocks.N_block == ATOM_TRANSFORMER_BLOCKS


def test_atom_transformer_gradient_flows(
    transformer: AtomTransformer,
    q_batched: Float[torch.Tensor, "B N_atom C_atom"],
    c_batched: Float[torch.Tensor, "B N_atom C_atom"],
    p_batched: Float[torch.Tensor, "B N_atom K C_atompair"],
    neighbor_idx: Int[torch.Tensor, "B N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """Gradients propagate through all stacked blocks to the query input."""
    q_g = q_batched.clone().requires_grad_(True)  # noqa:FBT003
    out = transformer(q_g, c_batched, p_batched, neighbor_idx, valid_mask)
    torch.autograd.backward([reduce(out, "b n c -> ", "sum")])
    assert q_g.grad is not None
    assert torch.isfinite(q_g.grad).all()


def test_atom_transformer_block_isolation() -> None:
    """Block-0 output is unchanged when block-1 inputs are scrambled.

    With n_queries=n_keys=4 and N=8, blocks [0..3] and [4..7] map to disjoint
    window centres (1.5 and 5.5). Since exp(-1e10)=0.0 in float32, cross-block
    attention weights are exactly zero, so block-1 inputs cannot affect block-0
    outputs.
    """
    n_q, N_t, B_t = 4, 8, 1
    model = AtomTransformer(
        C_ATOM,
        C_ATOMPAIR,
        n_blocks=1,
        n_heads=1,
        n_queries=n_q,
        n_keys=n_q,
    ).eval()
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> b n k", b=B_t, n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)

    _ = manual_seed(0)
    q = torch.randn(B_t, N_t, C_ATOM)
    c = torch.randn(B_t, N_t, C_ATOM)
    p = torch.randn(B_t, N_t, N_t, C_ATOMPAIR)

    with torch.no_grad():
        out1 = model(q, c, p, neighbor_idx_t, valid_mask_t)

    _ = manual_seed(1)
    q_alt = q.clone()
    c_alt = c.clone()
    q_alt[:, n_q:, :] = torch.randn(B_t, n_q, C_ATOM)
    c_alt[:, n_q:, :] = torch.randn(B_t, n_q, C_ATOM)

    with torch.no_grad():
        out2 = model(q_alt, c_alt, p, neighbor_idx_t, valid_mask_t)

    assert torch.equal(out1[:, :n_q, :], out2[:, :n_q, :])


def test_atom_transformer_gradient_isolation() -> None:
    """Backprop from block-0 outputs gives zero gradient for block-1 inputs.

    With n_queries=n_keys=4 and N=8, the two blocks are disjoint. Softmax
    weights for cross-block pairs are exactly 0.0 in float32, so the softmax
    gradient (s_lm * (delta - s_lm) = 0 when s_lm=0) propagates zero back to
    block-1 inputs. ConditionedTransitionBlock is pointwise, so it contributes
    no cross-block gradient.
    """
    n_q, N_t, B_t = 4, 8, 1
    model = AtomTransformer(
        C_ATOM,
        C_ATOMPAIR,
        n_blocks=1,
        n_heads=1,
        n_queries=n_q,
        n_keys=n_q,
    ).eval()
    neighbor_idx_t = repeat(torch.arange(N_t), "k -> b n k", b=B_t, n=N_t)
    valid_mask_t = torch.ones(B_t, N_t, N_t, dtype=torch.bool)

    _ = manual_seed(0)
    q = torch.randn(B_t, N_t, C_ATOM, requires_grad=True)
    c = torch.randn(B_t, N_t, C_ATOM, requires_grad=True)
    p = torch.randn(B_t, N_t, N_t, C_ATOMPAIR)

    out = model(q, c, p, neighbor_idx_t, valid_mask_t)
    torch.autograd.backward([reduce(out[:, :n_q, :], "b n c -> ", "sum")])

    assert q.grad is not None
    assert c.grad is not None
    assert q.grad[:, n_q:, :].abs().max().item() == 0.0
    assert c.grad[:, n_q:, :].abs().max().item() == 0.0


# ---------------------------------------------------------------------------
# AtomFeatureEncoder
# ---------------------------------------------------------------------------


def test_atom_feature_encoder_outputs(
    encoder: AtomFeatureEncoder,
    encoder_inputs: EncoderInputs,
) -> None:
    """Encoder returns five tensors with correct shapes and no NaN or Inf."""
    inp = encoder_inputs
    with torch.no_grad():
        s_i, enc_q_skip, enc_c_skip, enc_p_skip, enc_c_l = encoder(
            inp.ref_pos,
            inp.ref_element,
            inp.ref_space_uid,
            inp.s_input,
            inp.z_input,
            inp.r_scaled,
            inp.tok_idx_batched,
        )
    assert s_i.shape == (B, N_RES, C_TOKEN)
    assert enc_q_skip.shape == (B, N_ATOM, C_ATOM)
    assert enc_c_skip.shape == (B, N_ATOM, C_ATOM)
    assert enc_p_skip.shape == (B, N_ATOM, K, C_ATOMPAIR)
    assert enc_c_l.shape == (B, N_ATOM, C_ATOM)
    for t in (s_i, enc_q_skip, enc_c_skip, enc_p_skip, enc_c_l):
        assert torch.isfinite(t).all()


def test_atom_feature_encoder_s_i_varies_across_residues(
    encoder: AtomFeatureEncoder,
    encoder_inputs: EncoderInputs,
) -> None:
    """Residue embeddings are distinct — encoder does not collapse them."""
    inp = encoder_inputs
    with torch.no_grad():
        s_i, *_ = encoder(
            inp.ref_pos,
            inp.ref_element,
            inp.ref_space_uid,
            inp.s_input,
            inp.z_input,
            inp.r_scaled,
            inp.tok_idx_batched,
        )
    off_diag = pairwise_sq_dist(s_i[0]) + torch.eye(N_RES) * 1e10
    assert off_diag.min().item() > 0


def test_atom_feature_encoder_p_skip_k_matches_sparse_index(
    encoder: AtomFeatureEncoder,
    encoder_inputs: EncoderInputs,
    tok_idx: Int[torch.Tensor, "N_atom"],
) -> None:
    """p_skip K dim matches sparse neighbour count from build_sparse_pairs."""
    inp = encoder_inputs
    with torch.no_grad():
        _, _, _, enc_p_skip, _ = encoder(
            inp.ref_pos,
            inp.ref_element,
            inp.ref_space_uid,
            inp.s_input,
            inp.z_input,
            inp.r_scaled,
            inp.tok_idx_batched,
        )
    nbrs, _ = build_sparse_pairs(tok_idx)
    assert enc_p_skip.shape[2] == nbrs.size(
        1,
    )  # K dim is index 2 in (B, N_atom, K, c_atompair)


def test_atom_feature_encoder_gradient_flows_to_r_scaled(
    encoder: AtomFeatureEncoder,
    encoder_inputs: EncoderInputs,
) -> None:
    """The encoder is differentiable with respect to r_scaled."""
    inp = encoder_inputs
    r_scaled_g = torch.randn(B, N_ATOM, 3, requires_grad=True)
    _, enc_q_skip, *_ = encoder(
        inp.ref_pos,
        inp.ref_element,
        inp.ref_space_uid,
        inp.s_input,
        inp.z_input,
        r_scaled_g,
        inp.tok_idx_batched,
    )
    torch.autograd.backward([reduce(enc_q_skip, "b n c -> ", "sum")])
    assert r_scaled_g.grad is not None
    assert torch.isfinite(r_scaled_g.grad).all()


# ---------------------------------------------------------------------------
# AtomAttentionDecoder
# ---------------------------------------------------------------------------


def test_atom_attention_decoder_outputs(
    decoder: AtomAttentionDecoder,
    decoder_inputs: DecoderInputs,
) -> None:
    """Decoder returns [B, N_atom, 3] and [B, N_atom, C_atom], no NaN/Inf."""
    inp = decoder_inputs
    with torch.no_grad():
        _, _, r_update, c_out = decoder(
            inp.q_skip,
            inp.p_skip,
            inp.c_skip,
            inp.c_l,
            inp.s_input,
            inp.z_input,
            inp.tok_idx_batched,
        )
    assert r_update.shape == (B, N_ATOM, 3)
    assert torch.isfinite(r_update).all()
    assert c_out.shape == (B, N_ATOM, C_ATOM)
    assert torch.isfinite(c_out).all()


def test_atom_attention_decoder_gradient_flows_to_q_skip(
    decoder: AtomAttentionDecoder,
    decoder_inputs: DecoderInputs,
) -> None:
    """Position update is differentiable with respect to q_skip."""
    inp = decoder_inputs
    q_skip_g = torch.randn(B, N_ATOM, C_ATOM, requires_grad=True)
    _, _, r_update, _ = decoder(
        q_skip_g,
        inp.p_skip,
        inp.c_skip,
        inp.c_l,
        inp.s_input,
        inp.z_input,
        inp.tok_idx_batched,
    )
    torch.autograd.backward([reduce(r_update, "b n d -> ", "sum")])
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
def sparse_neighbor_idx() -> Int[torch.Tensor, "B N_atom_sparse K_sparse"]:
    """Sparse neighbour index [B=1, N_ATOM_SPARSE, K_SPARSE] with K < N."""
    return repeat(_sparse_nbrs, "n k -> 1 n k")


@pytest.fixture
def sparse_valid_mask() -> Bool[torch.Tensor, "B N_atom_sparse K_sparse"]:
    """Validity mask [B, N_ATOM_SPARSE, K_SPARSE] for the sparse scenario."""
    return repeat(_sparse_mask_raw, "n k -> b n k", b=B)


@pytest.fixture
def q_sparse() -> Float[torch.Tensor, "B N_atom_sparse C_atom"]:
    """Query atom embeddings [B, N_ATOM_SPARSE, C_ATOM] for sparse tests."""
    return torch.randn(B, N_ATOM_SPARSE, C_ATOM)


@pytest.fixture
def p_sparse() -> Float[torch.Tensor, "B N_atom_sparse K_sparse C_atompair"]:
    """Sparse pair embeddings [B, N_ATOM_SPARSE, K_SPARSE, C_ATOMPAIR]."""
    return torch.randn(B, N_ATOM_SPARSE, K_SPARSE, C_ATOMPAIR)


@pytest.fixture
def z_sparse_dt() -> Float[torch.Tensor, "B N_atom_sparse K_sparse C_atom"]:
    """Sparse pair embeddings for DiffusionTransformer [B, N, K, C_ATOM]."""
    return torch.randn(B, N_ATOM_SPARSE, K_SPARSE, C_ATOM)


@pytest.fixture
def beta_sparse_dt() -> Float[torch.Tensor, "B N_atom_sparse K_sparse"]:
    """Sparse attention bias [B, N, K] aligned with z_sparse_dt."""
    return torch.zeros(B, N_ATOM_SPARSE, K_SPARSE)


def test_diffusion_transformer_sparse_no_shape_mismatch(
    z_sparse_dt: Float[torch.Tensor, "B N_atom_sparse K_sparse C_atom"],
    beta_sparse_dt: Float[torch.Tensor, "B N_atom_sparse K_sparse"],
    sparse_neighbor_idx: Int[torch.Tensor, "B N_atom_sparse K_sparse"],
) -> None:
    """DiffusionTransformer with sparse z (K<N) and neighbor_idx works."""
    model = DiffusionTransformer(
        c_a=C_ATOM,
        c_s=C_ATOM,
        c_pair=C_ATOM,
        N_block=1,
        N_head=N_HEADS,
    ).eval()
    a = torch.randn(B, N_ATOM_SPARSE, C_ATOM)
    s = torch.randn(B, N_ATOM_SPARSE, C_ATOM)
    with torch.no_grad():
        out = model(
            a,
            s,
            z_sparse_dt,
            beta_sparse_dt,
            neighbor_idx=sparse_neighbor_idx,
        )
    assert out.shape == (B, N_ATOM_SPARSE, C_ATOM)
    assert torch.isfinite(out).all()


def test_atom_transformer_sparse_k_lt_n_no_shape_mismatch(
    q_sparse: Float[torch.Tensor, "B N_atom_sparse C_atom"],
    p_sparse: Float[torch.Tensor, "B N_atom_sparse K_sparse C_atompair"],
    sparse_neighbor_idx: Int[torch.Tensor, "B N_atom_sparse K_sparse"],
    sparse_valid_mask: Bool[torch.Tensor, "B N_atom_sparse K_sparse"],
) -> None:
    """AtomTransformer with K_SPARSE < N_ATOM_SPARSE raises no RuntimeError.

    This is the exact shape regime that caused the production crash:
        attn (B, n_heads, N, N) + pair_bias (B, n_heads, N, K) with N != K.
    """
    model = AtomTransformer(
        C_ATOM,
        C_ATOMPAIR,
        n_blocks=1,
        n_heads=N_HEADS,
    ).eval()
    c = torch.randn(B, N_ATOM_SPARSE, C_ATOM)
    with torch.no_grad():
        out = model(
            q_sparse,
            c,
            p_sparse,
            sparse_neighbor_idx,
            sparse_valid_mask,
        )
    assert out.shape == (B, N_ATOM_SPARSE, C_ATOM)
    assert torch.isfinite(out).all()


def test_atom_transformer_sparse_gradient_flows(
    p_sparse: Float[torch.Tensor, "B N_atom_sparse K_sparse C_atompair"],
    sparse_neighbor_idx: Int[torch.Tensor, "B N_atom_sparse K_sparse"],
    sparse_valid_mask: Bool[torch.Tensor, "B N_atom_sparse K_sparse"],
) -> None:
    """Gradients flow through the sparse attention path when K < N."""
    model = AtomTransformer(
        C_ATOM,
        C_ATOMPAIR,
        n_blocks=1,
        n_heads=N_HEADS,
    ).eval()
    q_g = torch.randn(B, N_ATOM_SPARSE, C_ATOM, requires_grad=True)
    c = torch.randn(B, N_ATOM_SPARSE, C_ATOM)
    out = model(q_g, c, p_sparse, sparse_neighbor_idx, sparse_valid_mask)
    torch.autograd.backward([reduce(out, "b n c -> ", "sum")])
    assert q_g.grad is not None
    assert torch.isfinite(q_g.grad).all()


# ---------------------------------------------------------------------------
# scatter_mean fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sm_src_uniform() -> Float[torch.Tensor, "1 4 1"]:
    """Source tensor (1, 4, 1) with values 0, 2, 4, 6 — two atoms per residue.

    Returns:
        Float tensor of shape [1, 4, 1] for uniform-group testing.
    """
    return torch.tensor([[[0.0], [2.0], [4.0], [6.0]]])


@pytest.fixture
def sm_index_uniform() -> Int[torch.Tensor, "1 4"]:
    """Segment index (1, 4): atoms 0-1 in seg 0; atoms 2-3 in seg 1.

    Returns:
        Int tensor [[0, 0, 1, 1]].
    """
    return torch.tensor([[0, 0, 1, 1]])


@pytest.fixture
def sm_src_nonuniform() -> Float[torch.Tensor, "1 5 1"]:
    """Source (1, 5, 1) values [1, 3, 5, 7, 9]: 3 then 2 atoms per residue.

    Returns:
        Float tensor of shape [1, 5, 1] for non-uniform group testing.
    """
    return torch.tensor([[[1.0], [3.0], [5.0], [7.0], [9.0]]])


@pytest.fixture
def sm_index_nonuniform() -> Int[torch.Tensor, "1 5"]:
    """Segment index (1, 5): atoms 0-2 in seg 0; atoms 3-4 in seg 1.

    Returns:
        Int tensor [[0, 0, 0, 1, 1]].
    """
    return torch.tensor([[0, 0, 0, 1, 1]])


@pytest.fixture
def sm_src_one_per_res() -> Float[torch.Tensor, "2 4 6"]:
    """Random source features (2, 4, 6) — one atom per residue, two batch items.

    Returns:
        Float tensor of shape [2, 4, 6] with i.i.d. standard normal entries.
    """
    return torch.randn(2, 4, 6)


@pytest.fixture
def sm_index_one_per_res() -> Int[torch.Tensor, "2 4"]:
    """Batch-offset identity index (2, 4) — each atom maps to its own segment.

    Returns:
        Int tensor [[0, 1, 2, 3], [4, 5, 6, 7]].
    """
    base = repeat(torch.arange(4), "n -> b n", b=2)
    offset = repeat(torch.arange(2) * 4, "b -> b n", n=4)
    return base + offset


@pytest.fixture
def sm_src_constant() -> Float[torch.Tensor, "2 6 4"]:
    """Constant source features (2, 6, 4) filled with 3.14, 3 atoms/residue.

    Returns:
        Float tensor of shape [2, 6, 4] with every entry equal to 3.14.
    """
    return torch.full((2, 6, 4), 3.14)


@pytest.fixture
def sm_index_constant() -> Int[torch.Tensor, "2 6"]:
    """Batch-offset index (2, 6): 3 atoms per residue, 2 residues per batch.

    Returns:
        Int tensor [[0, 0, 0, 1, 1, 1], [2, 2, 2, 3, 3, 3]].
    """
    tok = torch.repeat_interleave(torch.arange(2), 3)
    tok = repeat(tok, "n -> b n", b=2)
    offset = repeat(torch.arange(2) * 2, "b -> b n", n=6)
    return tok + offset


@pytest.fixture
def sm_src_batch_indep() -> Float[torch.Tensor, "2 8 6"]:
    """Random source features (2, 8, 6): 2 atoms/res, 4 residues, 2 batches.

    Returns:
        Float tensor of shape [2, 8, 6] with i.i.d. normal entries.
    """
    return torch.randn(2, 8, 6)


@pytest.fixture
def sm_src_batch_indep_zeroed(
    sm_src_batch_indep: Float[torch.Tensor, "2 8 6"],
) -> Float[torch.Tensor, "2 8 6"]:
    """Clone of sm_src_batch_indep with batch item 1 zeroed out.

    Args:
        sm_src_batch_indep: Fixture providing the [2, 8, 6] source tensor.

    Returns:
        Float [2, 8, 6]; batch 0 identical to sm_src_batch_indep,
        batch 1 all zeros.
    """
    src0 = sm_src_batch_indep.clone()
    src0[1] = 0.0
    return src0


@pytest.fixture
def sm_index_batch_indep() -> Int[torch.Tensor, "2 8"]:
    """Batch-offset index (2, 8): 2 atoms per residue, 4 residues per batch.

    Returns:
        Int tensor mapping atoms [0,0,1,1,2,2,3,3] with batch stride 4.
    """
    tok_base = torch.repeat_interleave(torch.arange(4), 2)
    tok = repeat(tok_base, "n -> b n", b=2)
    offset = repeat(torch.arange(2) * 4, "b -> b n", n=8)
    return tok + offset


@pytest.fixture
def sm_src_multichannel() -> Float[torch.Tensor, "1 12 5"]:
    """Random source features (1, 12, 5): 4 atoms/residue, 3 residues, 1 batch.

    Returns:
        Float tensor of shape [1, 12, 5] with i.i.d. standard normal entries.
    """
    return torch.randn(1, 12, 5)


@pytest.fixture
def sm_index_multichannel() -> Int[torch.Tensor, "1 12"]:
    """Index (1, 12) mapping 4 atoms per residue across 3 residues, 1 batch.

    Returns:
        Int tensor [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]].
    """
    tok = torch.repeat_interleave(torch.arange(3), 4)
    return repeat(tok, "n -> b n", b=1)


@pytest.fixture
def sm_src_bad() -> Float[torch.Tensor, "B N_atom"]:
    """Wrong-shape source (B, N_atom) missing the channel dim for scatter_mean.

    Returns:
        Float tensor of shape [B, N_atom], 2-D instead of the expected 3-D.
    """
    return torch.zeros(B, N_ATOM)


@pytest.fixture
def sm_index_bad() -> Int[torch.Tensor, "B N_atom"]:
    """All-zero int index (B, N_atom) paired with sm_src_bad for shape testing.

    Returns:
        Int tensor of shape [B, N_atom] filled with zeros.
    """
    return torch.zeros(B, N_ATOM, dtype=torch.long)


# ---------------------------------------------------------------------------
# scatter_mean
# ---------------------------------------------------------------------------


def test_scatter_mean_known_values_uniform(
    sm_src_uniform: Float[torch.Tensor, "1 4 1"],
    sm_index_uniform: Int[torch.Tensor, "1 4"],
) -> None:
    """With two atoms per residue, scatter_mean produces mean of each pair."""
    out = scatter_mean(sm_src_uniform, sm_index_uniform, 2, 1)
    expected = torch.tensor([[[1.0], [5.0]]])  # mean(0,2)=1, mean(4,6)=5
    assert torch.allclose(out, expected)


def test_scatter_mean_known_values_nonuniform(
    sm_src_nonuniform: Float[torch.Tensor, "1 5 1"],
    sm_index_nonuniform: Int[torch.Tensor, "1 5"],
) -> None:
    """scatter_mean handles variable group sizes (3 and 2 atoms per residue)."""
    out = scatter_mean(sm_src_nonuniform, sm_index_nonuniform, 2, 1)
    expected = torch.tensor([[[3.0], [8.0]]])  # mean(1,3,5)=3, mean(7,9)=8
    assert torch.allclose(out, expected)


def test_scatter_mean_one_atom_per_residue(
    sm_src_one_per_res: Float[torch.Tensor, "2 4 6"],
    sm_index_one_per_res: Int[torch.Tensor, "2 4"],
) -> None:
    """With one atom per residue, scatter_mean is identity over atoms."""
    out = scatter_mean(sm_src_one_per_res, sm_index_one_per_res, 2 * 4, 2)
    assert torch.allclose(out, sm_src_one_per_res)


def test_scatter_mean_constant_src_returns_that_constant(
    sm_src_constant: Float[torch.Tensor, "2 6 4"],
    sm_index_constant: Int[torch.Tensor, "2 6"],
) -> None:
    """Constant values average to themselves regardless of group size."""
    out = scatter_mean(sm_src_constant, sm_index_constant, 2 * 2, 2)
    assert torch.allclose(out, torch.full((2, 2, 4), 3.14))


def test_scatter_mean_batch_items_independent(
    sm_src_batch_indep: Float[torch.Tensor, "2 8 6"],
    sm_src_batch_indep_zeroed: Float[torch.Tensor, "2 8 6"],
    sm_index_batch_indep: Int[torch.Tensor, "2 8"],
) -> None:
    """Zeroing one batch item must not affect scatter_mean for other items."""
    out_full = scatter_mean(sm_src_batch_indep, sm_index_batch_indep, 2 * 4, 2)
    out_zero = scatter_mean(
        sm_src_batch_indep_zeroed,
        sm_index_batch_indep,
        2 * 4,
        2,
    )
    assert torch.allclose(out_full[0], out_zero[0])


def test_scatter_mean_multichannel_mean_matches_per_channel(
    sm_src_multichannel: Float[torch.Tensor, "1 12 5"],
    sm_index_multichannel: Int[torch.Tensor, "1 12"],
) -> None:
    """scatter_mean agrees with reshape+reduce mean across all C channels."""
    out = scatter_mean(sm_src_multichannel, sm_index_multichannel, 1 * 3, 1)
    src_grouped: Float[torch.Tensor, "Bsm Ntgt atoms_per Csm"] = rearrange(
        sm_src_multichannel,
        "b (n a) c -> b n a c",
        n=3,
        a=4,
    )
    expected: Float[torch.Tensor, "Bsm Ntgt Csm"] = reduce(
        src_grouped,
        "b n a c -> b n c",
        "mean",
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_scatter_mean_wrong_shape(
    sm_src_bad: Float[torch.Tensor, "B N_atom"],
    sm_index_bad: Int[torch.Tensor, "B N_atom"],
) -> None:
    """Wrong src ndim (2-D instead of 3-D) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        _ = scatter_mean(sm_src_bad, sm_index_bad, N_RES, B)


# ---------------------------------------------------------------------------
# Shape-contract enforcement — negative tests
# ---------------------------------------------------------------------------


def test_conditioned_transition_block_forward_wrong_shape(
    conditioned_transition_block: ConditionedTransitionBlock,
    a_res_bad: Float[torch.Tensor, "B N_res"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
) -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        _ = conditioned_transition_block(a_res_bad, s_input)


def test_diffusion_transformer_forward_wrong_shape(
    diffusion_transformer: DiffusionTransformer,
    a_res_bad: Float[torch.Tensor, "B N_res"],
    s_input: Float[torch.Tensor, "B N_res C_token"],
    z_input: Float[torch.Tensor, "B N_res N_res C_pair"],
) -> None:
    """Wrong a ndim (2-D instead of 3-D) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        _ = diffusion_transformer(a_res_bad, s_input, z_input)


def test_atom_transformer_forward_wrong_shape(
    transformer: AtomTransformer,
    q_batched_bad: Float[torch.Tensor, "B N_atom"],
    c_batched: Float[torch.Tensor, "B N_atom C_atom"],
    p_batched: Float[torch.Tensor, "B N_atom K C_atompair"],
    neighbor_idx: Int[torch.Tensor, "B N_atom K"],
    valid_mask: Bool[torch.Tensor, "B N_atom K"],
) -> None:
    """Wrong q ndim (2-D instead of 3-D) triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        _ = transformer(
            q_batched_bad,
            c_batched,
            p_batched,
            neighbor_idx,
            valid_mask,
        )


def test_atom_feature_encoder_forward_wrong_shape(
    encoder: AtomFeatureEncoder,
    encoder_inputs: EncoderInputs,
    ref_pos_bad: Float[torch.Tensor, "B N_atom"],
) -> None:
    """Wrong ref_pos ndim (2-D instead of 3-D) triggers TypeCheckError."""
    inp = encoder_inputs
    with pytest.raises(TypeCheckError):
        _ = encoder(
            ref_pos_bad,
            inp.ref_element,
            inp.ref_space_uid,
            inp.s_input,
            inp.z_input,
            inp.r_scaled,
            inp.tok_idx_batched,
        )


def test_atom_attention_decoder_forward_wrong_shape(
    decoder: AtomAttentionDecoder,
    decoder_inputs: DecoderInputs,
    q_skip_bad: Float[torch.Tensor, "B N_atom"],
) -> None:
    """Wrong q_skip ndim (2-D instead of 3-D) triggers TypeCheckError."""
    inp = decoder_inputs
    with pytest.raises(TypeCheckError):
        _ = decoder(
            q_skip_bad,
            inp.p_skip,
            inp.c_skip,
            inp.c_l,
            inp.s_input,
            inp.z_input,
            inp.tok_idx_batched,
        )

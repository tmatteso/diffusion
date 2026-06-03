"""Tests for the main trunk forward pass."""

import dataclasses

import pytest
import torch
import torch.nn.functional as F
from architecture.losses import (
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    seq_ce_loss,
    smooth_lddt_loss,
)
from architecture.main_trunk import (
    AtomDistogramHead,
    MainTrunk,
    PredictedOutputs,
    RelativePositionEncoding,
    ResidueDistogramHead,
    TimeFourierEmbedding,
)
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from helpers.featurize import FeaturizedBatch, sinusoidal_encoding
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Int, TypeCheckError, jaxtyped

# there should exist a set of enums that are imported here and in the configs
B = 2
N_RES = 50
ATOMS_PER_RES = 3
N_ATOM = N_RES * ATOMS_PER_RES  # 150
E = 4  # element one-hot dim
C_RES = 32
C_PAIR = 32
C_ATOM = 32
C_ATOMPAIR = 16
N_BINS = 38
N_ATOM_BINS = 22  # distinct from N_BINS; must match AtomDistogramParams.n_bins
K_UNIT = 2
# WINDOW_SIZE=128 (half=64); with N_RES=50 all residues fall within the half-window, so
# every atom neighbours every other atom: K = N_ATOM = 150
K_SPARSE = N_ATOM
F_REF_DIM = ATOMS_PER_RES * (
    3 + E
)  # encoder groups all sibling atoms: n_per_res*(pos_dim+elem_dim)

N_BLOCKS_ATOM_TRANSFORMER_ENCODER = 3
N_HEADS_ATOM_TRANSFORMER_ENCODER = 4
N_BLOCKS_ATOM_TRANSFORMER_DECODER = 3
N_HEADS_ATOM_TRANSFORMER_DECODER = 4
N_PAIRFORMER_BLOCKS_TEMPLATE_EMBEDDER = 2
N_PAIFORMER_HEADS_TEMPLATE_EMBEDDER = 16
SIGMA_DATA = 16
N_AMINO = 20
RESIDUE_NUMBER = 50
manual_seed(42)

# @pytest.fixture(autouse=True)
# def _reset_seed() -> None:
#     """Reset global RNG before every test so results are independent of test ordering."""
#     manual_seed(42)

# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def sq_dist_matrix(
    x: Float[torch.Tensor, "N D"],
) -> Float[torch.Tensor, "N N"]:
    """Compute pairwise squared Euclidean distances between N row-vectors."""
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    return einsum(diff, diff, "n m d, n m d -> n m")


@jaxtyped(typechecker=beartype)
def mean_abs_asymmetry(
    x: Float[torch.Tensor, "B N N D"],
) -> Float[torch.Tensor, ""]:
    """Return mean absolute difference between x[b,i,j] and x[b,j,i] — measures non-symmetry."""
    diff = x - rearrange(x, "b i j d -> b j i d")
    return reduce(diff.abs(), "b i j d -> ", "mean")


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> MainTrunk:
    """Provide a small MainTrunk instance in eval mode."""
    return MainTrunk(
        f_ref_dim=F_REF_DIM,
        n_bins=N_BINS,
        n_atom_bins=N_ATOM_BINS,
        c_atom=C_ATOM,
        c_pair=C_PAIR,
        c_res=C_RES,
        c_atompair=C_ATOMPAIR,
        K_unit=K_UNIT,
        n_blocks_atom_transformer_encoder=N_BLOCKS_ATOM_TRANSFORMER_ENCODER,
        n_heads_atom_transformer_encoder=N_HEADS_ATOM_TRANSFORMER_ENCODER,
        n_blocks_atom_transformer_decoder=N_BLOCKS_ATOM_TRANSFORMER_DECODER,
        n_heads_atom_transformer_decoder=N_HEADS_ATOM_TRANSFORMER_DECODER,
        n_pairformer_blocks_template_embedder=N_PAIRFORMER_BLOCKS_TEMPLATE_EMBEDDER,
        n_paiformer_heads_template_embedder=N_HEADS_ATOM_TRANSFORMER_DECODER,
        sigma_data=SIGMA_DATA,
        n_amino=N_AMINO,
        residue_number=RESIDUE_NUMBER,
    ).eval()


# ---------------------------------------------------------------------------
# Input tensor fixtures  (all have leading B dim)
# ---------------------------------------------------------------------------


@pytest.fixture
def ref_pos() -> Float[torch.Tensor, "B N_atom 3"]:
    """Provide random reference atom positions (B, N_ATOM, 3)."""
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def ref_element() -> Float[torch.Tensor, "B N_atom E"]:
    """Provide random one-hot element features (B, N_ATOM, E)."""
    return F.one_hot(torch.randint(0, E, (B, N_ATOM)), num_classes=E).float()


@pytest.fixture
def ref_space_uid() -> Int[torch.Tensor, "B N_atom"]:
    """Provide all-zero chain/space UIDs (B, N_ATOM) for a single-chain input."""
    return torch.zeros(B, N_ATOM, dtype=torch.long)


@pytest.fixture
def f_distogram() -> Float[torch.Tensor, "B N_res N_res N_bins"]:
    """One-hot template distogram [B, N_RES, N_RES, N_BINS] with random bin assignments."""
    return F.one_hot(torch.randint(0, N_BINS, (B, N_RES, N_RES)), num_classes=N_BINS).float()


@pytest.fixture
def f_pseudo_beta_mask() -> Float[torch.Tensor, "B N_res"]:
    """All-ones pseudo-β mask [B, N_RES] — every residue has a valid pseudo-β position."""
    return torch.ones(B, N_RES)


@pytest.fixture
def f_residue_idx() -> Int[torch.Tensor, "B N_res"]:
    """Integer residue position indices [B, N_RES] — monotonically increasing per batch item."""
    return repeat(torch.arange(N_RES, dtype=torch.long), "n -> b n", b=B).contiguous()


@pytest.fixture
def r_input() -> Float[torch.Tensor, "B N_atom 3"]:
    """Noisy atom positions [B, N_ATOM, 3] fed to the trunk as the diffusion denoising input."""
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def tok_idx() -> Int[torch.Tensor, "B N_atom"]:
    """Residue index for each atom [B, N_ATOM] — ATOMS_PER_RES atoms map to same residue."""
    single = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
    return repeat(single, "n -> b n", b=B).contiguous()


@pytest.fixture
def center_uid() -> Int[torch.Tensor, "B N_atom"]:
    """Center atom index per atom [B, N_ATOM]; all atoms in a residue share same center index."""
    res_centers = torch.arange(0, N_ATOM, ATOMS_PER_RES)  # [0, 3, 6, ..., 147] — one per residue
    single = repeat(
        res_centers, "n -> (n a)", a=ATOMS_PER_RES
    )  # broadcast to every atom in residue
    return repeat(single, "n -> b n", b=B).contiguous()


@pytest.fixture
def gt_atom_distogram_sparse() -> Float[torch.Tensor, "B N_atom K_sparse N_atom_bins"]:
    """Random ground-truth sparse atom distogram targets [B, N_ATOM, K_SPARSE, N_ATOM_BINS]."""
    return torch.randn(B, N_ATOM, K_SPARSE, N_ATOM_BINS)


@pytest.fixture
def gt_atom_distogram_mask_sparse() -> Bool[torch.Tensor, "B N_atom K_sparse"]:
    """All-True validity mask [B, N_ATOM, K_SPARSE] for sparse atom distogram — no padding slots."""
    return torch.ones(B, N_ATOM, K_SPARSE, dtype=torch.bool)


# ---------------------------------------------------------------------------
# FeaturizedBatch fixture and forward helper
# ---------------------------------------------------------------------------


@pytest.fixture
def featurized_batch(
    ref_pos: Float[torch.Tensor, "B N_atom 3"],
    ref_element: Float[torch.Tensor, "B N_atom E"],
    ref_space_uid: Int[torch.Tensor, "B N_atom"],
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
    f_residue_idx: Int[torch.Tensor, "B N_res"],
    r_input: Float[torch.Tensor, "B N_atom 3"],
    tok_idx: Int[torch.Tensor, "B N_atom"],
    center_uid: Int[torch.Tensor, "B N_res"],
    gt_atom_distogram_sparse: Float[torch.Tensor, "B N_atom K_sparse N_atom_bins"],
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K_sparse"],
) -> FeaturizedBatch:
    """Assemble a complete FeaturizedBatch from all input fixtures for end-to-end trunk tests."""
    return FeaturizedBatch(
        ref_pos=ref_pos,
        ref_element=ref_element,
        ref_space_uid=ref_space_uid,
        gt_res_distogram=f_distogram.long(),
        f_pseudo_beta_mask=f_pseudo_beta_mask.long(),
        f_residue_idx=f_residue_idx,
        r_gt=torch.zeros_like(r_input),
        r_gt_noised=r_input,
        atom5_mask=torch.ones(B, N_ATOM, dtype=torch.bool),
        aa_indices=torch.zeros(B, N_RES, dtype=torch.long),
        t_hat=torch.rand(B) + 0.1,
        t_normalized=torch.randn(B, N_RES, N_RES),
        tok_idx=tok_idx,
        center_uid=center_uid,
        gt_atom_distogram_sparse=gt_atom_distogram_sparse,
        gt_atom_distogram_mask_sparse=gt_atom_distogram_mask_sparse,
    )


def _forward(model: MainTrunk, batch: FeaturizedBatch) -> PredictedOutputs:
    """Run model forward pass under no_grad and return the PredictedOutputs dataclass."""
    with torch.no_grad():
        return model(batch)


# ---------------------------------------------------------------------------
# sinusoidal_encoding
# ---------------------------------------------------------------------------


def test_sinusoidal_encoding_output_shape():
    """sinusoidal_encoding returns [B, N_res, dim] with all finite values."""
    positions = repeat(torch.arange(N_RES, dtype=torch.float32), "n -> b n", b=2)
    out = sinusoidal_encoding(positions, dim=C_RES)
    assert out.shape == (2, N_RES, C_RES)
    assert torch.isfinite(out).all()


def test_sinusoidal_encoding_varies_across_positions():
    """Every residue position maps to a distinct encoding — no two rows are identical."""
    positions = repeat(torch.arange(N_RES, dtype=torch.float32), "n -> 1 n")
    enc = rearrange(sinusoidal_encoding(positions, dim=C_RES), "1 n d -> n d")
    off_diag = sq_dist_matrix(enc) + torch.eye(N_RES) * 1e10
    assert off_diag.min().item() > 0


# ---------------------------------------------------------------------------
# TimeFourierEmbedding
# ---------------------------------------------------------------------------


def test_time_fourier_embedding_output_shape(model: MainTrunk):
    """TimeFourierEmbedding maps a length-N noise level to [N, C_res] with all finite values."""
    out = model.time_fourier(torch.randn(N_RES))
    assert out.shape == (N_RES, C_RES)
    assert torch.isfinite(out).all()


def test_time_fourier_embedding_fixed_buffers():
    """Freqs and phases are non-trainable buffers, and outputs are bounded in [-1, 1]."""
    emb = TimeFourierEmbedding(C_RES)
    assert not emb.freqs.requires_grad
    assert not emb.phases.requires_grad
    assert len(list(emb.parameters())) == 0
    out = emb(torch.randn(N_RES))
    assert (out >= -1.0).all()
    assert (out <= 1.0).all()


# ---------------------------------------------------------------------------
# RelativePositionEncoding
# ---------------------------------------------------------------------------


def test_rel_pos_enc_output_shape(model: MainTrunk):
    """RelativePositionEncoding returns [B, N_res, N_res, C_pair] with all finite values."""
    res_idx = repeat(torch.arange(N_RES, dtype=torch.long), "n -> b n", b=B)
    zeros = torch.zeros(B, N_RES, dtype=torch.long)
    out = model.rel_pos_enc(
        residue_index=res_idx,
        asym_id=zeros,
        entity_id=zeros,
        token_index=res_idx,
        sym_id=zeros,
    )
    assert out.shape == (B, N_RES, N_RES, C_PAIR)
    assert torch.isfinite(out).all()


def test_rel_pos_enc_deterministic(model: MainTrunk):
    """RelativePositionEncoding is purely positional — same inputs always return same output."""
    res_idx = repeat(torch.arange(N_RES, dtype=torch.long), "n -> b n", b=B)
    zeros = torch.zeros(B, N_RES, dtype=torch.long)
    kwargs = {
        "residue_index": res_idx,
        "asym_id": zeros,
        "entity_id": zeros,
        "token_index": res_idx,
        "sym_id": zeros,
    }
    out1 = model.rel_pos_enc(**kwargs)
    out2 = model.rel_pos_enc(**kwargs)
    assert torch.allclose(out1, out2)


def test_rel_pos_enc_algo3_output_shape() -> None:
    """Algorithm 3 forward returns [1, N, N, C_pair] for batch-size-1 inputs."""
    enc = RelativePositionEncoding(C_PAIR, r_max=4, s_max=2)
    N = 10
    residue_index = rearrange(torch.arange(N, dtype=torch.long), "n -> 1 n")
    asym_id = torch.zeros(1, N, dtype=torch.long)
    entity_id = torch.zeros(1, N, dtype=torch.long)
    token_index = torch.zeros(1, N, dtype=torch.long)
    sym_id = torch.zeros(1, N, dtype=torch.long)
    out = enc(
        residue_index=residue_index,
        asym_id=asym_id,
        entity_id=entity_id,
        token_index=token_index,
        sym_id=sym_id,
    )
    assert out.shape == (1, N, N, C_PAIR)
    assert torch.isfinite(out).all()


def test_rel_pos_enc_algo3_cross_chain_ignores_residue_distance() -> None:
    """Cross-chain pairs always use the overflow bin regardless of residue_index values."""
    enc = RelativePositionEncoding(C_PAIR, r_max=4, s_max=2)
    N = 4
    asym_id = rearrange(torch.tensor([0, 0, 1, 1], dtype=torch.long), "n -> 1 n")
    entity_id = torch.zeros(1, N, dtype=torch.long)
    token_index = torch.zeros(1, N, dtype=torch.long)
    sym_id = rearrange(torch.tensor([0, 0, 1, 1], dtype=torch.long), "n -> 1 n")

    idx_a = rearrange(torch.tensor([0, 1, 0, 1], dtype=torch.long), "n -> 1 n")
    idx_b = rearrange(torch.tensor([0, 1, 50, 51], dtype=torch.long), "n -> 1 n")

    out_a = enc(
        residue_index=idx_a,
        asym_id=asym_id,
        entity_id=entity_id,
        token_index=token_index,
        sym_id=sym_id,
    )
    out_b = enc(
        residue_index=idx_b,
        asym_id=asym_id,
        entity_id=entity_id,
        token_index=token_index,
        sym_id=sym_id,
    )

    assert torch.allclose(out_a[0, 0, 2], out_b[0, 0, 2])
    assert torch.allclose(out_a[0, 1, 3], out_b[0, 1, 3])


def test_rel_pos_enc_algo3_same_chain_uses_residue_distance() -> None:
    """Same-chain pairs encode the clipped residue index difference, not the overflow bin."""
    enc = RelativePositionEncoding(C_PAIR, r_max=4, s_max=2)
    N = 6
    asym_id = torch.zeros(1, N, dtype=torch.long)
    entity_id = torch.zeros(1, N, dtype=torch.long)
    token_index = torch.zeros(1, N, dtype=torch.long)
    sym_id = torch.zeros(1, N, dtype=torch.long)

    idx_close = rearrange(torch.arange(N, dtype=torch.long), "n -> 1 n")
    idx_spread = rearrange(torch.arange(N, dtype=torch.long) * 3, "n -> 1 n")

    out_close = enc(
        residue_index=idx_close,
        asym_id=asym_id,
        entity_id=entity_id,
        token_index=token_index,
        sym_id=sym_id,
    )
    out_spread = enc(
        residue_index=idx_spread,
        asym_id=asym_id,
        entity_id=entity_id,
        token_index=token_index,
        sym_id=sym_id,
    )

    assert not torch.allclose(out_close[0, 0, 1], out_spread[0, 0, 1])


def test_rel_pos_enc_algo3_entity_flag_differs_across_entities() -> None:
    """b_same_entity is 1 iff entity_id[i]==entity_id[j]; differing entity_ids change output."""
    enc = RelativePositionEncoding(C_PAIR, r_max=4, s_max=2)
    N = 4
    residue_index = rearrange(torch.arange(N, dtype=torch.long), "n -> 1 n")
    asym_id = torch.zeros(1, N, dtype=torch.long)
    token_index = torch.zeros(1, N, dtype=torch.long)
    sym_id = torch.zeros(1, N, dtype=torch.long)

    entity_same = torch.zeros(1, N, dtype=torch.long)
    entity_split = rearrange(torch.tensor([0, 0, 1, 1], dtype=torch.long), "n -> 1 n")

    out_same = enc(
        residue_index=residue_index,
        asym_id=asym_id,
        entity_id=entity_same,
        token_index=token_index,
        sym_id=sym_id,
    )
    out_split = enc(
        residue_index=residue_index,
        asym_id=asym_id,
        entity_id=entity_split,
        token_index=token_index,
        sym_id=sym_id,
    )

    assert not torch.allclose(out_same[0, 0, 2], out_split[0, 0, 2])


def test_rel_pos_enc_algo3_chain_distance_varies_with_sym_id() -> None:
    """Cross-chain d_chain = clip(sym_i - sym_j + s_max, 0, 2*s_max); diff sym_ids change output."""
    enc = RelativePositionEncoding(C_PAIR, r_max=4, s_max=2)
    N = 4
    residue_index = rearrange(torch.arange(N, dtype=torch.long), "n -> 1 n")
    asym_id = rearrange(torch.tensor([0, 0, 1, 1], dtype=torch.long), "n -> 1 n")
    entity_id = torch.zeros(1, N, dtype=torch.long)
    token_index = torch.zeros(1, N, dtype=torch.long)

    sym_id_a = rearrange(torch.tensor([0, 0, 1, 1], dtype=torch.long), "n -> 1 n")
    sym_id_b = rearrange(torch.tensor([0, 0, 2, 2], dtype=torch.long), "n -> 1 n")

    out_a = enc(
        residue_index=residue_index,
        asym_id=asym_id,
        entity_id=entity_id,
        token_index=token_index,
        sym_id=sym_id_a,
    )
    out_b = enc(
        residue_index=residue_index,
        asym_id=asym_id,
        entity_id=entity_id,
        token_index=token_index,
        sym_id=sym_id_b,
    )

    assert not torch.allclose(out_a[0, 0, 2], out_b[0, 0, 2])


# ---------------------------------------------------------------------------
# ResidueDistogramHead
# ---------------------------------------------------------------------------


def test_residue_distogram_head_output_shape():
    """ResidueDistogramHead maps [B, N_res, N_res, C_pair] to finite [B, N_res, N_res, N_bins]."""
    head = ResidueDistogramHead(C_PAIR, n_bins=N_BINS)
    logits = head(torch.randn(B, N_RES, N_RES, C_PAIR))
    assert logits.shape == (B, N_RES, N_RES, N_BINS)
    assert torch.isfinite(logits).all()


def test_residue_distogram_head_output_symmetric():
    """The head symmetrises the pair embedding before projection so logits[i,j] == logits[j,i]."""
    head = ResidueDistogramHead(C_PAIR, n_bins=N_BINS)
    logits = head(torch.randn(B, N_RES, N_RES, C_PAIR))
    assert mean_abs_asymmetry(logits).item() < 1e-5


# ---------------------------------------------------------------------------
# AtomDistogramHead
# ---------------------------------------------------------------------------


def test_atom_distogram_head_output_shapes():
    """AtomDistogramHead returns finite logits [N_atom, N_atom, N_bins] and local-window mask."""
    head = AtomDistogramHead(C_ATOMPAIR, n_bins=N_BINS, atoms_per_res=ATOMS_PER_RES)
    logits, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert logits.shape == (N_ATOM, N_ATOM, N_BINS)
    assert mask.shape == (N_ATOM, N_ATOM)
    assert mask.dtype == torch.bool
    assert torch.isfinite(logits).all()
    assert (
        logits[mask].std(dim=-1).min().item() > 0
    )  # logits vary across bin dimension for local-window pairs


def test_atom_distogram_head_mask_includes_diagonal():
    """The local-window mask is True on the diagonal — every atom is within its own window."""
    head = AtomDistogramHead(C_ATOMPAIR, n_bins=N_BINS, atoms_per_res=ATOMS_PER_RES)
    _, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert mask.diagonal().all()


def test_atom_distogram_head_mask_symmetric():
    """If atom i is within the window of atom j, atom j must also be within the window of atom i."""
    head = AtomDistogramHead(C_ATOMPAIR, n_bins=N_BINS, atoms_per_res=ATOMS_PER_RES)
    _, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert torch.equal(mask, rearrange(mask, "i j -> j i"))


# ---------------------------------------------------------------------------
# MainTrunk.forward — output shapes and values
# ---------------------------------------------------------------------------


def test_main_trunk_r_denoised_shape_finite(model: MainTrunk, featurized_batch: FeaturizedBatch):
    """The main trunk returns denoised atom coordinates [B, N_atom, 3] with no NaN or Inf."""
    out = _forward(model, featurized_batch)
    assert out.r_denoised.shape == (B, N_ATOM, 3)
    assert torch.isfinite(out.r_denoised).all()


def test_main_trunk_residue_distogram_shape_finite(
    model: MainTrunk, featurized_batch: FeaturizedBatch
):
    """The residue distogram head returns [B, N_res, N_res, N_bins] logits with no NaN or Inf."""
    out = _forward(model, featurized_batch)
    assert out.residue_distogram_logits.shape == (B, N_RES, N_RES, N_BINS)
    assert torch.isfinite(out.residue_distogram_logits).all()


def test_main_trunk_atom_distogram_shape_finite(
    model: MainTrunk, featurized_batch: FeaturizedBatch
):
    """Atom distogram output shape is [B, N_ATOM, ..., N_ATOM_BINS]."""
    out = _forward(model, featurized_batch)
    assert out.atom_distogram_logits.ndim == 4
    assert out.atom_distogram_logits.shape[0] == B
    assert out.atom_distogram_logits.shape[1] == N_ATOM
    assert out.atom_distogram_logits.shape[3] == N_ATOM_BINS
    assert torch.isfinite(out.atom_distogram_logits).all()


def test_main_trunk_atom_distogram_bins_match_ground_truth(
    model: MainTrunk, featurized_batch: FeaturizedBatch
):
    """Number of predicted atom distance bins matches bin count in ground-truth sparse tensor."""
    out = _forward(model, featurized_batch)
    assert (
        out.atom_distogram_logits.shape[-1] == featurized_batch.gt_atom_distogram_sparse.shape[-1]
    )


def test_main_trunk_distogram_loss_atom_computable(
    model: MainTrunk, featurized_batch: FeaturizedBatch
):
    """Atom distogram logits from trunk passed directly to distogram_loss_atom without errors."""
    out = _forward(model, featurized_batch)
    loss = distogram_loss_atom(
        out.atom_distogram_logits,
        featurized_batch.gt_atom_distogram_sparse,
        featurized_batch.gt_atom_distogram_mask_sparse,
    ).mean()
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_main_trunk_sequence_head_shapes_and_vocab_bound(
    model: MainTrunk, featurized_batch: FeaturizedBatch
) -> None:
    """Sequence head output has correct shape, is finite, and never predicts the X/null slot."""
    out = _forward(model, featurized_batch)

    assert out.seq_logits.shape == (B, N_RES, 20)
    assert torch.isfinite(out.seq_logits).all()
    assert out.seq_logits.argmax(dim=-1).max() < 20  # X is never the argmax

    assert len(out.intermediate_denoised_coord_stack) == K_UNIT
    assert len(out.intermediate_pred_aa_logit_stack) == K_UNIT
    for inter_logits in out.intermediate_pred_aa_logit_stack:
        assert inter_logits.shape == (B, N_RES, 20)
        assert inter_logits.argmax(dim=-1).max() < 20  # X is never the argmax


def test_main_trunk_intermediate_coords_shape_finite(
    model: MainTrunk, featurized_batch: FeaturizedBatch
):
    """Every intermediate denoised coordinate tensor in stack has shape [B, N_atom, 3]."""
    out = _forward(model, featurized_batch)
    for r in out.intermediate_denoised_coord_stack:
        assert r.shape == (B, N_ATOM, 3)
        assert torch.isfinite(r).all()


def test_main_trunk_forward_with_mask_token_aa_indices(
    model: MainTrunk, featurized_batch: FeaturizedBatch
):
    """aa_indices of 20 (mask token 'X') must not raise an IndexError during the forward pass."""
    # aa_indices containing 20 (mask token "X") must not raise IndexError
    masked_batch = dataclasses.replace(
        featurized_batch,
        aa_indices=torch.full((B, N_RES), 20, dtype=torch.long),
    )
    with torch.no_grad():
        out = model(masked_batch)
    assert out.r_denoised.shape == (B, N_ATOM, 3)
    assert torch.isfinite(out.r_denoised).all()


# you need to add integration tests here. don't be afraid to use pytest mocks.


def _assert_submodule_grads(model: MainTrunk) -> None:
    """Assert every top-level submodule has at least one finite nonzero gradient.

    Args:
        model: Trunk module after a backward pass has been called.
    """
    buckets: dict[str, list[Float[torch.Tensor, "..."]]] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            buckets.setdefault(name.split(".")[0], []).append(param.grad)
    assert buckets, "no parameters have gradients — backward was not called"
    for prefix, grads in buckets.items():
        assert any(
            torch.isfinite(g).all().item() and g.abs().max().item() > 0 for g in grads
        ), f"submodule '{prefix}' has no finite nonzero gradients"


def test_integration_gradient_flow_composite_loss(
    model: MainTrunk, featurized_batch: FeaturizedBatch
) -> None:
    """Composite 7-term training loss propagates finite nonzero grads to every submodule."""
    model.train()
    pred: PredictedOutputs = model(featurized_batch)

    kabsch_loss = atom_loss(
        pred.r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask
    ).mean()
    ce_loss = seq_ce_loss(pred.seq_logits, featurized_batch.aa_indices)
    lddt = smooth_lddt_loss(
        pred.r_denoised, featurized_batch.r_gt, featurized_batch.atom5_mask, cutoff=15.0
    )
    gt_res_bin_idx = featurized_batch.gt_res_distogram.argmax(dim=-1).clamp(
        0, pred.residue_distogram_logits.size(-1) - 1
    )
    res_distogram_loss = distogram_loss_residue(
        pred.residue_distogram_logits, gt_res_bin_idx, featurized_batch.f_pseudo_beta_mask.bool()
    ).mean()
    atom_distogram_loss = distogram_loss_atom(
        pred.atom_distogram_logits,
        featurized_batch.gt_atom_distogram_sparse,
        featurized_batch.gt_atom_distogram_mask_sparse,
    ).mean()

    K_unit = len(pred.intermediate_denoised_coord_stack)
    intermediate_loss = torch.tensor(0.0)
    for k_idx, (inter_coords, inter_logits) in enumerate(
        zip(
            pred.intermediate_denoised_coord_stack,
            pred.intermediate_pred_aa_logit_stack,
            strict=False,
        )
    ):
        gamma: float = 0.99 ** (K_unit - k_idx - 1)
        k_loss = atom_loss(
            inter_coords, featurized_batch.r_gt, featurized_batch.atom5_mask
        ) + seq_ce_loss(inter_logits, featurized_batch.aa_indices)
        intermediate_loss = intermediate_loss + gamma * k_loss
    intermediate_loss = (intermediate_loss / max(K_unit, 1)).mean()

    total_loss = (
        kabsch_loss + ce_loss + lddt + res_distogram_loss + atom_distogram_loss + intermediate_loss
    )
    torch.autograd.backward([total_loss])

    _assert_submodule_grads(model)


def test_main_trunk_embed_inputs_wrong_type(model: MainTrunk) -> None:
    """Passing a non-FeaturizedBatch triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        model.embed_inputs("not a FeaturizedBatch")  # type: ignore[reportArgumentType]


def test_main_trunk_forward_wrong_type(model: MainTrunk) -> None:
    """Passing a non-FeaturizedBatch triggers TypeCheckError."""
    with pytest.raises(TypeCheckError):
        model("not a FeaturizedBatch")

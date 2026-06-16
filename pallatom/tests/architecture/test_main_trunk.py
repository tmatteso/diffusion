"""Tests for the main trunk forward pass.

Covers sinusoidal encoding, TimeFourierEmbedding, RelativePositionEncoding,
ResidueDistogramHead, AtomDistogramHead, and end-to-end MainTrunk forward
pass including output shapes, symmetry properties, loss computability, and
gradient flow through a composite training loss.
"""

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
from helpers.atom_utils import RESTYPE_NUM_NO_X
from helpers.batch_types import FeaturizedBatch
from helpers.featurize import sinusoidal_encoding
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Int, jaxtyped
from train.train_config import (
    AtomDistogramParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
)

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
# ModelParams.window_size=32 -> half=16;
# max neighbors = (32-1)*ATOMS_PER_RES = 93
MODEL_WINDOW_SIZE = 32
K_SPARSE = (MODEL_WINDOW_SIZE - 1) * ATOMS_PER_RES
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
N_AMINO = RESTYPE_NUM_NO_X
TOLERANCE = 1e-5
RESIDUE_NUMBER = 50
DISTOGRAM_RANK = 4
_ = manual_seed(42)

# ---------------------------------------------------------------------------
# Typed helpers
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def sq_dist_matrix(
    x: Float[torch.Tensor, "N D"],
) -> Float[torch.Tensor, "N N"]:
    """Compute pairwise squared Euclidean distances between N row-vectors.

    Args:
        x: Row vectors of shape (N, D).

    Returns:
        Symmetric (N, N) matrix of squared Euclidean distances.
    """
    diff = rearrange(x, "n d -> n 1 d") - rearrange(x, "n d -> 1 n d")
    return einsum(diff, diff, "n m d, n m d -> n m")


@jaxtyped(typechecker=beartype)
def mean_abs_asymmetry(
    x: Float[torch.Tensor, "B N N D"],
) -> Float[torch.Tensor, ""]:
    """Return mean abs diff between x[b,i,j] and x[b,j,i] for non-symmetry.

    Args:
        x: Tensor of shape (B, N, N, D) to test for transpose symmetry.

    Returns:
        Scalar mean absolute asymmetry; zero iff x is perfectly symmetric.
    """
    diff = x - rearrange(x, "b i j d -> b j i d")
    return reduce(diff.abs(), "b i j d -> ", "mean")


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> MainTrunk:
    """Provide a small MainTrunk instance in eval mode.

    Returns:
        A MainTrunk constructed with test-sized hyperparameters and set to eval.
    """
    return MainTrunk(
        model_params=ModelParams(
            window_size=MODEL_WINDOW_SIZE,
            f_ref_dim=F_REF_DIM,
            c_atom=C_ATOM,
            c_pair=C_PAIR,
            c_res=C_RES,
            c_atompair=C_ATOMPAIR,
            K_unit=K_UNIT,
            max_residues=RESIDUE_NUMBER,
            n_amino=N_AMINO,
            n_blocks_atom_transformer_encoder=N_BLOCKS_ATOM_TRANSFORMER_ENCODER,
            n_heads_atom_transformer_encoder=N_HEADS_ATOM_TRANSFORMER_ENCODER,
            n_blocks_atom_transformer_decoder=N_BLOCKS_ATOM_TRANSFORMER_DECODER,
            n_heads_atom_transformer_decoder=N_HEADS_ATOM_TRANSFORMER_DECODER,
            n_pairformer_blocks_template_embedder=N_PAIRFORMER_BLOCKS_TEMPLATE_EMBEDDER,
            n_paiformer_heads_template_embedder=N_HEADS_ATOM_TRANSFORMER_DECODER,
        ),
        res_distogram_params=ResidueDistogramParams(n_bins=N_BINS),
        atom_distogram_params=AtomDistogramParams(n_bins=N_ATOM_BINS),
        noise_params=NoiseScheduleParams(sigma_data=SIGMA_DATA),
    ).eval()


# ---------------------------------------------------------------------------
# Input tensor fixtures  (all have leading B dim)
# ---------------------------------------------------------------------------


@pytest.fixture
def ref_pos() -> Float[torch.Tensor, "B N_atom 3"]:
    """Provide random reference atom positions (B, N_ATOM, 3).

    Returns:
        Tensor of shape (B, N_ATOM, 3) drawn from standard normal distribution.
    """
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def ref_element() -> Float[torch.Tensor, "B N_atom E"]:
    """Provide random one-hot element features (B, N_ATOM, E).

    Returns:
        Float one-hot tensor of shape (B, N_ATOM, E) with randomly chosen
        element classes.
    """
    return F.one_hot(torch.randint(0, E, (B, N_ATOM)), num_classes=E).float()


@pytest.fixture
def ref_space_uid() -> Int[torch.Tensor, "B N_atom"]:
    """Provide all-zero chain/space UIDs (B, N_ATOM) for a single-chain input.

    Returns:
        Integer tensor of shape (B, N_ATOM) filled with zeros, representing
        a single chain.
    """
    return torch.zeros(B, N_ATOM, dtype=torch.long)


@pytest.fixture
def f_distogram() -> Float[torch.Tensor, "B N_res N_res N_bins"]:
    """One-hot template distogram with random bin assignments.

    Returns:
        Float one-hot tensor of shape (B, N_RES, N_RES, N_BINS) with each
        entry selecting a random distance bin.
    """
    return F.one_hot(
        torch.randint(0, N_BINS, (B, N_RES, N_RES)),
        num_classes=N_BINS,
    ).float()


@pytest.fixture
def f_pseudo_beta_mask() -> Float[torch.Tensor, "B N_res"]:
    """All-ones pseudo-β mask — every residue has a valid pseudo-β position.

    Returns:
        Float tensor of ones with shape (B, N_RES); no residue is masked out.
    """
    return torch.ones(B, N_RES)


@pytest.fixture
def f_residue_idx() -> Int[torch.Tensor, "B N_res"]:
    """Residue position indices— monotonically increasing per batch item.

    Returns:
        Long tensor of shape (B, N_RES) where each row is [0, 1, ..., N_RES-1].
    """
    return repeat(
        torch.arange(N_RES, dtype=torch.long),
        "n -> b n",
        b=B,
    ).contiguous()


@pytest.fixture
def r_input() -> Float[torch.Tensor, "B N_atom 3"]:
    """Noisy atom positions fed to the trunk as the diffusion denoising input.

    Returns:
        Random tensor of shape [B, N_ATOM, 3] sampled from a standard normal,
        simulating perturbed atom coordinates at an arbitrary diffusion step.
    """
    return torch.randn(B, N_ATOM, 3)


@pytest.fixture
def tok_idx() -> Int[torch.Tensor, "B N_atom"]:
    """Residue index for each atom, mapping atoms to their parent residue.

    Constructs a contiguous mapping where every ATOMS_PER_RES consecutive atoms
    share the same residue index in [0, N_RES), then tiles it across the batch.

    Returns:
        Integer tensor of shape [B, N_ATOM] where atom i belongs to residue
        i // ATOMS_PER_RES.
    """
    single = torch.repeat_interleave(torch.arange(N_RES), ATOMS_PER_RES)
    return repeat(single, "n -> b n", b=B).contiguous()


@pytest.fixture
def center_uid() -> Int[torch.Tensor, "B N_atom"]:
    """Center atom index for each atom, broadcast from the per-residue center.

    Picks one representative atom per residue (every ATOMS_PER_RES-th atom
    starting at 0) and broadcasts that index to all atoms in the same residue,
    then tiles across the batch.

    Returns:
        Integer tensor of shape [B, N_ATOM] where every atom in residue r holds
        the global index of residue r's center atom.
    """
    res_centers = torch.arange(
        0,
        N_ATOM,
        ATOMS_PER_RES,
    )  # [0, 3, 6, ..., 147] — one per residue
    single = repeat(
        res_centers,
        "n -> (n a)",
        a=ATOMS_PER_RES,
    )  # broadcast to every atom in residue
    return repeat(single, "n -> b n", b=B).contiguous()


@pytest.fixture
def gt_atom_distogram_sparse() -> (
    Float[torch.Tensor, "B N_atom K_sparse N_atom_bins"]
):
    """Random ground-truth sparse atom distogram targets."""
    return torch.randn(B, N_ATOM, K_SPARSE, N_ATOM_BINS)


@pytest.fixture
def gt_atom_distogram_mask_sparse() -> Bool[torch.Tensor, "B N_atom K_sparse"]:
    """All-True validity mask for sparse atom distogram — no padding."""
    return torch.ones(B, N_ATOM, K_SPARSE, dtype=torch.bool)


# ---------------------------------------------------------------------------
# FeaturizedBatch fixture and forward helper
# ---------------------------------------------------------------------------


@pytest.fixture
def featurized_batch(  # noqa: PLR0913
    ref_pos: Float[torch.Tensor, "B N_atom 3"],
    ref_element: Float[torch.Tensor, "B N_atom E"],
    ref_space_uid: Int[torch.Tensor, "B N_atom"],
    f_distogram: Float[torch.Tensor, "B N_res N_res N_bins"],
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"],
    f_residue_idx: Int[torch.Tensor, "B N_res"],
    r_input: Float[torch.Tensor, "B N_atom 3"],
    tok_idx: Int[torch.Tensor, "B N_atom"],
    center_uid: Int[torch.Tensor, "B N_res"],
    gt_atom_distogram_sparse: Float[
        torch.Tensor,
        "B N_atom K_sparse N_atom_bins",
    ],
    gt_atom_distogram_mask_sparse: Bool[torch.Tensor, "B N_atom K_sparse"],
) -> FeaturizedBatch:
    """Assemble FeaturizedBatch from input fixtures for trunk tests."""
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


def _forward(trunk: MainTrunk, batch: FeaturizedBatch) -> PredictedOutputs:
    """Run model forward pass and return the PredictedOutputs dataclass."""
    with torch.no_grad():
        return trunk(batch)


# ---------------------------------------------------------------------------
# sinusoidal_encoding
# ---------------------------------------------------------------------------


def test_sinusoidal_encoding_output_shape() -> None:
    """sinusoidal_encoding returns [B, N_res, dim] with all finite values."""
    positions = repeat(
        torch.arange(N_RES, dtype=torch.float32),
        "n -> b n",
        b=2,
    )
    out = sinusoidal_encoding(positions, dim=C_RES)
    assert out.shape == (2, N_RES, C_RES)
    assert torch.isfinite(out).all()


def test_sinusoidal_encoding_varies_across_positions() -> None:
    """Every residue position maps to distinct encoding."""
    positions = repeat(torch.arange(N_RES, dtype=torch.float32), "n -> 1 n")
    enc = rearrange(sinusoidal_encoding(positions, dim=C_RES), "1 n d -> n d")
    off_diag = sq_dist_matrix(enc) + torch.eye(N_RES) * 1e10
    assert off_diag.min().item() > 0


# ---------------------------------------------------------------------------
# TimeFourierEmbedding
# ---------------------------------------------------------------------------


def test_time_fourier_embedding_output_shape(model: MainTrunk) -> None:
    """TimeFourierEmbedding maps length-N noise level to finite [N, C_res]."""
    out = model.time_fourier(torch.randn(N_RES))
    assert out.shape == (N_RES, C_RES)
    assert torch.isfinite(out).all()


def test_time_fourier_embedding_fixed_buffers() -> None:
    """Freqs and phases are non-trainable buffers, outputs bounded [-1, 1]."""
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


def test_rel_pos_enc_output_shape(model: MainTrunk) -> None:
    """RelativePositionEncoding returns finite output of correct shape."""
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


def test_rel_pos_enc_deterministic(model: MainTrunk) -> None:
    """RelativePositionEncoding returns same inputs return same output."""
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
    """Algorithm 3 returns [1, N, N, C_pair] for batch-size-1 inputs."""
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
    """Cross-chain pairs always use overflow bin.

    Regardless of residue_index values, cross-chain pairs are assigned to the
    overflow bin.
    """
    enc = RelativePositionEncoding(C_PAIR, r_max=4, s_max=2)
    N = 4
    asym_id = rearrange(
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "n -> 1 n",
    )
    entity_id = torch.zeros(1, N, dtype=torch.long)
    token_index = torch.zeros(1, N, dtype=torch.long)
    sym_id = rearrange(torch.tensor([0, 0, 1, 1], dtype=torch.long), "n -> 1 n")

    idx_a = rearrange(torch.tensor([0, 1, 0, 1], dtype=torch.long), "n -> 1 n")
    idx_b = rearrange(
        torch.tensor([0, 1, 50, 51], dtype=torch.long),
        "n -> 1 n",
    )

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
    """Same-chain pairs encode clipped residue index diff, not overflow bin."""
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
    """b_same_entity flag differs when entity_ids differ.

    b_same_entity is 1 iff entity_id[i] == entity_id[j]; differing entity_ids
    produce a different output encoding.
    """
    enc = RelativePositionEncoding(C_PAIR, r_max=4, s_max=2)
    N = 4
    residue_index = rearrange(torch.arange(N, dtype=torch.long), "n -> 1 n")
    asym_id = torch.zeros(1, N, dtype=torch.long)
    token_index = torch.zeros(1, N, dtype=torch.long)
    sym_id = torch.zeros(1, N, dtype=torch.long)

    entity_same = torch.zeros(1, N, dtype=torch.long)
    entity_split = rearrange(
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "n -> 1 n",
    )

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
    """Cross-chain d_chain varies with sym_id.

    d_chain = clip(sym_i - sym_j + s_max, 0, 2*s_max); different sym_ids
    produce a different output encoding.
    """
    enc = RelativePositionEncoding(C_PAIR, r_max=4, s_max=2)
    N = 4
    residue_index = rearrange(torch.arange(N, dtype=torch.long), "n -> 1 n")
    asym_id = rearrange(
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "n -> 1 n",
    )
    entity_id = torch.zeros(1, N, dtype=torch.long)
    token_index = torch.zeros(1, N, dtype=torch.long)

    sym_id_a = rearrange(
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "n -> 1 n",
    )
    sym_id_b = rearrange(
        torch.tensor([0, 0, 2, 2], dtype=torch.long),
        "n -> 1 n",
    )

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


def test_residue_distogram_head_output() -> None:
    """ResidueDistogramHead output is finite, correctly shaped, and symmetric.

    Maps [B, N_res, N_res, C_pair] to [B, N_res, N_res, N_bins] and
    symmetrises the pair embedding before projection so logits[i,j] ==
    logits[j,i].
    """
    head = ResidueDistogramHead(C_PAIR, n_bins=N_BINS)
    logits = head(torch.randn(B, N_RES, N_RES, C_PAIR))
    assert logits.shape == (B, N_RES, N_RES, N_BINS)
    assert torch.isfinite(logits).all()
    assert mean_abs_asymmetry(logits).item() < TOLERANCE


# ---------------------------------------------------------------------------
# AtomDistogramHead
# ---------------------------------------------------------------------------


def test_atom_distogram_head_output_shapes() -> None:
    """AtomDistogramHead returns finite logits and local-window mask."""
    head = AtomDistogramHead(
        C_ATOMPAIR,
        n_bins=N_BINS,
        atoms_per_res=ATOMS_PER_RES,
    )
    logits, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert logits.shape == (N_ATOM, N_ATOM, N_BINS)
    assert mask.shape == (N_ATOM, N_ATOM)
    assert mask.dtype == torch.bool
    assert torch.isfinite(logits).all()
    assert (
        logits[mask].std(dim=-1).min().item() > 0
    )  # logits vary across bin dimension for local-window pairs


def test_atom_distogram_head_mask_includes_diagonal() -> None:
    """Window mask is True on diagonal, every atom is within own window."""
    head = AtomDistogramHead(
        C_ATOMPAIR,
        n_bins=N_BINS,
        atoms_per_res=ATOMS_PER_RES,
    )
    _, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert mask.diagonal().all()


def test_atom_distogram_head_mask_symmetric() -> None:
    """If atom i within window of atom j, atom j within window of atom i."""
    head = AtomDistogramHead(
        C_ATOMPAIR,
        n_bins=N_BINS,
        atoms_per_res=ATOMS_PER_RES,
    )
    _, mask = head(torch.randn(N_ATOM, N_ATOM, C_ATOMPAIR))
    assert torch.equal(mask, rearrange(mask, "i j -> j i"))


# ---------------------------------------------------------------------------
# MainTrunk.forward — output shapes and values
# ---------------------------------------------------------------------------


def test_main_trunk_r_denoised_shape_finite(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """Main trunk returns finite denoised atom coordinates of correct shape."""
    out = _forward(model, featurized_batch)
    assert out.r_denoised.shape == (B, N_ATOM, 3)
    assert torch.isfinite(out.r_denoised).all()


def test_main_trunk_residue_distogram_shape_finite(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """Residue distogram head returns finite logits of correct shape."""
    out = _forward(model, featurized_batch)
    assert out.residue_distogram_logits.shape == (B, N_RES, N_RES, N_BINS)
    assert torch.isfinite(out.residue_distogram_logits).all()


def test_main_trunk_atom_distogram_shape_finite(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """Atom distogram output shape is [B, N_ATOM, ..., N_ATOM_BINS]."""
    out = _forward(model, featurized_batch)
    assert out.atom_distogram_logits.ndim == DISTOGRAM_RANK
    assert out.atom_distogram_logits.shape[0] == B
    assert out.atom_distogram_logits.shape[1] == N_ATOM
    assert out.atom_distogram_logits.shape[3] == N_ATOM_BINS
    assert torch.isfinite(out.atom_distogram_logits).all()


def test_main_trunk_atom_distogram_bins_match_ground_truth(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """Number of pred atom distance bins matches bin count in ground-truth."""
    out = _forward(model, featurized_batch)
    assert (
        out.atom_distogram_logits.shape[-1]
        == featurized_batch.gt_atom_distogram_sparse.shape[-1]
    )


def test_main_trunk_distogram_loss_atom_computable(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """Atom distogram logits from trunk correct shape for loss."""
    out = _forward(model, featurized_batch)
    loss = distogram_loss_atom(
        out.atom_distogram_logits,
        featurized_batch.gt_atom_distogram_sparse,
        featurized_batch.gt_atom_distogram_mask_sparse,
    ).mean()
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_main_trunk_sequence_head_shapes_and_vocab_bound(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """Sequence head output has correct shape, finite, never predicts "X"."""
    out = _forward(model, featurized_batch)

    assert out.seq_logits.shape == (B, N_RES, N_AMINO)
    assert torch.isfinite(out.seq_logits).all()
    assert (
        out.seq_logits.argmax(dim=-1).max() < N_AMINO
    )  # X is never the argmax

    assert len(out.intermediate_denoised_coord_stack) == K_UNIT
    assert len(out.intermediate_pred_aa_logit_stack) == K_UNIT
    for inter_logits in out.intermediate_pred_aa_logit_stack:
        assert inter_logits.shape == (B, N_RES, N_AMINO)
        assert (
            inter_logits.argmax(dim=-1).max() < N_AMINO
        )  # X is never the argmax


def test_main_trunk_intermediate_coords_shape_finite(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """Every intermediate denoised coord tensor has shape [B, N_atom, 3]."""
    out = _forward(model, featurized_batch)
    for r in out.intermediate_denoised_coord_stack:
        assert r.shape == (B, N_ATOM, 3)
        assert torch.isfinite(r).all()


def test_main_trunk_forward_with_mask_token_aa_indices(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """Mask token 'X' must not raise IndexError during forward pass."""
    # aa_indices containing 20 (mask token "X") must not raise IndexError
    masked_batch = dataclasses.replace(
        featurized_batch,
        aa_indices=torch.full((B, N_RES), N_AMINO, dtype=torch.long),
    )
    with torch.no_grad():
        out = model(masked_batch)
    assert out.r_denoised.shape == (B, N_ATOM, 3)
    assert torch.isfinite(out.r_denoised).all()


# you need to add integration tests here. don't be afraid to use pytest mocks.


def _assert_submodule_grads(trunk: MainTrunk) -> None:
    """Assert top-level submodules have at least one finite nonzero gradient.

    Args:
        trunk: Trunk module after a backward pass has been called.
    """
    buckets: dict[str, list[Float[torch.Tensor, "..."]]] = {}
    for name, param in trunk.named_parameters():
        if param.grad is not None:
            buckets.setdefault(name.split(".")[0], []).append(param.grad)
    assert buckets, "no parameters have gradients — backward was not called"
    for prefix, grads in buckets.items():
        assert any(
            torch.isfinite(g).all().item() and g.abs().max().item() > 0
            for g in grads
        ), f"submodule '{prefix}' has no finite nonzero gradients"


def test_integration_gradient_flow_composite_loss(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """Composite loss propagates finite nonzero grads to every submodule."""
    _ = model.train()
    pred: PredictedOutputs = model(featurized_batch)

    kabsch_loss = atom_loss(
        pred.r_denoised,
        featurized_batch.r_gt,
        featurized_batch.atom5_mask,
    ).mean()
    ce_loss = seq_ce_loss(pred.seq_logits, featurized_batch.aa_indices)
    lddt = smooth_lddt_loss(
        pred.r_denoised,
        featurized_batch.r_gt,
        featurized_batch.atom5_mask,
        cutoff=15.0,
    )
    gt_res_bin_idx = featurized_batch.gt_res_distogram.argmax(dim=-1).clamp(
        0,
        pred.residue_distogram_logits.size(-1) - 1,
    )
    res_distogram_loss = distogram_loss_residue(
        pred.residue_distogram_logits,
        gt_res_bin_idx,
        featurized_batch.f_pseudo_beta_mask.bool(),
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
        ),
    ):
        gamma: float = 0.99 ** (K_unit - k_idx - 1)
        k_loss = atom_loss(
            inter_coords,
            featurized_batch.r_gt,
            featurized_batch.atom5_mask,
        ) + seq_ce_loss(inter_logits, featurized_batch.aa_indices)
        intermediate_loss = intermediate_loss + gamma * k_loss
    intermediate_loss = (intermediate_loss / max(K_unit, 1)).mean()

    total_loss = (
        kabsch_loss
        + ce_loss
        + lddt
        + res_distogram_loss
        + atom_distogram_loss
        + intermediate_loss
    )
    torch.autograd.backward([total_loss])

    _assert_submodule_grads(model)

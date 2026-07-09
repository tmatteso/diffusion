"""Tests for the main trunk forward pass.

Covers sinusoidal encoding, TimeFourierEmbedding, RelativePositionEncoding,
ResidueDistogramHead, AtomDistogramHead, and end-to-end MainTrunk forward
pass including output shapes, symmetry properties, loss computability, and
gradient flow through a composite training loss.
"""

import contextlib
import dataclasses
from collections.abc import Generator
from typing import cast

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from architecture.losses import (
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    seq_ce_loss,
    smooth_lddt_loss,
)
from architecture.main_trunk import (
    EMA,
    MainTrunk,
    PredictedOutputs,
    RelativePositionEncoding,
    TimeFourierEmbedding,
)
from beartype import beartype
from einops import einsum, rearrange, reduce, repeat
from helpers.atom_utils import RESTYPE_NUM_NO_X
from helpers.data import FeaturizedBatch, sinusoidal_encoding
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Int, jaxtyped
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from train.train_config import (
    AtomDistogramParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
    TemplateDistogramParams,
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
        template_distogram_params=TemplateDistogramParams(n_bins=N_BINS),
        residue_distogram_params=ResidueDistogramParams(n_bins=N_BINS),
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
def gt_atom_distogram_sparse() -> Int[torch.Tensor, "B N_atom K_sparse"]:
    """Random ground-truth sparse atom distogram bin indices."""
    return torch.randint(0, N_ATOM_BINS, (B, N_ATOM, K_SPARSE))


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
        gt_res_distogram_indices=torch.zeros(B, N_RES, N_RES, dtype=torch.long),
        noised_res_distogram=f_distogram,
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
    """K-neighbour dim matches and gt indices are valid bin indices."""
    out = _forward(model, featurized_batch)
    n_bins = out.atom_distogram_logits.shape[-1]
    assert (
        out.atom_distogram_logits.shape[2]
        == featurized_batch.gt_atom_distogram_sparse.shape[2]
    )
    assert featurized_batch.gt_atom_distogram_sparse.max() < n_bins
    assert featurized_batch.gt_atom_distogram_sparse.min() >= 0


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
        featurized_batch.atom5_mask.float(),
        aa_indices=featurized_batch.aa_indices,
        lambda_sigma_weight=torch.ones(B),
    ).mean()
    ce_loss = seq_ce_loss(pred.seq_logits, featurized_batch.aa_indices)
    lddt = smooth_lddt_loss(
        pred.r_denoised,
        featurized_batch.r_gt,
        featurized_batch.atom5_mask,
        cutoff=15.0,
    )
    gt_res_bin_idx = featurized_batch.gt_res_distogram_indices.clamp(
        0,
        pred.residue_distogram_logits.size(-1) - 1,
    )
    res_distogram_loss = distogram_loss_residue(
        pred.residue_distogram_logits,
        gt_res_bin_idx,
        featurized_batch.f_pseudo_beta_mask.float(),
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
            featurized_batch.atom5_mask.float(),
            aa_indices=featurized_batch.aa_indices,
            lambda_sigma_weight=torch.ones(B),
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


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

_EMA_TOLERANCE = 1e-6
_SYNTHETIC_STEP_COUNT = 7


def test_ema_init_shadow_matches_model_and_is_independent_copy(
    model: MainTrunk,
) -> None:
    """Shadow starts equal to model weights but is a detached, independent copy.

    Not parametrized: this test exercises a single invariant (the shadow is
    a snapshot, not a view) with no meaningful axis of variation; the decay
    arithmetic itself is covered by the parametrized formula test below.
    """
    ema = EMA(model, decay=0.999)
    state_dict = cast(dict[str, torch.Tensor], model.state_dict())
    for name, tensor in state_dict.items():
        assert torch.equal(ema.shadow[name], tensor)

    name, param = next(model.named_parameters())
    original = param.detach().clone()
    with torch.no_grad():
        _ = param.add_(1.0)
    assert torch.equal(ema.shadow[name], original)


@pytest.mark.parametrize(
    ("decay", "step", "expected_effective_decay"),
    [
        pytest.param(0.999, 0, 1.0 / 10.0, id="step0_ramp_dominates"),
        pytest.param(0.999, 89, 90.0 / 99.0, id="mid_ramp_below_target"),
        pytest.param(0.9, 10_000, 0.9, id="late_step_target_decay_dominates"),
    ],
)
def test_ema_update_matches_bias_corrected_decay_formula(
    model: MainTrunk,
    decay: float,
    step: int,
    expected_effective_decay: float,
) -> None:
    """update() blends shadow and param by min(decay, (step+1)/(step+10)).

    Covers the ramp's dominant regime at step 0, a mid-ramp step where the
    bias-correction term is still below the target decay, and a late step
    where the target decay has taken over.
    """
    ema = EMA(model, decay=decay)
    name, param = next(model.named_parameters())
    before = ema.shadow[name].clone()

    with torch.no_grad():
        _ = param.add_(1.0)
    after_param = cast(
        dict[str, torch.Tensor],
        model.state_dict(),
    )[name].clone()
    ema.update(model, step=step)

    expected = (
        expected_effective_decay * before
        + (1.0 - expected_effective_decay) * after_param
    )
    assert torch.allclose(ema.shadow[name], expected, atol=_EMA_TOLERANCE)


def test_ema_non_floating_point_buffer_is_copied_not_averaged(
    model: MainTrunk,
) -> None:
    """Non-float buffers are copied verbatim instead of decay-blended.

    Not parametrized: MainTrunk has no integer buffers today (freqs, phases,
    and centers are all float and fixed at init), so this test registers a
    synthetic int64 buffer to exercise the copy-not-blend branch directly.
    """
    model.register_buffer("step_count", torch.tensor(0, dtype=torch.int64))
    ema = EMA(model, decay=0.999)

    _ = model.get_buffer("step_count").fill_(_SYNTHETIC_STEP_COUNT)
    ema.update(model, step=0)

    assert ema.shadow["step_count"].item() == _SYNTHETIC_STEP_COUNT


@contextlib.contextmanager
def _single_process_gloo_group() -> Generator[None]:
    """Init and tear down a single-process gloo group for DDP-wrapping tests."""
    # HashStore is a private torch.distributed C-extension symbol that
    # isn't officially re-exported; getattr() sidesteps the static
    # private-import check, whose result is otherwise inconsistent
    # across torch's per-platform packaging.
    dist.init_process_group(
        backend="gloo",
        store=getattr(  # noqa: B009  # pyright: ignore[reportAny]
            dist,
            "HashStore",
        )(),
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_ema_unwraps_ddp_module_prefix_on_init_and_update(
    model: MainTrunk,
) -> None:
    """EMA strips the DDP "module." prefix so shadow keys match a plain model.

    Not parametrized: this is a single structural invariant (DDP-wrapped and
    plain models must produce identically-keyed shadows, matching the
    prefix-stripped keys ``save_checkpoint`` writes for the raw model) with
    no meaningful axis of variation.
    """
    plain_keys = set(model.state_dict().keys())
    with _single_process_gloo_group():
        ddp_model = DDP(model)
        ema = EMA(ddp_model, decay=0.999)
        assert set(ema.shadow.keys()) == plain_keys
        assert not any(k.startswith("module.") for k in ema.shadow)

        with torch.no_grad():
            for p in ddp_model.parameters():
                _ = p.add_(1.0)
        ema.update(ddp_model, step=0)
        assert set(ema.shadow.keys()) == plain_keys


def test_ema_state_dict_and_load_state_dict_round_trip(
    model: MainTrunk,
) -> None:
    """load_state_dict restores a previously saved shadow exactly.

    Not parametrized: this exercises a single serialization round trip with
    no meaningful axis of variation beyond what the decay-formula and
    buffer-handling tests above already cover.
    """
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        for p in model.parameters():
            _ = p.add_(1.0)
    ema.update(model, step=500)
    saved = {name: tensor.clone() for name, tensor in ema.state_dict().items()}

    fresh_ema = EMA(model, decay=0.999)
    fresh_ema.load_state_dict(saved)

    for name, tensor in saved.items():
        assert torch.equal(fresh_ema.shadow[name], tensor)


# ---------------------------------------------------------------------------
# EMA.swap (zero-copy pointer swap, used by swapped_in_ema_weights)
# ---------------------------------------------------------------------------


def test_ema_swap_exchanges_parameter_and_buffer_storage_with_shadow(
    model: MainTrunk,
) -> None:
    """swap() moves the shadow's values onto the model and vice versa.

    Arrange: build an EMA (shadow == initial weights), then perturb both a
    parameter and a buffer (``time_fourier.freqs``) on the live model so
    the model and the shadow diverge from each other and from their
    pre-perturbation values.
    Act: call ``ema.swap(model)`` once.
    Assert: the model's parameter and buffer now hold what the shadow held
    immediately before the swap; the shadow now holds what the model held
    (the perturbed values) immediately before the swap.
    """
    ema = EMA(model, decay=0.999)
    param_name, param = next(model.named_parameters())
    buffer_name = "time_fourier.freqs"
    shadow_param_before = ema.shadow[param_name].clone()
    shadow_buffer_before = ema.shadow[buffer_name].clone()

    with torch.no_grad():
        _ = param.add_(1.0)
        _ = model.time_fourier.freqs.add_(1.0)
    perturbed_param = param.clone()
    perturbed_buffer = model.time_fourier.freqs.clone()

    ema.swap(model)

    assert torch.equal(
        dict(model.named_parameters())[param_name],
        shadow_param_before,
    )
    assert torch.equal(model.time_fourier.freqs, shadow_buffer_before)
    assert torch.equal(ema.shadow[param_name], perturbed_param)
    assert torch.equal(ema.shadow[buffer_name], perturbed_buffer)


def test_ema_swap_twice_restores_original_state(model: MainTrunk) -> None:
    """Calling swap() twice is a no-op: it's its own inverse.

    Arrange: build an EMA, perturb the model's parameters and a buffer so
    the model and shadow diverge, then snapshot both the model's and the
    shadow's values at that divergent point.
    Act: call ``ema.swap(model)`` twice in a row.
    Assert: the model and the shadow both match their respective
    snapshots exactly, confirming swap()/swap() round-trips losslessly.
    """
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        for p in model.parameters():
            _ = p.add_(1.0)
        _ = model.time_fourier.freqs.add_(1.0)

    model_snapshot = {
        name: tensor.clone()
        for name, tensor in cast(
            "dict[str, torch.Tensor]",
            model.state_dict(),
        ).items()
    }
    shadow_snapshot = {
        name: tensor.clone() for name, tensor in ema.shadow.items()
    }

    ema.swap(model)
    ema.swap(model)

    current_model_sd = cast("dict[str, torch.Tensor]", model.state_dict())
    for name, tensor in model_snapshot.items():
        assert torch.equal(current_model_sd[name], tensor)
    for name, tensor in shadow_snapshot.items():
        assert torch.equal(ema.shadow[name], tensor)


def test_ema_swap_preserves_parameter_object_identity(model: MainTrunk) -> None:
    """swap() reassigns `.data` in place rather than replacing parameters.

    Arrange: build an EMA and record the Python object identity (``id()``)
    of every parameter the model currently holds.
    Act: call ``ema.swap(model)``.
    Assert: every parameter name still resolves to the exact same Python
    object as before (``is``, not just equal-valued) -- this is the
    property that keeps Adam's per-parameter state (keyed by object
    identity) valid across the swap.
    """
    ema = EMA(model, decay=0.999)
    params_before = dict(model.named_parameters())
    ids_before = {name: id(param) for name, param in params_before.items()}

    ema.swap(model)

    params_after = dict(model.named_parameters())
    for name, param in params_after.items():
        assert id(param) == ids_before[name], f"Parameter '{name}' was replaced"


def test_ema_swap_reuses_shadow_storage_without_copying(
    model: MainTrunk,
) -> None:
    """swap() exchanges storage pointers rather than copying values.

    Arrange: build an EMA and record the shadow tensor's storage address
    (``data_ptr()``) for one parameter before any swap happens.
    Act: call ``ema.swap(model)``.
    Assert: the model's parameter now shares that exact storage address --
    proof the swap reused the shadow's existing memory rather than
    allocating a new tensor and copying values into it.
    """
    ema = EMA(model, decay=0.999)
    name, _ = next(model.named_parameters())
    shadow_ptr_before = ema.shadow[name].data_ptr()

    ema.swap(model)

    assert dict(model.named_parameters())[name].data_ptr() == shadow_ptr_before


def test_ema_swap_preserves_optimizer_state_for_swapped_parameters(
    model: MainTrunk,
) -> None:
    """swap() doesn't break Adam's per-parameter state (keyed by identity).

    Arrange: attach an Adam optimizer to the model's parameters and run one
    manual step so ``optimizer.state`` is populated per parameter object.
    Act: build an EMA and call ``ema.swap(model)``.
    Assert: every parameter object the optimizer already holds a state
    entry for is still a valid key in ``optimizer.state`` after the swap --
    confirming the swap only reassigns ``.data`` rather than replacing the
    Parameter objects the optimizer references.
    """
    optimizer = Adam(model.parameters(), lr=1e-3)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    _ = optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    optimizer.zero_grad()
    params_before = list(model.parameters())

    ema = EMA(model, decay=0.999)
    ema.swap(model)

    for p in params_before:
        assert p in optimizer.state


def test_ema_swap_unwraps_ddp(model: MainTrunk) -> None:
    """swap() operates on the unwrapped .module when the model is DDP-wrapped.

    Arrange: wrap the model in a single-process gloo DDP group, build an
    EMA from the DDP-wrapped model, then perturb the underlying model's
    parameters so the shadow and the live weights diverge.
    Act: call ``ema.swap(ddp_model)``, passing the DDP wrapper directly.
    Assert: shadow keys carry no "module." prefix, and the model's
    parameters now hold the pre-swap shadow values -- confirming the swap
    reached through the DDP wrapper to the real MainTrunk rather than
    operating on wrapper-prefixed keys that would never match.
    """
    with _single_process_gloo_group():
        ddp_model = DDP(model)
        ema = EMA(ddp_model, decay=0.999)
        assert not any(k.startswith("module.") for k in ema.shadow)

        name, param = next(model.named_parameters())
        shadow_before = ema.shadow[name].clone()
        with torch.no_grad():
            _ = param.add_(1.0)

        ema.swap(ddp_model)

        assert torch.equal(dict(model.named_parameters())[name], shadow_before)

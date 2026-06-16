"""MainTrunk — pure PyTorch implementation.

Based on Algorithm 2 from the AlphaFold 3 paper.

Imports from previously implemented modules:
  - atom_attention_decoder.py  (LinearNoBias, AtomTransformer)
  - atom_feature_encoder.py    (AtomFeatureEncoder)
  - template_embedder.py       (TemplateEmbedder)

Inputs
------
batch           : FeaturizedBatch  (all tensors have leading B dim)

Outputs
-------
r_denoised      : (B, N_atom, 3)        — denoised atom positions
f_seq_logits    : (B, N_token, 20)      — amino-acid sequence logits
"""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.atom_transformers import (
    AtomAttentionDecoder,
    AtomFeatureEncoder,
    scatter_mean,
)
from architecture.layers import (
    LayerNorm,
    LinearNoBias,
    TypedEmbedding,
    TypedModuleList,
    TypedSequential,
)
from architecture.node_update import NodeUpdate
from architecture.pair_update import PairUpdate
from architecture.template_embedder import TemplateEmbedder
from beartype import beartype
from einops import rearrange, repeat
from helpers.batch_types import FeaturizedBatch
from jaxtyping import Bool, Float, Int, jaxtyped
from train.train_config import (
    AtomDistogramParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
)
from typing_extensions import override

# ---------------------------------------------------------------------------
# EmbeddedInputs — output of MainTrunk.embed_inputs (steps 1-8)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EmbeddedInputs:
    """Embeddings produced by MainTrunk.embed_inputs before decoder loop.

    Holds s_i, t_i, z_ij, atom skip embeddings, and metadata needed by the
    decoder loop. All tensor fields have a leading batch dimension B.
    """

    s_i: Float[torch.Tensor, "B N_res c_res"]
    t_i: Float[torch.Tensor, "B N_res c_res"]
    z_ij: Float[torch.Tensor, "B N_res N_res c_pair"]
    q_skip: Float[torch.Tensor, "B N_atom c_atom"]
    c_skip: Float[torch.Tensor, "B N_atom c_atom"]
    p_skip: Float[torch.Tensor, "B N_atom K c_atompair"]
    c_l: Float[torch.Tensor, "B N_atom c_atom"]
    r_input: Float[torch.Tensor, "B N_atom 3"]
    tok_idx: Int[torch.Tensor, "B N_atom"]
    center_uid: Int[torch.Tensor, "B N_atom"]
    t_hat: Float[torch.Tensor, "B"]
    B: int
    N_atom: int
    N_res: int
    device: torch.device
    denom: Float[torch.Tensor, "B"]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class PredictedOutputs:
    """Batched outputs from a single forward pass of MainTrunk.

    Includes denoised coordinates, sequence logits, residue and atom distogram
    logits, and intermediate decoder-block outputs used for auxiliary losses.
    """

    r_denoised: Float[torch.Tensor, "B N_atom 3"]
    seq_logits: Float[torch.Tensor, "B N_res n_amino"]
    residue_distogram_logits: Float[torch.Tensor, "B N_res N_res n_bins"]
    atom_distogram_logits: Float[torch.Tensor, "B N_atom K n_atom_bins"]
    intermediate_denoised_coord_stack: list[Float[torch.Tensor, "B N_atom 3"]]
    intermediate_pred_aa_logit_stack: list[
        Float[torch.Tensor, "B N_res n_amino"]
    ]


# ---------------------------------------------------------------------------
# TimeFourierEmbedding  (used in step 3)
# ---------------------------------------------------------------------------


class TimeFourierEmbedding(nn.Module):
    """Maps scalar t̂ to a Fourier feature vector ∈ R^{c_res} (Algorithm 22).

    Weights and biases are fixed random values sampled once at init, not
    learned.
    """

    def __init__(self, c_res: int) -> None:
        super().__init__()
        self.freqs: Float[torch.Tensor, "c_res"]
        self.phases: Float[torch.Tensor, "c_res"]
        self.register_buffer("freqs", torch.randn(c_res))
        self.register_buffer("phases", torch.randn(c_res))

    @override
    def __call__(
        self,
        x: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "... c_res"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(x)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "... c_res"]:
        """Map noise-level scalar to a Fourier feature vector.

        Args:
            x: Scalar or batched input (any leading shape).

        Returns:
            Cosine Fourier embedding of shape (*x.shape, c_res).
        """
        angles = (
            2
            * math.pi
            * (rearrange(x, "... -> ... 1") * self.freqs + self.phases)
        )
        return torch.cos(angles)


# ---------------------------------------------------------------------------
# RelativePositionEncoding  (step 5)
# Encodes relative token positions into pair embeddings z_ij^init.
# ---------------------------------------------------------------------------


class RelativePositionEncoding(nn.Module):
    """Algorithm 3 relative position encoding.

    Produces p_ij ∈ R^{c_z} from per-token identity features, encoding relative
    residue position, relative token position within a residue, entity
    identity,
    and relative chain index.
    """

    def __init__(self, c_pair: int, r_max: int = 32, s_max: int = 2) -> None:
        super().__init__()
        self.r_max: int = r_max
        self.s_max: int = s_max
        # Input dim: a_rel_pos (2r+2) + a_rel_token (2r+2)
        # + b_same_entity (1) + a_rel_chain (2s+2)
        in_dim = (2 * r_max + 2) + (2 * r_max + 2) + 1 + (2 * s_max + 2)
        self.proj: LinearNoBias = LinearNoBias(in_dim, c_pair)

    @override
    def __call__(
        self,
        *,
        residue_index: Int[torch.Tensor, "B N_res"],
        asym_id: Int[torch.Tensor, "B N_res"],
        entity_id: Int[torch.Tensor, "B N_res"],
        token_index: Int[torch.Tensor, "B N_res"],
        sym_id: Int[torch.Tensor, "B N_res"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(
            residue_index=residue_index,
            asym_id=asym_id,
            entity_id=entity_id,
            token_index=token_index,
            sym_id=sym_id,
        )

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        *,
        residue_index: Int[torch.Tensor, "B N_res"],
        asym_id: Int[torch.Tensor, "B N_res"],
        entity_id: Int[torch.Tensor, "B N_res"],
        token_index: Int[torch.Tensor, "B N_res"],
        sym_id: Int[torch.Tensor, "B N_res"],
    ) -> Float[torch.Tensor, "B N_res N_res c_pair"]:
        """Compute Algorithm 3 relative position embeddings from features.

        Args:
            residue_index: Integer residue index per token, shape (B, N_res).
            asym_id: Asymmetric chain ID per token, shape (B, N_res).
            entity_id: Entity ID per token, shape (B, N_res).
            token_index: Within-residue token index per token, shape (B,
                N_res).
            sym_id: Symmetric chain ID per token, shape (B, N_res).

        Returns:
            Pair embedding of shape (B, N_res, N_res, c_pair).
        """
        r_max = self.r_max
        s_max = self.s_max

        # Lines 1-3: pairwise boolean masks
        b_same_chain: Bool[torch.Tensor, "B N_res N_res"] = rearrange(
            asym_id,
            "b n -> b n 1",
        ) == rearrange(asym_id, "b m -> b 1 m")
        b_same_residue: Bool[torch.Tensor, "B N_res N_res"] = rearrange(
            residue_index,
            "b n -> b n 1",
        ) == rearrange(residue_index, "b m -> b 1 m")
        b_same_entity: Bool[torch.Tensor, "B N_res N_res"] = rearrange(
            entity_id,
            "b n -> b n 1",
        ) == rearrange(entity_id, "b m -> b 1 m")

        # Line 4: d_residue — clipped within-chain, overflow bin across chains
        d_res_raw: Int[torch.Tensor, "B N_res N_res"] = rearrange(
            residue_index,
            "b n -> b n 1",
        ) - rearrange(residue_index, "b m -> b 1 m")
        d_residue: Int[torch.Tensor, "B N_res N_res"] = torch.where(
            b_same_chain,
            (d_res_raw + r_max).clamp(0, 2 * r_max),
            torch.full_like(d_res_raw, 2 * r_max + 1),
        )

        # Line 5: a_rel_pos
        a_rel_pos: Float[torch.Tensor, "B N_res N_res rel_pos_bins"] = (
            F.one_hot(d_residue, num_classes=2 * r_max + 2).float()
        )

        # Line 6: d_token — clipped when same chain+residue, overflow otherwise
        d_tok_raw: Int[torch.Tensor, "B N_res N_res"] = rearrange(
            token_index,
            "b n -> b n 1",
        ) - rearrange(token_index, "b m -> b 1 m")
        d_token: Int[torch.Tensor, "B N_res N_res"] = torch.where(
            b_same_chain & b_same_residue,
            (d_tok_raw + r_max).clamp(0, 2 * r_max),
            torch.full_like(d_tok_raw, 2 * r_max + 1),
        )

        # Line 7: a_rel_token
        a_rel_token: Float[torch.Tensor, "B N_res N_res rel_tok_bins"] = (
            F.one_hot(d_token, num_classes=2 * r_max + 2).float()
        )

        # Line 8: d_chain clipped when NOT same chain, overflow when same chain
        d_chain_raw: Int[torch.Tensor, "B N_res N_res"] = rearrange(
            sym_id,
            "b n -> b n 1",
        ) - rearrange(sym_id, "b m -> b 1 m")
        d_chain: Int[torch.Tensor, "B N_res N_res"] = torch.where(
            ~b_same_chain,
            (d_chain_raw + s_max).clamp(0, 2 * s_max),
            torch.full_like(d_chain_raw, 2 * s_max + 1),
        )

        # Line 9: a_rel_chain
        a_rel_chain: Float[torch.Tensor, "B N_res N_res rel_chain_bins"] = (
            F.one_hot(d_chain, num_classes=2 * s_max + 2).float()
        )

        # Line 10: concat and project
        b_entity_float: Float[torch.Tensor, "B N_res N_res 1"] = rearrange(
            b_same_entity.float(),
            "b n m -> b n m 1",
        )
        features: Float[torch.Tensor, "B N_res N_res feat"] = torch.cat(
            [a_rel_pos, a_rel_token, b_entity_float, a_rel_chain],
            dim=-1,
        )
        return self.proj(features)  # feat -> c_pair


# ---------------------------------------------------------------------------
# 1. ResidueDistogramHead
# ---------------------------------------------------------------------------
# Input is z_ij  (B, N_token, N_token, c_pair)  — trunk pair representation
# Output the logits (B, N_token, N_token, 64)
#
# Symmetrisation: average z_ij and z_ji before projection, so the predicted
# distance matrix is symmetric by construction.
# ---------------------------------------------------------------------------


class ResidueDistogramHead(nn.Module):
    """Projects symmetrised pair embeddings z_ij → 64 distance-bin logits.

    Parameters
    ----------
    c_pair  : input pair embedding dim
    n_bins  : number of distance bins (default 64)
    d_min   : minimum distance in Å   (default 2.0)
    d_max   : maximum distance in Å   (default 22.0)
    """

    def __init__(
        self,
        c_pair: int,
        n_bins: int = 64,
        d_min: float = 2.0,
        d_max: float = 22.0,
    ) -> None:
        super().__init__()
        self.n_bins: int = n_bins
        self.d_min: float = d_min
        self.d_max: float = d_max

        self.layer_norm: LayerNorm = LayerNorm(c_pair)
        self.proj1: LinearNoBias = LinearNoBias(c_pair, c_pair)
        self.proj2: LinearNoBias = LinearNoBias(c_pair, n_bins)

    @override
    def __call__(
        self,
        z: Float[torch.Tensor, "... N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "... N_res N_res n_bins"]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(z)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        z: Float[torch.Tensor, "... N_res N_res c_pair"],
    ) -> Float[torch.Tensor, "... N_res N_res n_bins"]:
        """Project symmetrised pair embeddings to distance-bin logits.

        Args:
            z: Pair embeddings of shape ``(... N_res N_res c_pair)``.

        Returns:
            Logits of shape ``(... N_res N_res n_bins)``.
        """
        # Symmetrise z_ij_sym = (z_ij + z_ji) / 2
        # transpose(-3, -2) swaps the two N_token dims regardless of leading B
        z_sym = (z + z.transpose(-3, -2)) * 0.5
        x = self.layer_norm(z_sym)
        x = F.relu(self.proj1(x))
        return self.proj2(x)

    def loss(
        self,
        z: Float[torch.Tensor, "... N_res N_res c_pair"],
        targets: Float[torch.Tensor, "... N_res N_res n_bins"],
    ) -> Float[torch.Tensor, ""]:
        """Compute cross-entropy loss against one-hot ground-truth assignments.

        Args:
            z: Pair embeddings of shape ``(... N_res N_res c_pair)``.
            targets: One-hot bin targets of shape ``(... N_res N_res n_bins)``.

        Returns:
            Scalar cross-entropy loss.
        """
        logits = self.forward(z)
        return F.cross_entropy(
            rearrange(logits, "b n1 n2 k -> (b n1 n2) k"),
            rearrange(targets, "b n1 n2 k -> (b n1 n2) k"),
        )


# ---------------------------------------------------------------------------
# 2. AtomDistogramHead
# ---------------------------------------------------------------------------
# Input : p_lm  (N_atom, N_atom, c_atompair) — atom-pair representation
# Output: logits over 22 bins, 0-10 Å
#
# LOCAL WINDOW: only the 5L by 5L sub-block around each atom is used,
# where L is the number of atoms per residue (typically 3-5).
# Atoms outside this window are masked out in both prediction and loss.
# ---------------------------------------------------------------------------


class AtomDistogramHead(nn.Module):
    """Projects atom-pair embeddings to local window distance-bin logits.

    Parameters
    ----------
    c_atompair   : input atom-pair embedding dim
    n_bins       : number of distance bins (default 22)
    d_min        : minimum distance in Å   (default 0.0)
    d_max        : maximum distance in Å   (default 10.0)
    atoms_per_res: L — average atoms per residue, used to define window size
    """

    def __init__(
        self,
        c_atompair: int,
        n_bins: int = 22,
        d_min: float = 0.0,
        d_max: float = 10.0,
        atoms_per_res: int = 3,
    ) -> None:
        super().__init__()
        self.n_bins: int = n_bins
        self.d_min: float = d_min
        self.d_max: float = d_max
        self.window: int = 5 * atoms_per_res  # 5L

        self.layer_norm: LayerNorm = LayerNorm(c_atompair)
        self.proj1: LinearNoBias = LinearNoBias(c_atompair, c_atompair)
        self.proj2: LinearNoBias = LinearNoBias(c_atompair, n_bins)

    def _local_mask(
        self,
        N_atom: int,
        device: torch.device,
    ) -> Bool[torch.Tensor, "N_atom N_atom"]:
        """Build a local-window boolean mask over atom pairs.

        Args:
            N_atom: Number of atoms.
            device: Device for the output tensor.

        Returns:
            Boolean tensor of shape ``(N_atom, N_atom)``; True where
            ``|i - j| <= window // 2``.
        """
        idx = torch.arange(N_atom, device=device)
        dist = (
            rearrange(idx, "n -> 1 n") - rearrange(idx, "n -> n 1")
        ).abs()  # (N, N)
        return dist <= (self.window // 2)  # (N, N) bool

    @override
    def __call__(
        self,
        p: Float[torch.Tensor, "N_atom N_atom c_atompair"],
    ) -> tuple[
        Float[torch.Tensor, "N_atom N_atom n_bins"],
        Bool[torch.Tensor, "N_atom N_atom"],
    ]:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(p)

    @override
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        p: Float[torch.Tensor, "N_atom N_atom c_atompair"],
    ) -> tuple[
        Float[torch.Tensor, "N_atom N_atom n_bins"],
        Bool[torch.Tensor, "N_atom N_atom"],
    ]:
        """Returns:.

        -------
        logits: (N_atom, N_atom, n_bins) — full grid (unmasked pairs = 0 logit)
        mask: (N_atom, N_atom) — True for pairs inside the local window
        """
        N = p.size(0)
        x = self.layer_norm(p)
        x = F.relu(self.proj1(x))
        logits = self.proj2(x)  # (N, N, n_bins)

        mask = self._local_mask(N, p.device)  # (N, N)
        return logits, mask

    def loss(
        self,
        p: Float[torch.Tensor, "N_atom N_atom c_atompair"],
        targets: Float[torch.Tensor, "N_atom N_atom n_bins"],
    ) -> Float[torch.Tensor, ""]:
        """Compute cross-entropy loss restricted to local 5L by 5L atom window.

        Args:
            p: Atom-pair embeddings of shape ``(N_atom N_atom c_atompair)``.
            targets: One-hot bin targets of shape ``(N_atom N_atom n_bins)``.

        Returns:
            Scalar cross-entropy loss averaged over local pairs.
        """
        logits, mask = self.forward(p)  # (N, N, n_bins), (N, N)

        # Apply window mask — flatten and select only local pairs
        logits_local = logits[mask]  # (M, n_bins)
        targets_local = targets[mask]  # (M, n_bins)

        return F.cross_entropy(logits_local, targets_local)


# ---------------------------------------------------------------------------
# MainTrunk — Algorithm 2
# ---------------------------------------------------------------------------


class MainTrunk(nn.Module):
    """Parameters

    ----------
    f_ref_dim    : per-atom f^ref feature size (3 + element_dim after tile)
    n_bins: number of distance bins; used for both TemplateEmbedder input and
                   distogram head output — must equal the number of bins the
                   data
                   distogram produces (base bins + 1 overflow when
                   overflow_bin=True)
    c_atom       : atom single dim
    c_pair       : trunk pair dim         (default 128)
    c_res        : trunk single/residue dim (default 256)
    c_atompair   : atom-pair dim          (default 16)
    sigma_data   : data noise level       (default 16)
    K_unit       : number of decoder units (default 3)
    n_amino      : amino-acid vocabulary  (default 20)
    """

    def __init__(
        self,
        model_params: ModelParams,
        res_distogram_params: ResidueDistogramParams,
        atom_distogram_params: AtomDistogramParams,
        noise_params: NoiseScheduleParams,
    ) -> None:
        super().__init__()
        # 5 * (3+4) 4 because N C O UNK
        self.f_ref_dim: int = model_params.f_ref_dim
        self.sigma_data: float = noise_params.sigma_data
        self.K_unit: int = model_params.K_unit
        self.n_amino: int = model_params.n_amino
        self.c_res: int = model_params.c_res
        self.c_pair: int = model_params.c_pair
        self.c_atom: int = model_params.c_atom
        self.c_atompair: int = model_params.c_atompair
        self.max_residue_number: int = model_params.max_residues
        self.n_blocks_atom_transformer_encoder: int = (
            model_params.n_blocks_atom_transformer_encoder
        )
        self.n_heads_atom_transformer_encoder: int = (
            model_params.n_heads_atom_transformer_encoder
        )
        self.n_blocks_atom_transformer_decoder: int = (
            model_params.n_blocks_atom_transformer_decoder
        )
        self.n_heads_atom_transformer_decoder: int = (
            model_params.n_heads_atom_transformer_decoder
        )
        self.n_pairformer_blocks_template_embedder: int = (
            model_params.n_pairformer_blocks_template_embedder
        )
        self.n_paiformer_heads_template_embedder: int = (
            model_params.n_paiformer_heads_template_embedder
        )

        self.n_residue_bins: int = res_distogram_params.n_bins
        self.n_atom_bins: int = atom_distogram_params.n_bins
        # Step 2: residue-idx feature → s_init
        self.proj_residue_idx: LinearNoBias = LinearNoBias(
            self.max_residue_number,
            self.c_res,
        )
        # Step 3: time Fourier embedding
        self.time_fourier: TimeFourierEmbedding = TimeFourierEmbedding(
            self.c_res,
        )
        # Amino-acid sequence conditioning: 21 entries
        # (0-19 = amino acids, 20 = mask token "X"). magic number
        self.aa_embedding: TypedEmbedding = TypedEmbedding(21, self.c_res)

        # Step 5: relative position encoding → z_init
        self.rel_pos_enc: RelativePositionEncoding = RelativePositionEncoding(
            self.c_pair,
        )

        # Step 6: template embedder
        self.template_embedder: TemplateEmbedder = TemplateEmbedder(
            n_bins=self.n_residue_bins,
            c_z=self.c_pair,
            c=64,  # what is this magic number?
            d=self.c_pair,
            n_blocks=self.n_pairformer_blocks_template_embedder,
            n_heads=self.n_paiformer_heads_template_embedder,
        )

        # Step 7: atom feature encoder
        self.atom_encoder: AtomFeatureEncoder = AtomFeatureEncoder(
            model_params=model_params,
            c=self.c_res,
            d=self.c_atompair,
            m=self.c_atom,
        )

        # Step 8: project s_init → s_i addition
        self.norm_s_init: LayerNorm = LayerNorm(self.c_res)
        self.proj_s_init: LinearNoBias = LinearNoBias(self.c_res, self.c_res)

        # Per-unit modules (steps 11, 12, 16)
        self.node_updates: TypedModuleList[NodeUpdate] = TypedModuleList(
            [NodeUpdate(self.c_res, self.c_pair) for _ in range(self.K_unit)],
        )
        self.atom_decoders: TypedModuleList[AtomAttentionDecoder] = (
            TypedModuleList(
                [
                    AtomAttentionDecoder(
                        model_params=model_params,
                    )
                    for _ in range(self.K_unit)
                ],
            )
        )
        self.pair_updates: TypedModuleList[PairUpdate] = TypedModuleList(
            [PairUpdate(self.c_pair) for _ in range(self.K_unit)],
        )

        self.residue_distogram_head: TypedSequential = TypedSequential(
            LayerNorm(self.c_pair),
            LinearNoBias(self.c_pair, self.c_pair),
            nn.ReLU(),
            LinearNoBias(self.c_pair, self.n_residue_bins),
        )

        self.atom_distogram_head: TypedSequential = TypedSequential(
            LayerNorm(self.c_atompair),
            LinearNoBias(self.c_atompair, self.c_atompair),
            nn.ReLU(),
            LinearNoBias(self.c_atompair, self.n_atom_bins),
        )

        # Per-decoder-unit sequence heads for intermediate aa logit supervision
        self.inter_proj_seq: TypedModuleList[LinearNoBias] = TypedModuleList(
            [LinearNoBias(self.c_atom, self.c_res) for _ in range(self.K_unit)],
        )
        self.inter_seq_logits: TypedModuleList[LinearNoBias] = TypedModuleList(
            [
                LinearNoBias(self.c_res, self.n_amino)
                for _ in range(self.K_unit)
            ],
        )

        # Step 18-19: SeqHead (final)
        self.proj_seq: LinearNoBias = LinearNoBias(self.c_atom, self.c_res)
        self.seq_logits: LinearNoBias = LinearNoBias(self.c_res, self.n_amino)

    # ----------------------------------------------------------------------
    @jaxtyped(typechecker=beartype)
    def embed_inputs(self, batch: FeaturizedBatch) -> EmbeddedInputs:
        """Compute embeddings from featurized batch (steps 1-8 of the trunk).

        Unpacks the batch, builds s_init / t_i / z_ij, runs the
        AtomFeatureEncoder, and applies the skip-connection projection.
        The returned EmbeddedInputs can be passed directly to the decoder loop
        or inspected for interpretability.
        """
        ref_pos: Float[torch.Tensor, "B N_atom 3"] = batch.ref_pos
        ref_element: Float[torch.Tensor, "B N_atom E"] = batch.ref_element
        ref_space_uid: Int[torch.Tensor, "B N_atom"] = batch.ref_space_uid
        f_distogram: Float[torch.Tensor, "B N_res N_res n_bins"] = (
            batch.gt_res_distogram.float()
        )
        f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"] = (
            batch.f_pseudo_beta_mask.float()
        )
        f_residue_idx: Int[torch.Tensor, "B N_res"] = batch.f_residue_idx

        r_input: Float[torch.Tensor, "B N_atom 3"] = batch.r_gt_noised
        t_hat: Float[torch.Tensor, "B"] = batch.t_hat
        t: Float[torch.Tensor, "B N_res N_res"] = batch.t_normalized
        tok_idx: Int[torch.Tensor, "B N_atom"] = batch.tok_idx
        center_uid: Int[torch.Tensor, "B N_atom"] = batch.center_uid
        aa_indices: Int[torch.Tensor, "B N_res"] = batch.aa_indices

        B = r_input.size(0)
        N_atom = r_input.size(1)
        N_res = f_residue_idx.size(1)
        device = r_input.device
        sd = self.sigma_data

        # ------------------------------------------------------------------
        # Step 1: r_scaled = r_input / sqrt(sigma_data² + t̂²)  [B, N_atom, 3]
        # this is c_in from EDM. t_hat is sigma
        # ------------------------------------------------------------------
        denom: Float[torch.Tensor, "B"] = torch.sqrt(sd**2 + t_hat**2)
        r_scaled: Float[torch.Tensor, "B N_atom 3"] = r_input / rearrange(
            denom,
            "b -> b 1 1",
        )

        # ------------------------------------------------------------------
        # Step 2: s_init = LinearNoBias(f_residue_idx)       [B, N_res, c_res]
        # ------------------------------------------------------------------
        s_init: Float[torch.Tensor, "B N_res c_res"] = self.proj_residue_idx(
            F.one_hot(
                f_residue_idx,
                num_classes=self.proj_residue_idx.in_features,
            ).float(),
        )

        # ------------------------------------------------------------------
        # Step 2b: s_init += aa_embedding(aa_indices)
        # Padding (aa_indices = -100) → zero vector via the valid gate.
        # PDB-X / conditioning-dropped (aa_indices = 20) → aa_embedding[20]
        # (null embedding).
        # Real AAs (0-19) -> their embedding.
        # ------------------------------------------------------------------
        valid: Bool[torch.Tensor, "B N_res"] = aa_indices >= 0
        emb: Float[torch.Tensor, "B N_res c_res"] = self.aa_embedding(
            aa_indices.clamp(0, 20),
        )
        s_init = s_init + emb * rearrange(valid.float(), "b n -> b n 1")

        # ------------------------------------------------------------------
        # Step 3: t_i = TimeFourierEmbedding(¼·log(t̂))  [B, N_res, c_res]
        # this is noise conditioning, c_noise from EDM.
        # ------------------------------------------------------------------
        log_val: Float[torch.Tensor, "B"] = 0.25 * torch.log(t_hat)
        log_arg: Float[torch.Tensor, "B N_res"] = repeat(
            log_val,
            "b -> b n",
            n=N_res,
        )
        t_i: Float[torch.Tensor, "B N_res c_res"] = self.time_fourier(log_arg)

        # ------------------------------------------------------------------
        # Step 4: s_init += t_i
        # ------------------------------------------------------------------
        s_init = s_init + t_i

        # ------------------------------------------------------------------
        # Step 5: z_ij = RelativePositionEncoding(f*)    [N_res, N_res, c_pair]
        # expanded to [B, N_res, N_res, c_pair]
        # ------------------------------------------------------------------

        # residue_index is protein.residue_index
        # asym_id = spoof for now. make all zeros.
        # entity_id = spoof for now. make all zeros.
        # token_index = Token number. Increases monotonically; does not restart
        # at 1 for new chains. this is an arange
        # sym_id = spoof for now. make all zeros.
        zeros: Int[torch.Tensor, "B N_res"] = torch.zeros(
            B,
            N_res,
            dtype=torch.long,
            device=device,
        )
        explicit_arange_token_idx: Int[torch.Tensor, "B N_res"] = repeat(
            torch.arange(N_res, dtype=torch.long, device=device),
            "n -> b n",
            b=B,
        )

        z_ij: Float[torch.Tensor, "B N_res N_res c_pair"] = self.rel_pos_enc(
            residue_index=f_residue_idx,
            asym_id=zeros,
            entity_id=zeros,
            token_index=explicit_arange_token_idx,
            sym_id=zeros,
        )

        # ------------------------------------------------------------------
        # Step 6: z_ij += TemplateEmbedder(...)    [B, N_res, N_res, c_pair]
        # TemplateEmbedder already operates on [B, N_res, N_res, *]
        # ------------------------------------------------------------------
        z_ij = z_ij + self.template_embedder(
            f_distogram,
            f_pseudo_beta_mask,
            z_ij,
            t,
        )

        # ------------------------------------------------------------------
        # Step 7: AtomFeatureEncoder — all tensors [B, N_atom/*N_res, *]
        # ------------------------------------------------------------------
        s_i: Float[torch.Tensor, "B N_res c_res"]
        q_skip: Float[torch.Tensor, "B N_atom c_atom"]
        c_skip: Float[torch.Tensor, "B N_atom c_atom"]
        p_skip: Float[torch.Tensor, "B N_atom K c_atompair"]
        c_l: Float[torch.Tensor, "B N_atom c_atom"]
        s_i, q_skip, c_skip, p_skip, c_l = self.atom_encoder(
            ref_pos,
            ref_element,
            ref_space_uid,
            s_init,
            z_ij,
            r_scaled,
            tok_idx,
        )

        # ------------------------------------------------------------------
        # Step 8: s_i += LinearNoBias(LayerNorm(s_init))
        # ------------------------------------------------------------------
        s_i = s_i + self.proj_s_init(self.norm_s_init(s_init))

        return EmbeddedInputs(
            s_i=s_i,
            t_i=t_i,
            z_ij=z_ij,
            q_skip=q_skip,
            c_skip=c_skip,
            p_skip=p_skip,
            c_l=c_l,
            r_input=r_input,
            tok_idx=tok_idx,
            center_uid=center_uid,
            t_hat=t_hat,
            B=B,
            N_atom=N_atom,
            N_res=N_res,
            device=device,
            denom=denom,
        )

    @override
    def __call__(self, batch: FeaturizedBatch) -> PredictedOutputs:
        """Call forward; typed override so call-site return types are not Any.

        See ``forward`` for full documentation of arguments and return values.
        """
        return self.forward(batch)

    # ----------------------------------------------------------------------
    @override
    @jaxtyped(typechecker=beartype)
    def forward(self, batch: FeaturizedBatch) -> PredictedOutputs:
        """Run main trunk and return predicted coords and sequence.

        Args:
            batch: Featurized protein batch with all input tensors.

        Returns:
            PredictedOutputs containing r_denoised, seq_logits, residue and
            atom distogram logits, and per-decoder-block intermediate outputs.
        """
        emb = self.embed_inputs(batch)
        s_i = emb.s_i
        z_ij = emb.z_ij
        c_l = emb.c_l
        sd = self.sigma_data

        # ------------------------------------------------------------------
        # Step 9: initialise accumulators                  [B, N_atom, 3]
        # ------------------------------------------------------------------
        r_updates: Float[torch.Tensor, "B N_atom 3"] = torch.zeros(
            emb.B,
            emb.N_atom,
            3,
            device=emb.device,
        )
        r_denoised: Float[torch.Tensor, "B N_atom 3"] = emb.r_input

        # ------------------------------------------------------------------
        # Steps 10-17: K_unit decoder loop
        # ------------------------------------------------------------------
        intermediate_denoised_coord_stack: list[
            Float[torch.Tensor, "B N_atom 3"]
        ] = []
        intermediate_pred_aa_logit_stack: list[
            Float[torch.Tensor, "B N_res n_amino"]
        ] = []

        # Vectorized tok_idx offset base (reused for every scatter in the loop)
        tok_offset_base: Int[torch.Tensor, "B N_atom"] = (
            emb.tok_idx
            + rearrange(torch.arange(emb.B, device=emb.device), "b -> b 1")
            * emb.N_res
        )

        q_update: Float[torch.Tensor, "B N_atom c_atom"] = emb.q_skip
        p_update: Float[torch.Tensor, "B N_atom K c_atompair"] = emb.p_skip

        for k in range(self.K_unit):

            # Step 11: s_i = NodeUpdate(s_i, t_i, z_ij)    [B, N_res, c_res]
            s_i = self.node_updates[k](s_i, emb.t_i, z_ij)

            # Step 12: AtomAttentionDecoder
            r_update: Float[torch.Tensor, "B N_atom 3"]

            q_update, p_update, r_update, c_l = self.atom_decoders[k](
                emb.q_skip,
                emb.p_skip,
                emb.c_skip,
                c_l,
                s_i,
                z_ij,
                emb.tok_idx,
            )

            # Step 13: r_updates += r_update
            r_updates = r_updates + r_update

            # Step 14: EDM adjustment
            # c_skip = sigma²/(sigma²+t̂²)
            # c_out = sigma·t̂/√(sigma²+t̂²)
            # EDM r_denoised = c_skip * r_input + c_out * r_updates
            w_gt = rearrange(sd**2 / (sd**2 + emb.t_hat**2), "b -> b 1 1")
            w_upd = rearrange((sd * emb.t_hat) / emb.denom, "b -> b 1 1")
            r_denoised = w_gt * emb.r_input + w_upd * r_updates

            intermediate_denoised_coord_stack.append(r_denoised)

            # Intermediate aa logits: scatter c_l atoms to residues
            # [B, N_res, n_amino] n_amino should be 0-19 -> 20
            proj_c: Float[torch.Tensor, "B N_atom c_res"] = F.relu(
                self.inter_proj_seq[k](c_l),
            )
            a_inter: Float[torch.Tensor, "B N_res c_res"] = scatter_mean(
                proj_c,
                tok_offset_base,
                emb.B * emb.N_res,
                emb.B,
            )
            inter_logits: Float[torch.Tensor, "B N_res n_amino"] = (
                self.inter_seq_logits[k](a_inter)
            )
            intermediate_pred_aa_logit_stack.append(inter_logits)

            # Step 15: r_center = r_denoised[center_uid]  [B, N_res, 3]
            # center_uid is [B, N_atom]: every atom stores its residue's center
            # atom index. Stride by atoms_per_res to get one unique center
            # index per residue [B, N_res], then gather from r_denoised along
            # the atom dimension.
            atoms_per_res: int = emb.N_atom // emb.N_res
            center_uid_per_res: Int[torch.Tensor, "B N_res"] = emb.center_uid[
                :,
                ::atoms_per_res,
            ]
            r_center: Float[torch.Tensor, "B N_res 3"] = r_denoised.gather(
                1,
                repeat(center_uid_per_res, "b n -> b n d", d=3),
            )

            # Step 16:
            z_ij = self.pair_updates[k](z_ij, r_center)

        # ------------------------------------------------------------------
        # Distogram heads
        # ------------------------------------------------------------------
        residue_distogram_logits: Float[
            torch.Tensor,
            "B N_res N_res n_bins",
        ] = self.residue_distogram_head(z_ij)

        # Project atom-pair representation from local atomic attention into
        # distance bins. Local region defined by the attention window within
        # the 5L by 5L atomic-level map.
        atom_distogram_logits: Float[torch.Tensor, "B N_atom K n_atom_bins"] = (
            self.atom_distogram_head(p_update)
        )

        # ------------------------------------------------------------------
        # Step 18:
        # a_i = mean_{l: tok_idx(l)=i} ReLU(proj(q_update))  [B, N_res, c_res]
        # ------------------------------------------------------------------
        proj_q: Float[torch.Tensor, "B N_atom c_res"] = F.relu(
            self.proj_seq(q_update),
        )
        a_i: Float[torch.Tensor, "B N_res c_res"] = scatter_mean(
            proj_q,
            tok_offset_base,
            emb.B * emb.N_res,
            emb.B,
        )
        # Step 19: f_seq_logits = LinearNoBias(a_i)  [B, N_res, n_amino]
        f_seq_logits: Float[torch.Tensor, "B N_res n_amino"] = self.seq_logits(
            a_i,
        )

        return PredictedOutputs(
            r_denoised=r_denoised,
            seq_logits=f_seq_logits,
            residue_distogram_logits=residue_distogram_logits,
            atom_distogram_logits=atom_distogram_logits,
            intermediate_denoised_coord_stack=intermediate_denoised_coord_stack,
            intermediate_pred_aa_logit_stack=intermediate_pred_aa_logit_stack,
        )

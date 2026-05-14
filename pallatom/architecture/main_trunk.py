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
from architecture.atom_transformers import AtomAttentionDecoder, AtomFeatureEncoder, LinearNoBias
from architecture.node_update import NodeUpdate
from architecture.pair_update import PairUpdate
from architecture.template_embedder import TemplateEmbedder
from beartype import beartype
from einops import rearrange
from helpers.featurize import FeaturizedBatch  # re-exported for callers
from jaxtyping import Float, Int, jaxtyped


@jaxtyped(typechecker=beartype)
def scatter_mean(
    src: Float[torch.Tensor, "B N_src C"],
    index: Int[torch.Tensor, "B N_src"],
    num_segments: int,
    B: int,
) -> Float[torch.Tensor, "B N_target C"]:
    """Per-segment mean pooling via scatter.

    Maps atom-level features to residue-level by averaging all atoms that share
    the same flat segment index.  `index` must already encode the batch offset
    (i.e. atom j in batch item b maps to index[b, j] = tok_idx[b, j] + b * N_tgt).
    """
    C: int = src.size(-1)
    device = src.device

    flat_index: Int[torch.Tensor, "BN_src"] = rearrange(index, "b n -> (b n)")
    flat_src: Float[torch.Tensor, "BN_src C"] = rearrange(src, "b n c -> (b n) c")

    sum_flat: Float[torch.Tensor, "BN_target C"] = torch.zeros(num_segments, C, device=device)
    sum_flat.scatter_add_(0, flat_index.unsqueeze(1).expand(-1, C), flat_src)

    cnt_flat: Float[torch.Tensor, "BN_target 1"] = torch.zeros(num_segments, 1, device=device)
    cnt_flat.scatter_add_(
        0, flat_index.unsqueeze(1), torch.ones(flat_index.size(0), 1, device=device)
    )

    result: Float[torch.Tensor, "B N_target C"] = rearrange(
        sum_flat / cnt_flat.clamp(min=1), "(b n) c -> b n c", b=B
    )
    return result


# ---------------------------------------------------------------------------
# TimeFourierEmbedding  (used in step 3)
# ---------------------------------------------------------------------------


class TimeFourierEmbedding(nn.Module):
    """Maps scalar x = ¼·log(t̂/σ_data) to a Fourier feature vector ∈ R^{c_res}.

    Uses learnable frequencies (as in AF3 / common diffusion practice).
    """

    def __init__(self, c_res: int) -> None:
        super().__init__()
        assert c_res % 2 == 0, "c_res must be even for sin/cos pairs"
        self.freqs = nn.Parameter(torch.randn(c_res // 2))
        self.proj = LinearNoBias(c_res, c_res)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map scalar noise-level encoding to a Fourier feature vector.

        Args:
            x: Scalar or batched input (any leading shape); typically ¼·log(t̂/σ_data).

        Returns:
            Projected Fourier embedding of shape (*x.shape, c_res).
        """
        # x: any leading shape; last dim expanded to c_res
        angles = 2 * math.pi * x.unsqueeze(-1) * self.freqs  # (..., c_res//2)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(emb)


# ---------------------------------------------------------------------------
# RelativePositionEncoding  (step 5)
# Encodes relative token positions into pair embeddings z_ij^init.
# ---------------------------------------------------------------------------


class RelativePositionEncoding(nn.Module):
    """Standard clipped relative position encoding.

    Produces z_ij^init ∈ R^{c_pair} from residue index differences.
    """

    def __init__(self, c_pair: int, max_rel: int = 32) -> None:
        super().__init__()
        self.max_rel = max_rel
        n_bins = 2 * max_rel + 1
        self.proj = LinearNoBias(n_bins, c_pair)

    def forward(self, N_token: int, device: torch.device) -> torch.Tensor:
        """Compute relative position embeddings for a sequence of tokens.

        Args:
            N_token: Number of residue tokens.
            device: Target device for the output tensor.

        Returns:
            Pair embedding of shape (N_token, N_token, c_pair).
        """
        idx = torch.arange(N_token, device=device)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # (N, N)
        diff = diff.clamp(-self.max_rel, self.max_rel) + self.max_rel  # shift to [0, 2R]
        n_bins = 2 * self.max_rel + 1
        onehot = F.one_hot(diff, num_classes=n_bins).float()  # (N, N, n_bins)
        return self.proj(onehot)  # (N, N, c_pair)


# ---------------------------------------------------------------------------
# 1. ResidueDistogramHead
# ---------------------------------------------------------------------------
# Input : z_ij  (B, N_token, N_token, c_pair)  — global trunk pair representation
# Output: logits (B, N_token, N_token, 64)
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
        self.n_bins = n_bins
        self.d_min = d_min
        self.d_max = d_max

        self.norm = nn.LayerNorm(c_pair)
        self.proj1 = LinearNoBias(c_pair, c_pair)
        self.proj2 = LinearNoBias(c_pair, n_bins)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Z : (..., N_token, N_token, c_pair); returns logits : (..., N_token, N_token, n_bins)."""
        # Symmetrise: z_ij_sym = (z_ij + z_ji) / 2
        # transpose(-3, -2) swaps the two N_token dims regardless of leading B
        z_sym = (z + z.transpose(-3, -2)) * 0.5
        x = self.norm(z_sym)
        x = F.relu(self.proj1(x))
        return self.proj2(x)

    def loss(
        self,
        z: torch.Tensor,  # (..., N_token, N_token, c_pair)
        targets: torch.Tensor,  # (..., N, N, n_bins)  one-hot
    ) -> torch.Tensor:
        """Convenience: compute cross-entropy loss against ground-truth positions."""
        logits = self.forward(z)
        return F.cross_entropy(
            logits.reshape(-1, self.n_bins),
            targets.reshape(-1, self.n_bins),
        )


# ---------------------------------------------------------------------------
# 2. AtomDistogramHead
# ---------------------------------------------------------------------------
# Input : p_lm  (N_atom, N_atom, c_atompair) — atom-pair representation
# Output: logits over 22 bins, 0–10 Å
#
# LOCAL WINDOW: only the 5L × 5L sub-block around each atom is used,
# where L is the number of atoms per residue (typically 3–5).
# Atoms outside this window are masked out in both prediction and loss.
# ---------------------------------------------------------------------------


class AtomDistogramHead(nn.Module):
    """Projects atom-pair embeddings p_lm → 22 distance-bin logits, restricted to local window.

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
        self.n_bins = n_bins
        self.d_min = d_min
        self.d_max = d_max
        self.window = 5 * atoms_per_res  # 5L

        self.norm = nn.LayerNorm(c_atompair)
        self.proj1 = LinearNoBias(c_atompair, c_atompair)
        self.proj2 = LinearNoBias(c_atompair, n_bins)

    def _local_mask(self, N_atom: int, device: torch.device) -> torch.Tensor:
        """Boolean mask (N_atom, N_atom): True where |i-j| <= window//2."""
        idx = torch.arange(N_atom, device=device)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # (N, N)
        return dist <= (self.window // 2)  # (N, N) bool

    def forward(
        self,
        p: torch.Tensor,  # (N_atom, N_atom, c_atompair)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns:.

        -------
        logits : (N_atom, N_atom, n_bins)  — full grid (unmasked pairs = 0 logit)
        mask   : (N_atom, N_atom)          — True for pairs inside the local window
        """
        N = p.size(0)
        x = self.norm(p)
        x = F.relu(self.proj1(x))
        logits = self.proj2(x)  # (N, N, n_bins)

        mask = self._local_mask(N, p.device)  # (N, N)
        return logits, mask

    def loss(
        self,
        p: torch.Tensor,  # (N_atom, N_atom, c_atompair)
        targets: torch.Tensor,  # atom residue distogram # (N, N, n_bins)  one-hot
    ) -> torch.Tensor:
        """Cross-entropy loss over the local 5L × 5L window only."""
        logits, mask = self.forward(p)  # (N, N, n_bins), (N, N)

        # Apply window mask — flatten and select only local pairs
        logits_local = logits[mask]  # (M, n_bins)
        targets_local = targets[mask]  # (M, n_bins)

        return F.cross_entropy(logits_local, targets_local)


# ---------------------------------------------------------------------------
# EmbeddedInputs — output of MainTrunk.embed_inputs (steps 1–8)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EmbeddedInputs:
    """Initial embeddings produced by MainTrunk.embed_inputs before the decoder loop."""

    s_i: Float[torch.Tensor, "B N_res c_res"]
    t_i: Float[torch.Tensor, "B N_res c_res"]
    z_ij: Float[torch.Tensor, "B N_res N_res c_pair"]
    q_skip: Float[torch.Tensor, "B N_atom c_atom"]
    c_skip: Float[torch.Tensor, "B N_atom c_atom"]
    p_skip: Float[torch.Tensor, "B N_atom K c_atompair"]
    c_l: Float[torch.Tensor, "B N_atom c_atom"]
    r_input: Float[torch.Tensor, "B N_atom 3"]
    tok_idx: Int[torch.Tensor, "B N_atom"]
    center_uid: Int[torch.Tensor, "B N_res"]
    t_hat: float
    B: int
    N_atom: int
    N_res: int
    device: torch.device
    denom: float


# ---------------------------------------------------------------------------
# MainTrunk — Algorithm 2
# ---------------------------------------------------------------------------


class MainTrunk(nn.Module):
    """Parameters

    ----------
    f_ref_dim    : per-atom f^ref feature size (3 + element_dim after tile)
    n_bins       : distogram bins for TemplateEmbedder
    c_atom       : atom single dim        (default 128)
    c_pair       : trunk pair dim         (default 128)
    c_res        : trunk single/residue dim (default 256)
    c_atompair   : atom-pair dim          (default 16)
    sigma_data   : data noise level       (default 16)
    K_unit       : number of decoder units (default 3)
    n_amino      : amino-acid vocabulary  (default 20)
    """

    def __init__(
        self,
        f_ref_dim: int = 35,  # 5 * 7
        n_bins: int = 38,
        n_atom_bins: int = 22,
        c_atom: int = 128,
        c_pair: int = 128,
        c_res: int = 256,
        c_atompair: int = 16,
        n_blocks: int = 2,
        n_heads: int = 4,
        sigma_data: float = 16.0,
        K_unit: int = 3,
        n_amino: int = 20,
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.K_unit = K_unit

        # Step 2: residue-idx feature → s_init
        self.proj_residue_idx = LinearNoBias(c_res, c_res)
        # Step 3: time Fourier embedding
        self.time_fourier = TimeFourierEmbedding(c_res)
        # Amino-acid sequence conditioning: 21 entries (0-19 = amino acids, 20 = mask token "X")
        self.aa_embedding = nn.Embedding(21, c_res)

        # Step 5: relative position encoding → z_init
        self.rel_pos_enc = RelativePositionEncoding(c_pair)

        # Step 6: template embedder
        self.template_embedder = TemplateEmbedder(
            n_bins=n_bins, c_z=c_pair, c=64, d=c_pair, n_blocks=2
        )

        # Step 7: atom feature encoder
        self.atom_encoder = AtomFeatureEncoder(
            f_ref_dim=f_ref_dim,
            c_token=c_res,
            c_pair=c_pair,
            c=c_res,
            d=c_atompair,
            m=c_atom,
            n_blocks=n_blocks,
            n_heads=n_heads,
        )

        # Step 8: project s_init → s_i addition
        self.norm_s_init = nn.LayerNorm(c_res)
        self.proj_s_init = LinearNoBias(c_res, c_res)

        # Per-unit modules (steps 11, 12, 16)
        self.node_updates = nn.ModuleList([NodeUpdate(c_res, c_pair) for _ in range(K_unit)])
        self.atom_decoders = nn.ModuleList(
            [
                AtomAttentionDecoder(
                    c_token=c_res,
                    c_pair=c_pair,
                    c_atom=c_atom,
                    c_atompair=c_atompair,
                )
                for _ in range(K_unit)
            ]
        )
        self.pair_updates = nn.ModuleList([PairUpdate(c_pair) for _ in range(K_unit)])

        self.residue_distogram_head = nn.Sequential(
            nn.LayerNorm(c_pair),
            LinearNoBias(c_pair, c_pair),
            nn.ReLU(),
            LinearNoBias(c_pair, n_bins),
        )

        self.atom_distogram_head = nn.Sequential(
            nn.LayerNorm(c_atompair),
            LinearNoBias(c_atompair, c_atompair),
            nn.ReLU(),
            LinearNoBias(c_atompair, n_atom_bins),
        )

        # Per-decoder-unit sequence heads for intermediate aa logit supervision
        self.inter_proj_seq = nn.ModuleList([LinearNoBias(c_atom, c_res) for _ in range(K_unit)])
        self.inter_seq_logits = nn.ModuleList([LinearNoBias(c_res, n_amino) for _ in range(K_unit)])

        # Step 18-19: SeqHead (final)
        self.proj_seq = LinearNoBias(c_atom, c_res)
        self.seq_logits = LinearNoBias(c_res, n_amino)

    # ----------------------------------------------------------------------
    @jaxtyped(typechecker=beartype)
    def embed_inputs(self, batch: FeaturizedBatch) -> EmbeddedInputs:
        """Compute initial embeddings from a featurized batch (steps 1–8 of the trunk).

        Unpacks the batch, builds s_init / t_i / z_ij, runs the AtomFeatureEncoder,
        and applies the skip-connection projection.  The returned EmbeddedInputs can
        be passed directly to the decoder loop or inspected for interpretability.
        """
        # n_templ_bins ≠ n_bins: template distogram has an overflow bin (39);
        # the distogram heads produce n_bins (38) from the trunk config.
        ref_pos: Float[torch.Tensor, "B N_atom 3"] = batch.ref_pos
        ref_element: Float[torch.Tensor, "B N_atom E"] = batch.ref_element
        ref_space_uid: Int[torch.Tensor, "B N_atom"] = batch.ref_space_uid
        f_distogram: Float[torch.Tensor, "B N_res N_res n_templ_bins"] = (
            batch.gt_res_distogram.float()
        )
        f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"] = batch.f_pseudo_beta_mask.float()
        f_residue_idx: Float[torch.Tensor, "B N_res c_res"] = batch.f_residue_idx
        r_input: Float[torch.Tensor, "B N_atom 3"] = batch.r_input
        t_hat: float = batch.t_hat
        t: float = batch.t_normalized
        tok_idx: Int[torch.Tensor, "B N_atom"] = batch.tok_idx
        center_uid: Int[torch.Tensor, "B N_res"] = batch.center_uid
        aa_indices: Int[torch.Tensor, "B N_res"] = batch.aa_indices

        B = r_input.size(0)
        N_atom = r_input.size(1)
        N_res = f_residue_idx.size(1)
        device = r_input.device
        sd = self.sigma_data

        # ------------------------------------------------------------------
        # Step 1: r_scaled = r_input / sqrt(σ_data² + t̂²)    [B, N_atom, 3]
        # ------------------------------------------------------------------
        r_scaled: Float[torch.Tensor, "B N_atom 3"] = r_input / math.sqrt(sd**2 + t_hat**2)

        # ------------------------------------------------------------------
        # Step 2: s_init = LinearNoBias(f_residue_idx)         [B, N_res, c_res]
        # ------------------------------------------------------------------
        s_init: Float[torch.Tensor, "B N_res c_res"] = self.proj_residue_idx(f_residue_idx)

        # ------------------------------------------------------------------
        # Step 2b: s_init += aa_embedding(aa_indices)
        # Clamp to [0, 20]: padding residues have aa_indices=-100 which clamps to 0
        # (padding is excluded from all downstream losses; embedding value doesn't matter).
        # Dropout-masked residues have aa_indices=20 and get the learned "X" embedding.
        # ------------------------------------------------------------------
        aa_idx_clamped: Int[torch.Tensor, "B N_res"] = aa_indices.clamp(min=0, max=20)
        s_init = s_init + self.aa_embedding(aa_idx_clamped)

        # ------------------------------------------------------------------
        # Step 3: t_i = TimeFourierEmbedding(¼·log(t̂/σ_data))  [B, N_res, c_res]
        # ------------------------------------------------------------------
        log_val = 0.25 * math.log(t_hat / sd + 1e-8)
        log_arg = torch.full((B, N_res), log_val, device=device)
        t_i: Float[torch.Tensor, "B N_res c_res"] = self.time_fourier(log_arg)

        # ------------------------------------------------------------------
        # Step 4: s_init += t_i
        # ------------------------------------------------------------------
        s_init = s_init + t_i

        # ------------------------------------------------------------------
        # Step 5: z_ij = RelativePositionEncoding(f*)    [N_res, N_res, c_pair]
        # expanded to [B, N_res, N_res, c_pair]
        # ------------------------------------------------------------------
        z_ij_base: Float[torch.Tensor, "N_res N_res c_pair"] = self.rel_pos_enc(N_res, device)
        z_ij: Float[torch.Tensor, "B N_res N_res c_pair"] = z_ij_base.unsqueeze(0).expand(
            B, -1, -1, -1
        )

        # ------------------------------------------------------------------
        # Step 6: z_ij += TemplateEmbedder(...)          [B, N_res, N_res, c_pair]
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
            denom=math.sqrt(sd**2 + t_hat**2),
        )

    # ----------------------------------------------------------------------
    @jaxtyped(typechecker=beartype)
    def forward(self, batch: FeaturizedBatch) -> tuple[
        Float[torch.Tensor, "B N_atom 3"],
        Float[torch.Tensor, "B N_res n_amino"],
        Float[torch.Tensor, "B N_res N_res n_bins"],
        Float[torch.Tensor, "B N_atom K n_atom_bins"],
        list[Float[torch.Tensor, "B N_atom 3"]],
        list[Float[torch.Tensor, "B N_res n_amino"]],
    ]:
        """Run full denoising trunk and return predicted coords, sequence and distogram logits."""
        emb = self.embed_inputs(batch)
        s_i = emb.s_i
        z_ij = emb.z_ij
        c_l = emb.c_l
        sd = self.sigma_data

        # ------------------------------------------------------------------
        # Step 9: initialise accumulators                  [B, N_atom, 3]
        # ------------------------------------------------------------------
        r_updates: Float[torch.Tensor, "B N_atom 3"] = torch.zeros(
            emb.B, emb.N_atom, 3, device=emb.device
        )
        r_denoised: Float[torch.Tensor, "B N_atom 3"] = emb.r_input

        # ------------------------------------------------------------------
        # Steps 10-17: K_unit decoder loop
        # ------------------------------------------------------------------
        intermediate_denoised_coord_stack: list[Float[torch.Tensor, "B N_atom 3"]] = []
        intermediate_pred_aa_logit_stack: list[Float[torch.Tensor, "B N_res n_amino"]] = []

        # Vectorized tok_idx offset base (reused for every scatter in the loop)
        tok_offset_base: Int[torch.Tensor, "B N_atom"] = (
            emb.tok_idx + torch.arange(emb.B, device=emb.device).unsqueeze(1) * emb.N_res
        )

        for k in range(self.K_unit):

            # Step 11: s_i = NodeUpdate(s_i, t_i, z_ij)    [B, N_res, c_res]
            s_i = self.node_updates[k](s_i, emb.t_i, z_ij)

            # Step 12: AtomAttentionDecoder
            q_update: Float[torch.Tensor, "B N_atom c_atom"]
            p_update: Float[torch.Tensor, "B N_atom K c_atompair"]
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

            # Step 14: r_denoised = σ²/(σ²+t̂²)·r_input + σ·t̂/√(σ²+t̂²)·r_updates
            r_denoised = (sd**2 / (sd**2 + emb.t_hat**2)) * emb.r_input + (
                sd * emb.t_hat / emb.denom
            ) * r_updates

            intermediate_denoised_coord_stack.append(r_denoised)

            # Intermediate aa logits: scatter c_l atoms to residues [B, N_res, n_amino]
            proj_c: Float[torch.Tensor, "B N_atom c_res"] = F.relu(self.inter_proj_seq[k](c_l))
            a_inter: Float[torch.Tensor, "B N_res c_res"] = scatter_mean(
                proj_c, tok_offset_base, emb.B * emb.N_res, emb.B
            )
            inter_logits: Float[torch.Tensor, "B N_res n_amino"] = self.inter_seq_logits[k](a_inter)
            intermediate_pred_aa_logit_stack.append(inter_logits)

            # Step 15: r_center = r_denoised[:, center_uid[0]]  [B, N_res, 3]
            # center_uid identical across batch items
            r_center: Float[torch.Tensor, "B N_res 3"] = r_denoised[:, emb.center_uid[0]]

            # Step 16: z_ij = PairUpdate(z_ij, r_center)        [B, N_res, N_res, c_pair]
            z_ij = self.pair_updates[k](z_ij, r_center)

        # ------------------------------------------------------------------
        # Distogram heads
        # ------------------------------------------------------------------
        residue_distogram_logits: Float[torch.Tensor, "B N_res N_res n_bins"] = (
            self.residue_distogram_head(z_ij)
        )

        # Project atom-pair representation from local atomic attention into distance bins.
        # Local region defined by the attention window within the 5L × 5L atomic-level map.
        atom_distogram_logits: Float[torch.Tensor, "B N_atom K n_atom_bins"] = (
            self.atom_distogram_head(p_update)
        )

        # ------------------------------------------------------------------
        # Step 18: a_i = mean_{l: tok_idx(l)=i} ReLU(proj(q_update))  [B, N_res, c_res]
        # ------------------------------------------------------------------
        proj_q: Float[torch.Tensor, "B N_atom c_res"] = F.relu(self.proj_seq(q_update))
        a_i: Float[torch.Tensor, "B N_res c_res"] = scatter_mean(
            proj_q, tok_offset_base, emb.B * emb.N_res, emb.B
        )
        # Step 19: f_seq_logits = LinearNoBias(a_i)             [B, N_res, n_amino]
        f_seq_logits: Float[torch.Tensor, "B N_res n_amino"] = self.seq_logits(a_i)

        return (
            r_denoised,
            f_seq_logits,
            residue_distogram_logits,
            atom_distogram_logits,
            intermediate_denoised_coord_stack,
            intermediate_pred_aa_logit_stack,
        )

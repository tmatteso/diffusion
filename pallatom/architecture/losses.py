"""Loss functions for diffusion model training.

Implements atom-coordinate MSE (Kabsch-aligned), smooth lDDT, residue and
atom distogram cross-entropy, intermediate (L_med) supervision, and sequence
cross-entropy losses.
"""

import torch
import torch.nn.functional as F
from architecture.errors import (
    BlockCountMismatchError,
    LossComputationError,
    NoDenoiserBlockError,
)
from beartype import beartype
from einops import einsum, rearrange
from helpers.alignment import kabsch_align
from helpers.batch_types import FeaturizedBatch
from jaxtyping import Bool, Float, Int, jaxtyped
from train.train_config import LossParams

PAD_TOKEN = -100
# ---------------------------------------------------------------------------
# atom_loss  (Algorithm 2 structure term)
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def atom_loss(
    r_denoised: Float[torch.Tensor, "B N_res 3"],
    r_gt: Float[torch.Tensor, "B N_res 3"],
    mask: Bool[torch.Tensor, "B N_res"] | None = None,
    *,
    lambda_sigma_weight: Float[torch.Tensor, "B"],
) -> Float[torch.Tensor, "B"]:
    """Kabsch-aligned atom-coordinate MSE loss with EDM noise weighting.

    Rigidly aligns the ground-truth structure onto the denoised structure
    (i.e. the *prediction* is held fixed; the GT is rotated/translated to
    match it), then computes the mean squared deviation per coordinate:

        L_atom = lambda_sigma · ||r_denoised - r_aligned||² / (3L)

    where L is the number of (unmasked) residues, the factor 3 accounts
    for the x, y, z dimensions, and ``lambda_sigma = (t̂² + sigma_data²) /
    (t̂ · sigma_data)²`` is the per-sample EDM noise weighting.

    Args:
        r_denoised: (B, N_res, 3) — model output / denoised coordinates.
        r_gt:       (B, N_res, 3) — ground-truth coordinates  r̄⁰.
        mask:       (B, N_res)    — boolean or float mask; 1 = valid residue.
                                    When None, all N_res residues are used.
        lambda_sigma_weight: Per-batch EDM loss weight
            ``(t̂² + sigma_data²) / (t̂ · sigma_data)²``, shape ``(B,)``.

    Returns:
        (B,) EDM-weighted loss per batch element.

    Raises:
        LossComputationError: If the computed loss contains NaN values.

    Notes:
        • Alignment is done with float weights derived from `mask`, so
          missing / padded residues do not pollute the rotation estimate.
        • Gradients flow through r_denoised (r_aligned is treated as a
          constant frame — the GT is being moved, not the prediction).
    """
    weights: Float[torch.Tensor, "B N_res"] | None = (
        mask.float() if mask is not None else None
    )

    # Align GT → denoised  (mobile=r_gt, target=r_denoised).
    # Detach so gradients flow only through r_denoised, not through the SVD.
    (r_aligned,) = kabsch_align(  # pylint: disable=unpacking-non-sequence
        r_gt,
        r_denoised,
        weights=weights,
        return_transform=False,
    )
    r_aligned = r_aligned.detach()

    # Squared residuals summed over xyz → (B, N_res)
    diff: Float[torch.Tensor, "B N_res 3"] = r_denoised - r_aligned
    sq: Float[torch.Tensor, "B N_res"] = einsum(
        diff,
        diff,
        "b l d, b l d -> b l",
    )

    if mask is not None:
        m: Float[torch.Tensor, "B N_res"] = mask.float()
        L_eff: Float[torch.Tensor, "B"] = m.sum(dim=-1).clamp(min=1)
        loss: Float[torch.Tensor, "B"] = einsum(
            sq,
            m,
            "b l, b l -> b",
        ) / (3.0 * L_eff)
    else:
        L = r_denoised.shape[-2]
        loss = sq.sum(dim=-1) / (3.0 * L)

    loss = loss * lambda_sigma_weight

    if torch.isnan(loss).any():
        raise LossComputationError

    return loss


# ---------------------------------------------------------------------------
# Intermediate supervision loss with decoder-block weight decay  (L_med)
# ---------------------------------------------------------------------------


def med_loss(
    r_denoised_blocks: list[Float[torch.Tensor, "K N_atom 3"]],
    logits_aa_blocks: list[Float[torch.Tensor, "K N_res n_amino"]],
    batch: FeaturizedBatch,
    loss_params: LossParams,
    lambda_sigma_weight: Float[torch.Tensor, "B"],
) -> Float[torch.Tensor, ""]:
    """Intermediate loss over K decoder blocks.

        L_med = (1/K) · Σ_{k=1}^{K} gamma^(K-k) · L^k_med

    gamma < 1 means early blocks get lower weight (gamma^(K-1), gamma^(K-2), …,
    gamma^0 = 1), so the final block always receives weight 1 and earlier
    blocks are progressively discounted.

    The EDM noise weighting ``lambda_sigma_weight`` is applied to the
    structural term of each block, matching the outer Kabsch MSE loss so all
    loss components remain on the same scale across noise levels.

    Args:
        r_denoised_blocks: K tensors of denoised atom positions, each
            ``(B, N_atom, 3)``.
        logits_aa_blocks: K tensors of sequence logits, each
            ``(B, N_res, n_amino)``.
        batch: Ground-truth coordinates, atom masks, and amino-acid indices.
        loss_params: Scalar hyperparameters lam (lambda), alpha_0, and gamma
            controlling loss weighting.
        lambda_sigma_weight: Per-batch EDM loss weight
            ``(t̂² + sigma_data²) / (t̂ · sigma_data)²``, shape ``(B,)``.

    Returns:
        Scalar mean intermediate loss L_med.

    Raises:
        NoDenoiserBlockError: if ``r_denoised_blocks`` is empty.
        BlockCountMismatchError: if the two block lists have different lengths.
        LossComputationError: If the accumulated intermediate loss contains NaN.
    """
    lam = loss_params.lam
    alpha_0 = loss_params.alpha_0
    gamma = loss_params.gamma

    K = len(r_denoised_blocks)
    if K == 0:
        raise NoDenoiserBlockError
    if len(logits_aa_blocks) != K:
        raise BlockCountMismatchError(K, len(logits_aa_blocks))

    intermediate_loss: Float[torch.Tensor, "B"] = torch.zeros(
        lambda_sigma_weight.shape[0],
        device=r_denoised_blocks[0].device,
    )
    for k_idx, intermediate_denoised_coord in enumerate(
        r_denoised_blocks,
    ):
        intermediate_denoised_coord: Float[torch.Tensor, "B N_atom 3"]
        gamma_K_minus_k: float = gamma ** (K - k_idx - 1)
        seq_loss: Float[torch.Tensor, ""] = seq_ce_loss(
            logits_aa_blocks[k_idx],
            batch.aa_indices,
        )
        structure_loss: Float[torch.Tensor, "B"] = atom_loss(
            intermediate_denoised_coord,
            batch.r_gt,
            batch.atom5_mask,
            lambda_sigma_weight=lambda_sigma_weight,
        )

        k_loss: Float[torch.Tensor, "B"] = (
            lam * structure_loss + alpha_0 * seq_loss
        )
        intermediate_loss = intermediate_loss + gamma_K_minus_k * k_loss
        if torch.isnan(intermediate_loss).any():
            raise LossComputationError

    return (intermediate_loss / max(K, 1)).mean()


# ---------------------------------------------------------------------------
# Smooth lDDT loss
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def pairwise_dist(
    x: Float[torch.Tensor, "... N_atom 3"],
) -> Float[torch.Tensor, "... N_atom N_atom"]:
    """Compute pairwise Euclidean distances via einsum.

    Numerically stable: adds 1e-8 under the square root to avoid zero-gradient
    at coincident points.

    Args:
        x: Atom positions of shape ``(... N_atom 3)``.

    Returns:
        Pairwise distance matrix of shape ``(... N_atom N_atom)``.
    """
    diff: Float[torch.Tensor, "... N_atom N_atom 3"] = rearrange(
        x,
        "... n d -> ... n 1 d",
    ) - rearrange(x, "... n d -> ... 1 n d")
    sq: Float[torch.Tensor, "... N_atom N_atom"] = einsum(
        diff,
        diff,
        "... n m d, ... n m d -> ... n m",
    )
    return torch.sqrt(sq + 1e-8)


@jaxtyped(typechecker=beartype)
def smooth_lddt_loss(
    r_pred: Float[torch.Tensor, "... N_atom 3"],
    r_true: Float[torch.Tensor, "... N_atom 3"],
    mask: Bool[torch.Tensor, "... N_atom"] | None = None,
    cutoff: float = 15.0,
) -> Float[torch.Tensor, ""]:
    """Smooth lDDT loss — Algorithm 8 (from Pallatom).

    Exact algorithm:

    .. code-block:: none

        1. δr_lm     = ||r_l  - r_m||          (predicted pairwise distances)
        2. δr_lm_GT  = ||r_l_GT - r_m_GT||     (GT pairwise distances)
        3. δ_lm      = |δr_lm_GT - δr_lm|      (absolute distance difference)
        4. ε_lm      = ¼ [ sigmoid(½ - δ_lm)
                          + sigmoid(1  - δ_lm)
                          + sigmoid(2  - δ_lm)
                          + sigmoid(4  - δ_lm) ]     (smooth score ∈ (0,1))
        5. c_lm      = 1( δr_lm_GT < 15 Å )    (local-neighbourhood mask, l≠m)
           lddt      = mean_{l≠m}(c_lm · ε_lm)
                     / mean_{l≠m}(c_lm)
        return 1 - lddt

    Args:
        r_pred:  (..., N_atom, 3) — predicted atom coordinates  r̄_l.
        r_true:  (..., N_atom, 3) — ground-truth coordinates  r̄_l^GT.
        mask:    (..., N_atom)    — float/bool validity mask (optional).
                                    Pairs (l, m) are included only when both
                                    atoms are valid.
        cutoff: float — local neighbourhood radius in Å (default 15.0).

    Returns:
        Scalar loss = 1 - lddt_smooth  (averaged over batch).
    """
    dr_pred: Float[torch.Tensor, "... N_atom N_atom"] = pairwise_dist(r_pred)
    dr_true: Float[torch.Tensor, "... N_atom N_atom"] = pairwise_dist(r_true)

    # Absolute distance difference δ_lm
    delta: Float[torch.Tensor, "... N_atom N_atom"] = (dr_true - dr_pred).abs()

    # Smooth per-pair score ε_lm ∈ (0, 1)
    epsilon: Float[torch.Tensor, "... N_atom N_atom"] = 0.25 * (
        torch.sigmoid(0.5 - delta)
        + torch.sigmoid(1.0 - delta)
        + torch.sigmoid(2.0 - delta)
        + torch.sigmoid(4.0 - delta)
    )

    # Local-neighbourhood mask c_lm  (l ≠ m, d_GT < cutoff)
    N_atom = r_pred.shape[-2]
    not_diag: Bool[torch.Tensor, "N_atom N_atom"] = ~torch.eye(
        N_atom,
        device=r_pred.device,
        dtype=torch.bool,
    )
    c: Bool[torch.Tensor, "... N_atom N_atom"] = (dr_true < cutoff) & not_diag

    if mask is not None:
        m: Float[torch.Tensor, "... N_atom"] = mask.float()
        pair_valid: Bool[torch.Tensor, "... N_atom N_atom"] = einsum(
            m,
            m,
            "... n, ... m -> ... n m",
        ).bool()
        c = c & pair_valid

    c_f: Float[torch.Tensor, "... N_atom N_atom"] = c.float()

    # lddt = sum of (c·ε) / sum of (c)
    numer: Float[torch.Tensor, ...] = einsum(
        c_f,
        epsilon,
        "... n m, ... n m -> ...",
    )
    denom: Float[torch.Tensor, ...] = c_f.sum(dim=(-2, -1)).clamp(min=1)
    lddt: Float[torch.Tensor, ...] = numer / denom

    return (1.0 - lddt).mean()


# ---------------------------------------------------------------------------
# Distogram losses
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def distogram_loss_residue(
    p: Float[torch.Tensor, "... N_res N_res n_bins"],
    y: (
        Float[torch.Tensor, "... N_res N_res n_bins"]
        | Int[torch.Tensor, "... N_res N_res"]
    ),
    mask: Bool[torch.Tensor, "... N_res"] | None = None,
) -> Float[torch.Tensor, "..."]:
    """Residue-level distogram cross-entropy loss  (L_dist_res).

    Supervises predicted inter-residue distance bin probabilities against
    one-hot encoded ground-truth distance bins:

        L_dist_res = -1/N_res² · Σ_{i,j} Σ_{b=1}^{n_bins} y^b_{ij} · log
        p^b_{ij}

    Args:
        p:    (..., N_res, N_res, n_bins) — predicted bin logits.
        y:    (..., N_res, N_res, n_bins) — one-hot target bin assignments,
                                            OR (..., N_res, N_res) integer bin
                                            indices.
        mask: (..., N_res) — boolean/float residue mask (optional).

    Returns:
        (...,) scalar loss per batch element.

    Raises:
        LossComputationError: If the computed loss contains NaN values.
    """
    log_p: Float[torch.Tensor, "... N_res N_res n_bins"] = F.log_softmax(
        p,
        dim=-1,
    )

    if y.dim() == p.dim():
        # one-hot targets: CE = -Σ_b y^b · log p^b
        ce: Float[torch.Tensor, "... N_res N_res"] = -einsum(
            y.float(),
            log_p,
            "... l m b, ... l m b -> ... l m",
        )
    else:
        # integer class indices → gather along bin dim
        ce = -rearrange(
            log_p.gather(-1, rearrange(y.long(), "... l m -> ... l m 1")),
            "... l m 1 -> ... l m",
        )

    if mask is not None:
        m: Float[torch.Tensor, "... N_res"] = mask.float()
        pair_mask: Float[torch.Tensor, "... N_res N_res"] = einsum(
            m,
            m,
            "... l, ... m -> ... l m",
        )
        L_eff: Float[torch.Tensor, ...] = pair_mask.sum(dim=(-2, -1)).clamp(
            min=1,
        )
        loss: Float[torch.Tensor, ...] = (
            einsum(ce, pair_mask, "... l m, ... l m -> ...") / L_eff
        )
    else:
        L = p.shape[-2]
        loss = ce.sum(dim=(-2, -1)) / (L * L)

    if torch.isnan(loss).any():
        raise LossComputationError

    return loss


@jaxtyped(typechecker=beartype)
def distogram_loss_atom(
    q: Float[torch.Tensor, "... N_atom K n_bins"],
    y: (
        Float[torch.Tensor, "... N_atom K n_bins"]
        | Int[torch.Tensor, "... N_atom K"]
    ),
    local_mask: Bool[torch.Tensor, "... N_atom K"] | None = None,
) -> Float[torch.Tensor, "..."]:
    """Atomic-level local distogram cross-entropy loss  (L_dist_atom).

    Supervises predicted local inter-atom distance bin probabilities against
    one-hot encoded ground-truth bins within a local attention window:

        L_dist_atom = -1/(N_atom·K) · Σ_{n,k ∈ local} Σ_{b=1}^{n_bins} y^b_{nk}
        · log q^b_{nk}

    Args:
        q: (..., N_atom, K, n_bins) — predicted bin logits for K sparse
            neighbours.
        y:          (..., N_atom, K, n_bins) — one-hot targets,
                    OR (..., N_atom, K) integer bin indices.
        local_mask: (..., N_atom, K) — boolean validity mask (optional).

    Returns:
        (...,) scalar loss per batch element.

    Raises:
        LossComputationError: If the computed loss contains NaN values.
    """
    log_q: Float[torch.Tensor, "... N_atom K n_bins"] = F.log_softmax(q, dim=-1)

    if y.dim() == q.dim():
        # one-hot targets: CE = -Σ_b y^b · log q^b
        ce: Float[torch.Tensor, "... N_atom K"] = -einsum(
            y.float(),
            log_q,
            "... n m b, ... n m b -> ... n m",
        )
    else:
        # integer class indices → gather along bin dim
        ce = -rearrange(
            log_q.gather(-1, rearrange(y.long(), "... n m -> ... n m 1")),
            "... n m 1 -> ... n m",
        )

    if local_mask is not None:
        m: Float[torch.Tensor, "... N_atom K"] = local_mask.float()
        NM: Float[torch.Tensor, ...] = m.sum(dim=(-2, -1)).clamp(min=1)
        loss: Float[torch.Tensor, ...] = (
            einsum(ce, m, "... n m, ... n m -> ...") / NM
        )
    else:
        N_at = q.shape[-2]
        loss = ce.sum(dim=(-2, -1)) / (N_at * N_at)

    if torch.isnan(loss).any():
        raise LossComputationError

    return loss


# ---------------------------------------------------------------------------
# Sequence cross-entropy loss
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def seq_ce_loss(
    logits: Float[torch.Tensor, "B N_res n_amino"],
    aa_indices: Int[torch.Tensor, "B N_res"],
) -> Float[torch.Tensor, ""]:
    """CE loss for the sequence head; ignores padding and null/unknown tokens.

    Positions are ignored when aa_indices < 0 (padding) or aa_indices >=
    n_amino
    (PDB-X at index 20, conditioning-dropped at index 20). Only real visible
    amino
    acids (indices 0 to n_amino-1) contribute to the loss.

    Args:
        logits: Sequence logits from any head (final or intermediate).
        aa_indices: Per-residue amino acid indices, post-conditioning-dropout.

    Returns:
        Scalar mean CE loss over all valid positions.

    Raises:
        LossComputationError: If the computed loss contains NaN values.
    """
    n_amino: int = logits.size(-1)
    targets: Int[torch.Tensor, "B N_res"] = aa_indices.masked_fill(
        aa_indices >= n_amino,
        -100,
    )
    if (targets != PAD_TOKEN).sum() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    loss = F.cross_entropy(
        rearrange(logits, "b n c -> (b n) c"),
        rearrange(targets, "b n -> (b n)"),
        ignore_index=-100,
    )
    if torch.isnan(loss).any():
        raise LossComputationError
    return loss

"""Loss functions for diffusion model training."""

import torch
import torch.nn.functional as F
from beartype import beartype
from einops import einsum, rearrange
from helpers.alignment import kabsch_align
from jaxtyping import Bool, Float, Int, jaxtyped

# ---------------------------------------------------------------------------
# atom_loss  (Algorithm 2 structure term)
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def atom_loss(
    r_denoised: Float[torch.Tensor, "... N_res 3"],
    r_gt: Float[torch.Tensor, "... N_res 3"],
    mask: Bool[torch.Tensor, "... N_res"] | None = None,
) -> Float[torch.Tensor, "..."]:
    """Kabsch-aligned atom-coordinate MSE loss.

    Rigidly aligns the ground-truth structure onto the denoised structure
    (i.e. the *prediction* is held fixed; the GT is rotated/translated to
    match it), then computes the mean squared deviation per coordinate:

        L_atom = ||r_denoised - r_aligned||² / (3L)

    where L is the number of (unmasked) residues and the factor 3 accounts
    for the x, y, z dimensions, matching the formulation in the screenshot.

    Args:
        r_denoised: (..., N_res, 3) — model output / denoised coordinates.
        r_gt:       (..., N_res, 3) — ground-truth coordinates  r̄⁰.
        mask:       (..., N_res)    — boolean or float mask; 1 = valid residue.
                                      When None, all N_res residues are used.

    Returns:
        (...,) scalar loss per batch element.

    Notes:
        • Alignment is done with float weights derived from `mask`, so
          missing / padded residues do not pollute the rotation estimate.
        • Gradients flow through r_denoised (r_aligned is treated as a
          constant frame — the GT is being moved, not the prediction).
    """
    weights: Float[torch.Tensor, "... N_res"] | None = mask.float() if mask is not None else None

    # Align GT → denoised  (mobile=r_gt, target=r_denoised).
    # Detach so gradients flow only through r_denoised, not through the SVD.
    (r_aligned,) = kabsch_align(r_gt, r_denoised, weights=weights, return_transform=False)
    r_aligned: Float[torch.Tensor, "... N_res 3"] = r_aligned.detach()

    # Squared residuals summed over xyz → (..., N_res)
    diff: Float[torch.Tensor, "... N_res 3"] = r_denoised - r_aligned
    sq: Float[torch.Tensor, "... N_res"] = einsum(diff, diff, "... l d, ... l d -> ... l")

    if mask is not None:
        m: Float[torch.Tensor, "... N_res"] = mask.float()
        L_eff: Float[torch.Tensor, ...] = m.sum(dim=-1).clamp(min=1)
        loss: Float[torch.Tensor, ...] = einsum(sq, m, "... l, ... l -> ...") / (3.0 * L_eff)
    else:
        L = r_denoised.shape[-2]
        loss = sq.sum(dim=-1) / (3.0 * L)

    return loss


# # ---- 5. atom_loss (pseudo-test — see tests/architecture/test_losses.py) ----
# r_denoised = torch.randn(B, N, 3)
# loss_perfect = atom_loss(r_denoised, r_denoised).mean().item()   # expect ~0
# r_gt_noisy = r_denoised + 0.5 * torch.randn(B, N, 3)
# loss_noisy = atom_loss(r_denoised, r_gt_noisy).mean().item()
# mask = torch.ones(B, N, dtype=torch.bool); mask[:, 40:] = False
# loss_masked = atom_loss(r_denoised, r_gt_noisy, mask=mask).mean().item()
# r_pred = torch.randn(N, 3, requires_grad=True)
# atom_loss(r_pred, torch.randn(N, 3)).backward()
# assert r_pred.grad is not None


# ---------------------------------------------------------------------------
# Intermediate supervision loss with decoder-block weight decay  (L_med)
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def med_loss_per_block(
    r_denoised_k: Float[torch.Tensor, "... N_res 3"],
    r_gt: Float[torch.Tensor, "... N_res 3"],
    logits_aa_k: Float[torch.Tensor, "... N_res n_amino"],
    aa_gt: Int[torch.Tensor, "... N_res"],
    lam: float,
    alpha_0: float,
    mask: Bool[torch.Tensor, "... N_res"] | None = None,
) -> Float[torch.Tensor, "..."]:
    """Per-decoder-block intermediate loss  L^k_med.

        L^k_med = λ(t) · ||r̄^denoised_(k) - r̄^aligned||² / 3L
                + α₀ · CE(â_(k), a⁰)

    The structure term reuses `atom_loss` (Kabsch-aligned MSE).
    The sequence term is standard cross-entropy over amino-acid logits.

    Args:
        r_denoised_k: (..., N_res, 3)       — denoised coordinates from block k.
        r_gt:         (..., N_res, 3)       — ground-truth coordinates  r̄⁰.
        logits_aa_k:  (..., N_res, n_amino) — amino-acid logits â_(k).
        aa_gt:        (..., N_res)          — ground-truth amino-acid class indices a⁰.
        lam:          float                 — λ(t), noise-level weight for the structure term.
        alpha_0:      float                 — α₀, fixed weight for the sequence CE term.
        mask:         (..., N_res)          — boolean/float residue mask (optional).

    Returns:
        (...,) per-batch loss for block k.
    """
    # Structure term: Kabsch-aligned MSE
    struct: Float[torch.Tensor, ...] = atom_loss(r_denoised_k, r_gt, mask=mask)

    # Sequence term: cross-entropy via log-softmax + einsum over vocab
    n_amino = logits_aa_k.shape[-1]
    log_p: Float[torch.Tensor, "... N_res n_amino"] = F.log_softmax(logits_aa_k, dim=-1)
    aa_gt_oh: Float[torch.Tensor, "... N_res n_amino"] = F.one_hot(
        aa_gt.long(), num_classes=n_amino
    ).float()
    ce_per_res: Float[torch.Tensor, "... N_res"] = -einsum(
        aa_gt_oh, log_p, "... l c, ... l c -> ... l"
    )

    if mask is not None:
        m: Float[torch.Tensor, "... N_res"] = mask.float()
        L_eff: Float[torch.Tensor, ...] = m.sum(dim=-1).clamp(min=1)
        seq: Float[torch.Tensor, ...] = einsum(ce_per_res, m, "... l, ... l -> ...") / L_eff
    else:
        seq = ce_per_res.mean(dim=-1)

    return lam * struct + alpha_0 * seq


def med_loss(
    r_denoised_blocks: list[Float[torch.Tensor, "... N_res 3"]],
    r_gt: Float[torch.Tensor, "... N_res 3"],
    logits_aa_blocks: list[Float[torch.Tensor, "... N_res n_amino"]],
    aa_gt: Int[torch.Tensor, "... N_res"],
    lam: float,
    alpha_0: float,
    gamma: float = 0.99,
    mask: Bool[torch.Tensor, "... N_res"] | None = None,
) -> Float[torch.Tensor, ""]:
    """Intermediate loss over K decoder blocks with exp weight decay to upweight later blocks.

        L_med = (1/K) · Σ_{k=1}^{K} gamma^(K-k) · L^k_med

    gamma < 1 means early blocks get lower weight (gamma^(K-1), gamma^(K-2), …, gamma^0 = 1),
    so the final block always receives weight 1 and earlier blocks are
    progressively discounted.

    Args:
        r_denoised_blocks: list of K tensors, each (..., N_res, 3).
        r_gt:              (..., N_res, 3)  — ground-truth coordinates r̄⁰.
        logits_aa_blocks:  list of K tensors, each (..., N_res, n_amino).
        aa_gt:             (..., N_res)    — ground-truth amino-acid indices a⁰.
        lam:               float           — λ(t), structure loss weight.
        alpha_0:           float           — α₀, sequence loss weight.
        gamma:             float           — decay factor gamma (default 0.99).
        mask:              (..., N_res)    — optional residue mask.

    Returns:
        Scalar (mean over batch) total intermediate loss L_med.

    Raises:
        ValueError: if the two block lists have different lengths.
    """
    K = len(r_denoised_blocks)
    if K == 0:
        raise ValueError("r_denoised_blocks must be non-empty.")
    if len(logits_aa_blocks) != K:
        raise ValueError(
            f"r_denoised_blocks has {K} entries but "
            f"logits_aa_blocks has {len(logits_aa_blocks)}."
        )

    total: Float[torch.Tensor, ""] | None = None
    for k_idx, (r_k, logits_k) in enumerate(zip(r_denoised_blocks, logits_aa_blocks, strict=True)):
        k = k_idx + 1  # 1-indexed block number
        w = gamma ** (K - k)  # gamma^(K-k)
        lk: Float[torch.Tensor, ...] = med_loss_per_block(
            r_k, r_gt, logits_k, aa_gt, lam, alpha_0, mask=mask
        )
        total = w * lk if total is None else total + w * lk

    assert total is not None
    return total.mean() / K  # scalar


# ---------------------------------------------------------------------------
# Smooth lDDT loss
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def pairwise_dist(
    x: Float[torch.Tensor, "... N_atom 3"],
) -> Float[torch.Tensor, "... N_atom N_atom"]:
    """||x_i - x_j|| for all pairs via einsum — numerically stable."""
    diff: Float[torch.Tensor, "... N_atom N_atom 3"] = rearrange(
        x, "... n d -> ... n 1 d"
    ) - rearrange(x, "... n d -> ... 1 n d")
    sq: Float[torch.Tensor, "... N_atom N_atom"] = einsum(
        diff, diff, "... n m d, ... n m d -> ... n m"
    )
    return torch.sqrt(sq + 1e-8)


@jaxtyped(typechecker=beartype)
def smooth_lddt_loss(
    r_pred: Float[torch.Tensor, "... N_atom 3"],
    r_true: Float[torch.Tensor, "... N_atom 3"],
    mask: Bool[torch.Tensor, "... N_atom"] | None = None,
    cutoff: float = 15.0,
) -> Float[torch.Tensor, ""]:
    """Smooth lDDT loss — Algorithm 8 (simplified AF3 version for all-atom design).

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
        cutoff:  float            — local neighbourhood radius in Å (default 15.0).

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
        N_atom, device=r_pred.device, dtype=torch.bool
    )
    c: Bool[torch.Tensor, "... N_atom N_atom"] = (dr_true < cutoff) & not_diag

    if mask is not None:
        m: Float[torch.Tensor, "... N_atom"] = mask.float()
        pair_valid: Bool[torch.Tensor, "... N_atom N_atom"] = einsum(
            m, m, "... n, ... m -> ... n m"
        ).bool()
        c = c & pair_valid

    c_f: Float[torch.Tensor, "... N_atom N_atom"] = c.float()

    # lddt = Σ(c·ε) / Σ(c)
    numer: Float[torch.Tensor, ...] = einsum(c_f, epsilon, "... n m, ... n m -> ...")
    denom: Float[torch.Tensor, ...] = c_f.sum(dim=(-2, -1)).clamp(min=1)
    lddt: Float[torch.Tensor, ...] = numer / denom

    return (1.0 - lddt).mean()


# ---------------------------------------------------------------------------
# Distogram losses
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def distogram_loss_residue(
    p: Float[torch.Tensor, "... N_res N_res n_bins"],
    y: Float[torch.Tensor, "... N_res N_res n_bins"] | Int[torch.Tensor, "... N_res N_res"],
    mask: Bool[torch.Tensor, "... N_res"] | None = None,
) -> Float[torch.Tensor, "..."]:
    """Residue-level distogram cross-entropy loss  (L_dist_res).

    Supervises predicted inter-residue distance bin probabilities against
    one-hot encoded ground-truth distance bins:

        L_dist_res = -1/N_res² · Σ_{i,j} Σ_{b=1}^{n_bins} y^b_{ij} · log p^b_{ij}

    Args:
        p:    (..., N_res, N_res, n_bins) — predicted bin logits.
        y:    (..., N_res, N_res, n_bins) — one-hot target bin assignments,
                                            OR (..., N_res, N_res) integer bin indices.
        mask: (..., N_res)                — boolean/float residue mask (optional).

    Returns:
        (...,) scalar loss per batch element.
    """
    log_p: Float[torch.Tensor, "... N_res N_res n_bins"] = F.log_softmax(p, dim=-1)

    if y.dim() == p.dim():
        # one-hot targets: CE = -Σ_b y^b · log p^b
        ce: Float[torch.Tensor, "... N_res N_res"] = -einsum(
            y.float(), log_p, "... l m b, ... l m b -> ... l m"
        )
    else:
        # integer class indices → gather along bin dim
        ce = -rearrange(
            log_p.gather(-1, rearrange(y.long(), "... l m -> ... l m 1")),
            "... l m 1 -> ... l m",
        )

    if mask is not None:
        m: Float[torch.Tensor, "... N_res"] = mask.float()
        pair_mask: Float[torch.Tensor, "... N_res N_res"] = einsum(m, m, "... l, ... m -> ... l m")
        L_eff: Float[torch.Tensor, ...] = pair_mask.sum(dim=(-2, -1)).clamp(min=1)
        loss: Float[torch.Tensor, ...] = einsum(ce, pair_mask, "... l m, ... l m -> ...") / L_eff
    else:
        L = p.shape[-2]
        loss = ce.sum(dim=(-2, -1)) / (L * L)

    return loss


@jaxtyped(typechecker=beartype)
def distogram_loss_atom(
    q: Float[torch.Tensor, "... N_atom K n_bins"],
    y: Float[torch.Tensor, "... N_atom K n_bins"] | Int[torch.Tensor, "... N_atom K"],
    local_mask: Bool[torch.Tensor, "... N_atom K"] | None = None,
) -> Float[torch.Tensor, "..."]:
    """Atomic-level local distogram cross-entropy loss  (L_dist_atom).

    Supervises predicted local inter-atom distance bin probabilities against
    one-hot encoded ground-truth bins within a local attention window:

        L_dist_atom = -1/(N_atom·K) · Σ_{n,k ∈ local} Σ_{b=1}^{n_bins} y^b_{nk} · log q^b_{nk}

    Args:
        q:          (..., N_atom, K, n_bins) — predicted bin logits for K sparse neighbours.
        y:          (..., N_atom, K, n_bins) — one-hot targets,
                    OR (..., N_atom, K) integer bin indices.
        local_mask: (..., N_atom, K)         — boolean validity mask (optional).

    Returns:
        (...,) scalar loss per batch element.
    """
    log_q: Float[torch.Tensor, "... N_atom K n_bins"] = F.log_softmax(q, dim=-1)

    if y.dim() == q.dim():
        # one-hot targets: CE = -Σ_b y^b · log q^b
        ce: Float[torch.Tensor, "... N_atom K"] = -einsum(
            y.float(), log_q, "... n m b, ... n m b -> ... n m"
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
        loss: Float[torch.Tensor, ...] = einsum(ce, m, "... n m, ... n m -> ...") / NM
    else:
        N_at = q.shape[-2]
        loss = ce.sum(dim=(-2, -1)) / (N_at * N_at)

    return loss


# ---- 6 & 7. distogram losses (pseudo-test — see tests/architecture/test_losses.py) ----
# L, B_res = 32, 64
# logits_res = torch.randn(B, L, L, B_res)
# bin_idx = torch.randint(0, B_res, (B, L, L))
# y_onehot = F.one_hot(bin_idx, num_classes=B_res).float()
# loss_res_onehot = distogram_loss_residue(logits_res, y_onehot).mean().item()
# loss_res_idx    = distogram_loss_residue(logits_res, bin_idx).mean().item()
# assert abs(loss_res_onehot - loss_res_idx) < 1e-4
# perfect_logits = y_onehot * 1e6
# loss_res_perfect = distogram_loss_residue(perfect_logits, y_onehot).mean().item()
# L_small, A, N_atoms, B_atom = 8, 14, 112, 22
# logits_atom = torch.randn(B, N_atoms, N_atoms, B_atom)
# y_atom = F.one_hot(torch.randint(0, B_atom, (B, N_atoms, N_atoms)), num_classes=B_atom).float()
# loss_atom_local = distogram_loss_atom(logits_atom, y_atom, local_mask).mean().item()
# loss_atom_perfect = distogram_loss_atom(y_atom * 1e6, y_atom, local_mask).mean().item()


# ---------------------------------------------------------------------------
# Sequence cross-entropy loss
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def seq_ce_loss(
    logits: Float[torch.Tensor, "B N_res n_amino"],
    aa_indices: Int[torch.Tensor, "B N_res"],
) -> Float[torch.Tensor, ""]:
    """CE loss for the sequence head; ignores padding and null/unknown tokens.

    Positions are ignored when aa_indices < 0 (padding) or aa_indices >= n_amino
    (PDB-X at index 20, conditioning-dropped at index 20). Only real visible amino
    acids (indices 0 to n_amino-1) contribute to the loss.

    Args:
        logits: Sequence logits from any head (final or intermediate).
        aa_indices: Per-residue amino acid indices, post-conditioning-dropout.

    Returns:
        Scalar mean CE loss over all valid positions.
    """
    n_amino: int = logits.size(-1)
    targets: Int[torch.Tensor, "B N_res"] = aa_indices.masked_fill(aa_indices >= n_amino, -100)
    if (targets != -100).sum() == 0:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(
        rearrange(logits, "b n c -> (b n) c"),
        rearrange(targets, "b n -> (b n)"),
        ignore_index=-100,
    )

# Kabsch algorithm util

"""
kabsch.py — Rigid protein structure alignment via the Kabsch Algorithm
Pure PyTorch implementation. Supports batched inputs and optional masking.

References:
    Kabsch, W. (1976). A solution for the best rotation to relate two sets
    of vectors. Acta Crystallographica, A32, 922-923.
    https://doi.org/10.1107/S0567739476001873
"""

import torch
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Bool, Float, jaxtyped
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

@jaxtyped(typechecker=beartype)
def kabsch_rotation(
    P: Float[torch.Tensor, "... N 3"],
    Q: Float[torch.Tensor, "... N 3"],
    weights: Optional[Float[torch.Tensor, "... N"]] = None,
) -> Float[torch.Tensor, "... 3 3"]:
    """
    Compute the optimal rotation matrix R that minimises RMSD between P and Q
    after centering, using singular value decomposition (SVD).

    The algorithm finds R such that:
        ||W^(1/2) (P @ R.T - Q)||_F  is minimised

    Args:
        P:        (..., N, 3) — mobile structure (will be rotated).
        Q:        (..., N, 3) — target/reference structure.
        weights:  (..., N)    — per-residue weights (optional, non-negative).
                               Useful for masking missing residues or
                               down-weighting flexible loops.

    Returns:
        R:  (..., 3, 3) — rotation matrix (det = +1, i.e. proper rotation).

    Notes:
        • Inputs must already be **centred** (mean-subtracted) when this
          function is called directly.  Use `kabsch_align` for the full
          pipeline (centre → rotate → translate).
        • Handles reflection by flipping the sign of the column corresponding
          to the smallest singular value when det(V @ U^T) < 0.
    """
    if weights is not None:
        w = weights / (reduce(weights, "... n -> ... 1", "sum") + 1e-8)
        H = torch.einsum("...ni,...nj->...ij", P * rearrange(w, "... n -> ... n 1"), Q)
    else:
        H = torch.einsum("...ni,...nj->...ij", P, Q)

    # SVD:  H = U S V^T
    U, S, Vh = torch.linalg.svd(H)          # (...,3,3), (...,3), (...,3,3)
    V = Vh.mH                                # V = Vh^H (conjugate transpose)

    # Correct for improper rotation (reflection) when det < 0
    d = torch.linalg.det(torch.einsum("...ij,...jk->...ik", V, U.mH))
    sign = torch.ones_like(d)
    sign[d < 0] = -1.0

    # Build diagonal correction matrix: diag(1, 1, sign)
    ones = torch.ones(*sign.shape, 2, device=P.device, dtype=P.dtype)
    diag_vals = torch.cat([ones, sign.unsqueeze(-1)], dim=-1)  # (..., 3)
    D = torch.diag_embed(diag_vals)                             # (..., 3, 3)

    VD = torch.einsum("...ij,...jk->...ik", V, D)
    return torch.einsum("...ij,...jk->...ik", VD, U.mH)        # (..., 3, 3)


@jaxtyped(typechecker=beartype)
def kabsch_align(
    mobile: Float[torch.Tensor, "... N 3"],
    target: Float[torch.Tensor, "... N 3"],
    weights: Optional[Float[torch.Tensor, "... N"]] = None,
    return_transform: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """
    Rigidly align `mobile` onto `target` using the Kabsch algorithm.

    Pipeline:
        1. Compute (weighted) centroids of both structures.
        2. Centre both structures.
        3. Compute optimal rotation R via SVD (Kabsch).
        4. Apply: aligned = (mobile − c_mobile) @ R^T + c_target

    Args:
        mobile:           (..., N, 3)  — coordinates to be aligned.
        target:           (..., N, 3)  — reference coordinates.
        weights:          (..., N)     — optional per-residue weights
                                        (e.g. 1 for structured, 0 for missing).
        return_transform: bool         — if True, also return (R, t_mobile, t_target).

    Returns:
        aligned:    (..., N, 3)  — mobile after optimal rigid alignment to target.
        R:          (..., 3, 3)  — rotation matrix  [only if return_transform]
        t_mobile:   (..., 1, 3)  — centroid of mobile  [only if return_transform]
        t_target:   (..., 1, 3)  — centroid of target  [only if return_transform]
    """
    # ---- 1. Centroids -------------------------------------------------
    if weights is not None:
        w = weights / (reduce(weights, "... n -> ... 1", "sum") + 1e-8)
        w3 = rearrange(w, "... n -> ... n 1")
        c_mobile = reduce(mobile * w3, "... n d -> ... 1 d", "sum")
        c_target = reduce(target * w3, "... n d -> ... 1 d", "sum")
    else:
        c_mobile = rearrange(reduce(mobile, "... n d -> ... d", "mean"), "... d -> ... 1 d")
        c_target = rearrange(reduce(target, "... n d -> ... d", "mean"), "... d -> ... 1 d")

    # ---- 2. Centre ----------------------------------------------------
    P = mobile - c_mobile   # (..., N, 3)
    Q = target - c_target

    # ---- 3. Optimal rotation ------------------------------------------
    R = kabsch_rotation(P, Q, weights=weights)   # (..., 3, 3)

    # ---- 4. Apply transform -------------------------------------------
    # aligned = P @ R^T + c_target
    aligned = torch.einsum("...ni,...ij->...nj", P, R.mH) + c_target

    if return_transform:
        return aligned, R, c_mobile, c_target
    return (aligned,)


# ---------------------------------------------------------------------------
# RMSD helpers
# ---------------------------------------------------------------------------

@jaxtyped(typechecker=beartype)
def rmsd(
    P: Float[torch.Tensor, "... N 3"],
    Q: Float[torch.Tensor, "... N 3"],
    weights: Optional[Float[torch.Tensor, "... N"]] = None,
    mask: Optional[Bool[torch.Tensor, "... N"]] = None,
) -> torch.Tensor:
    """
    Root-mean-square deviation between two (already aligned) coordinate sets.

    Args:
        P:       (..., N, 3) — predicted / mobile coordinates.
        Q:       (..., N, 3) — reference coordinates.
        weights: (..., N)    — per-residue weights (optional).
        mask:    (..., N)    — boolean mask; True = include residue (optional).
                               Applied on top of weights.

    Returns:
        (...,) tensor of RMSD values in the same units as the input coordinates
        (typically Ångströms for protein structures).
    """
    if mask is not None:
        m = rearrange(mask.float(), "... n -> ... n 1")
        P = P * m
        Q = Q * m
        if weights is not None:
            weights = weights * mask.float()

    sq_dev = reduce((P - Q) ** 2, "... n d -> ... n", "sum")   # (..., N)

    if weights is not None:
        w = weights / (reduce(weights, "... n -> ... 1", "sum") + 1e-8)
        msd = reduce(sq_dev * w, "... n -> ...", "sum")
    else:
        n = sq_dev.shape[-1] if mask is None else reduce(mask.float(), "... n -> ...", "sum").clamp(min=1)
        msd = reduce(sq_dev, "... n -> ...", "sum") / n

    return torch.sqrt(msd + 1e-8)


@jaxtyped(typechecker=beartype)
def kabsch_rmsd(
    mobile: Float[torch.Tensor, "... N 3"],
    target: Float[torch.Tensor, "... N 3"],
    weights: Optional[Float[torch.Tensor, "... N"]] = None,
    mask: Optional[Bool[torch.Tensor, "... N"]] = None,
) -> torch.Tensor:
    """
    Align `mobile` to `target`, then return the RMSD.

    Args:
        mobile:  (..., N, 3) — coordinates to align.
        target:  (..., N, 3) — reference coordinates.
        weights: (..., N)    — per-residue weights (optional).
        mask:    (..., N)    — boolean residue mask (optional).

    Returns:
        (...,) RMSD after optimal rigid alignment (Ångströms).
    """
    eff_weights = weights
    if mask is not None:
        eff_weights = (weights * mask.float()) if weights is not None else mask.float()

    (aligned,) = kabsch_align(mobile, target, weights=eff_weights)
    return rmsd(aligned, target, weights=eff_weights, mask=mask)


# ---------------------------------------------------------------------------
# Convenience: apply a stored rigid transform to new coordinates
# ---------------------------------------------------------------------------

@jaxtyped(typechecker=beartype)
def apply_transform(
    coords: Float[torch.Tensor, "... M 3"],
    R: Float[torch.Tensor, "... 3 3"],
    t_from: Float[torch.Tensor, "... 1 3"],
    t_to: Float[torch.Tensor, "... 1 3"],
) -> Float[torch.Tensor, "... M 3"]:
    """
    Apply a previously computed Kabsch rigid transform to a new set of
    coordinates (e.g. all atoms after fitting on Cα only).

    Transform:  coords_aligned = (coords − t_from) @ R^T + t_to

    Args:
        coords:  (..., M, 3) — coordinates to transform (can differ from
                               the N atoms used to compute the transform).
        R:       (..., 3, 3) — rotation matrix from `kabsch_align`.
        t_from:  (..., 1, 3) — centroid of the mobile set (c_mobile).
        t_to:    (..., 1, 3) — centroid of the target set (c_target).

    Returns:
        (..., M, 3) — transformed coordinates.
    """
    return torch.einsum("...mi,...ij->...mj", coords - t_from, R.mH) + t_to

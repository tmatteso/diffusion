pallatom.helpers.alignment
==========================

.. py:module:: pallatom.helpers.alignment

.. autoapi-nested-parse::

   kabsch.py — Rigid protein structure alignment via the Kabsch Algorithm
   Pure PyTorch implementation. Supports batched inputs and optional masking.

   .. rubric:: References

   Kabsch, W. (1976). A solution for the best rotation to relate two sets
   of vectors. Acta Crystallographica, A32, 922-923.
   https://doi.org/10.1107/S0567739476001873



Functions
---------

.. autoapisummary::

   pallatom.helpers.alignment.apply_transform
   pallatom.helpers.alignment.kabsch_align
   pallatom.helpers.alignment.kabsch_rmsd
   pallatom.helpers.alignment.kabsch_rotation
   pallatom.helpers.alignment.rmsd


Module Contents
---------------

.. py:function:: apply_transform(coords: jaxtyping.Float[torch.Tensor, ... M 3], R: jaxtyping.Float[torch.Tensor, ... 3 3], t_from: jaxtyping.Float[torch.Tensor, ... 1 3], t_to: jaxtyping.Float[torch.Tensor, ... 1 3]) -> jaxtyping.Float[torch.Tensor, ... M 3]

   Apply a previously computed Kabsch rigid transform to a new set of
   coordinates (e.g. all atoms after fitting on Cα only).

   Transform:  coords_aligned = (coords − t_from) @ R^T + t_to

   :param coords: (..., M, 3) — coordinates to transform (can differ from
                  the N atoms used to compute the transform).
   :param R: (..., 3, 3) — rotation matrix from `kabsch_align`.
   :param t_from: (..., 1, 3) — centroid of the mobile set (c_mobile).
   :param t_to: (..., 1, 3) — centroid of the target set (c_target).

   :returns: (..., M, 3) — transformed coordinates.


.. py:function:: kabsch_align(mobile: jaxtyping.Float[torch.Tensor, ... N 3], target: jaxtyping.Float[torch.Tensor, ... N 3], weights: Optional[jaxtyping.Float[torch.Tensor, ... N]] = None, return_transform: bool = False) -> Tuple[torch.Tensor, Ellipsis]

   Rigidly align `mobile` onto `target` using the Kabsch algorithm.

   Pipeline:
       1. Compute (weighted) centroids of both structures.
       2. Centre both structures.
       3. Compute optimal rotation R via SVD (Kabsch).
       4. Apply: aligned = (mobile − c_mobile) @ R^T + c_target

   :param mobile: (..., N, 3)  — coordinates to be aligned.
   :param target: (..., N, 3)  — reference coordinates.
   :param weights: (..., N)     — optional per-residue weights
                   (e.g. 1 for structured, 0 for missing).
   :param return_transform: bool         — if True, also return (R, t_mobile, t_target).

   :returns: (..., N, 3)  — mobile after optimal rigid alignment to target.
             R:          (..., 3, 3)  — rotation matrix  [only if return_transform]
             t_mobile:   (..., 1, 3)  — centroid of mobile  [only if return_transform]
             t_target:   (..., 1, 3)  — centroid of target  [only if return_transform]
   :rtype: aligned


.. py:function:: kabsch_rmsd(mobile: jaxtyping.Float[torch.Tensor, ... N 3], target: jaxtyping.Float[torch.Tensor, ... N 3], weights: Optional[jaxtyping.Float[torch.Tensor, ... N]] = None, mask: Optional[jaxtyping.Bool[torch.Tensor, ... N]] = None) -> torch.Tensor

   Align `mobile` to `target`, then return the RMSD.

   :param mobile: (..., N, 3) — coordinates to align.
   :param target: (..., N, 3) — reference coordinates.
   :param weights: (..., N)    — per-residue weights (optional).
   :param mask: (..., N)    — boolean residue mask (optional).

   :returns: (...,) RMSD after optimal rigid alignment (Ångströms).


.. py:function:: kabsch_rotation(P: jaxtyping.Float[torch.Tensor, ... N 3], Q: jaxtyping.Float[torch.Tensor, ... N 3], weights: Optional[jaxtyping.Float[torch.Tensor, ... N]] = None) -> jaxtyping.Float[torch.Tensor, ... 3 3]

   Compute the optimal rotation matrix R that minimises RMSD between P and Q
   after centering, using singular value decomposition (SVD).

   The algorithm finds R such that:
       ||W^(1/2) (P @ R.T - Q)||_F  is minimised

   :param P: (..., N, 3) — mobile structure (will be rotated).
   :param Q: (..., N, 3) — target/reference structure.
   :param weights: (..., N)    — per-residue weights (optional, non-negative).
                   Useful for masking missing residues or
                   down-weighting flexible loops.

   :returns: (..., 3, 3) — rotation matrix (det = +1, i.e. proper rotation).
   :rtype: R

   .. rubric:: Notes

   • Inputs must already be **centred** (mean-subtracted) when this
     function is called directly.  Use `kabsch_align` for the full
     pipeline (centre → rotate → translate).
   • Handles reflection by flipping the sign of the column corresponding
     to the smallest singular value when det(V @ U^T) < 0.


.. py:function:: rmsd(P: jaxtyping.Float[torch.Tensor, ... N 3], Q: jaxtyping.Float[torch.Tensor, ... N 3], weights: Optional[jaxtyping.Float[torch.Tensor, ... N]] = None, mask: Optional[jaxtyping.Bool[torch.Tensor, ... N]] = None) -> torch.Tensor

   Root-mean-square deviation between two (already aligned) coordinate sets.

   :param P: (..., N, 3) — predicted / mobile coordinates.
   :param Q: (..., N, 3) — reference coordinates.
   :param weights: (..., N)    — per-residue weights (optional).
   :param mask: (..., N)    — boolean mask; True = include residue (optional).
                Applied on top of weights.

   :returns: (...,) tensor of RMSD values in the same units as the input coordinates
             (typically Ångströms for protein structures).



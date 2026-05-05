pallatom.architecture.losses
============================

.. py:module:: pallatom.architecture.losses


Functions
---------

.. autoapisummary::

   pallatom.architecture.losses.atom_loss
   pallatom.architecture.losses.distogram_loss_atom
   pallatom.architecture.losses.distogram_loss_residue
   pallatom.architecture.losses.med_loss
   pallatom.architecture.losses.med_loss_per_block
   pallatom.architecture.losses.smooth_lddt_loss


Module Contents
---------------

.. py:function:: atom_loss(r_denoised: jaxtyping.Float[torch.Tensor, ... N_res 3], r_gt: jaxtyping.Float[torch.Tensor, ... N_res 3], mask: Optional[jaxtyping.Bool[torch.Tensor, ... N_res]] = None) -> jaxtyping.Float[torch.Tensor, ...]

   Kabsch-aligned atom-coordinate MSE loss.

   Rigidly aligns the ground-truth structure onto the denoised structure
   (i.e. the *prediction* is held fixed; the GT is rotated/translated to
   match it), then computes the mean squared deviation per coordinate:

       L_atom = ||r_denoised − r_aligned||² / (3L)

   where L is the number of (unmasked) residues and the factor 3 accounts
   for the x, y, z dimensions, matching the formulation in the screenshot.

   :param r_denoised: (..., N_res, 3) — model output / denoised coordinates.
   :param r_gt: (..., N_res, 3) — ground-truth coordinates  r̄⁰.
   :param mask: (..., N_res)    — boolean or float mask; 1 = valid residue.
                When None, all N_res residues are used.

   :returns: (...,) scalar loss per batch element.

   .. rubric:: Notes

   • Alignment is done with float weights derived from `mask`, so
     missing / padded residues do not pollute the rotation estimate.
   • Gradients flow through r_denoised (r_aligned is treated as a
     constant frame — the GT is being moved, not the prediction).


.. py:function:: distogram_loss_atom(q: jaxtyping.Float[torch.Tensor, ... N_atom K n_bins], y: torch.Tensor, local_mask: Optional[jaxtyping.Bool[torch.Tensor, ... N_atom K]] = None) -> jaxtyping.Float[torch.Tensor, ...]

   Atomic-level local distogram cross-entropy loss  (L_dist_atom).

   Supervises predicted local inter-atom distance bin probabilities against
   one-hot encoded ground-truth bins within a local attention window:

       L_dist_atom = -1/(N_atom·K) · Σ_{n,k ∈ local} Σ_{b=1}^{n_bins} y^b_{nk} · log q^b_{nk}

   :param q: (..., N_atom, K, n_bins) — predicted bin logits for K sparse neighbours.
   :param y: (..., N_atom, K, n_bins) — one-hot targets,
             OR (..., N_atom, K) integer bin indices.
   :param local_mask: (..., N_atom, K)         — boolean validity mask (optional).

   :returns: (...,) scalar loss per batch element.


.. py:function:: distogram_loss_residue(p: jaxtyping.Float[torch.Tensor, ... N_res N_res n_bins], y: torch.Tensor, mask: Optional[jaxtyping.Bool[torch.Tensor, ... N_res]] = None) -> jaxtyping.Float[torch.Tensor, ...]

   Residue-level distogram cross-entropy loss  (L_dist_res).

   Supervises predicted inter-residue distance bin probabilities against
   one-hot encoded ground-truth distance bins:

       L_dist_res = -1/N_res² · Σ_{i,j} Σ_{b=1}^{n_bins} y^b_{ij} · log p^b_{ij}

   :param p: (..., N_res, N_res, n_bins) — predicted bin logits.
   :param y: (..., N_res, N_res, n_bins) — one-hot target bin assignments,
             OR (..., N_res, N_res) integer bin indices.
   :param mask: (..., N_res)                — boolean/float residue mask (optional).

   :returns: (...,) scalar loss per batch element.


.. py:function:: med_loss(r_denoised_blocks: list, r_gt: jaxtyping.Float[torch.Tensor, ... N_res 3], logits_aa_blocks: list, aa_gt: jaxtyping.Int[torch.Tensor, ... N_res], lam: float, alpha_0: float, gamma: float = 0.99, mask: Optional[jaxtyping.Bool[torch.Tensor, ... N_res]] = None) -> jaxtyping.Float[torch.Tensor, ]

   Full intermediate supervision loss averaged over K decoder blocks with
   exponential weight decay that up-weights later blocks:

       L_med = (1/K) · Σ_{k=1}^{K} γ^(K−k) · L^k_med

   γ < 1 means early blocks get lower weight (γ^(K-1), γ^(K-2), …, γ^0 = 1),
   so the final block always receives weight 1 and earlier blocks are
   progressively discounted.

   :param r_denoised_blocks: list of K tensors, each (..., N_res, 3).
   :param r_gt: (..., N_res, 3)  — ground-truth coordinates r̄⁰.
   :param logits_aa_blocks: list of K tensors, each (..., N_res, n_amino).
   :param aa_gt: (..., N_res)    — ground-truth amino-acid indices a⁰.
   :param lam: float           — λ(t), structure loss weight.
   :param alpha_0: float           — α₀, sequence loss weight.
   :param gamma: float           — decay factor γ (default 0.99).
   :param mask: (..., N_res)    — optional residue mask.

   :returns: Scalar (mean over batch) total intermediate loss L_med.

   :raises ValueError: if the two block lists have different lengths.


.. py:function:: med_loss_per_block(r_denoised_k: jaxtyping.Float[torch.Tensor, ... N_res 3], r_gt: jaxtyping.Float[torch.Tensor, ... N_res 3], logits_aa_k: jaxtyping.Float[torch.Tensor, ... N_res n_amino], aa_gt: jaxtyping.Int[torch.Tensor, ... N_res], lam: float, alpha_0: float, mask: Optional[jaxtyping.Bool[torch.Tensor, ... N_res]] = None) -> jaxtyping.Float[torch.Tensor, ...]

   Per-decoder-block intermediate loss  L^k_med.

       L^k_med = λ(t) · ||r̄^denoised_(k) − r̄^aligned||² / 3L
               + α₀ · CE(â_(k), a⁰)

   The structure term reuses `atom_loss` (Kabsch-aligned MSE).
   The sequence term is standard cross-entropy over amino-acid logits.

   :param r_denoised_k: (..., N_res, 3)       — denoised coordinates from block k.
   :param r_gt: (..., N_res, 3)       — ground-truth coordinates  r̄⁰.
   :param logits_aa_k: (..., N_res, n_amino) — amino-acid logits â_(k).
   :param aa_gt: (..., N_res)          — ground-truth amino-acid class indices a⁰.
   :param lam: float                 — λ(t), noise-level weight for the structure term.
   :param alpha_0: float                 — α₀, fixed weight for the sequence CE term.
   :param mask: (..., N_res)          — boolean/float residue mask (optional).

   :returns: (...,) per-batch loss for block k.


.. py:function:: smooth_lddt_loss(r_pred: jaxtyping.Float[torch.Tensor, ... N_atom 3], r_true: jaxtyping.Float[torch.Tensor, ... N_atom 3], mask: Optional[jaxtyping.Bool[torch.Tensor, ... N_atom]] = None, cutoff: float = 15.0) -> jaxtyping.Float[torch.Tensor, ]

   Smooth lDDT loss — Algorithm 8 (simplified AF3 version for all-atom design).

   Exact algorithm:

   .. code-block:: none

       1. δr_lm     = ||r_l  − r_m||          (predicted pairwise distances)
       2. δr_lm_GT  = ||r_l_GT − r_m_GT||     (GT pairwise distances)
       3. δ_lm      = |δr_lm_GT − δr_lm|      (absolute distance difference)
       4. ε_lm      = ¼ [ σ(½ − δ_lm)
                         + σ(1  − δ_lm)
                         + σ(2  − δ_lm)
                         + σ(4  − δ_lm) ]     (smooth score ∈ (0,1))
       5. c_lm      = 1( δr_lm_GT < 15 Å )    (local-neighbourhood mask, l≠m)
          lddt      = mean_{l≠m}(c_lm · ε_lm)
                    / mean_{l≠m}(c_lm)
       return 1 − lddt

   :param r_pred: (..., N_atom, 3) — predicted atom coordinates  r̄_l.
   :param r_true: (..., N_atom, 3) — ground-truth coordinates  r̄_l^GT.
   :param mask: (..., N_atom)    — float/bool validity mask (optional).
                Pairs (l, m) are included only when both
                atoms are valid.
   :param cutoff: float            — local neighbourhood radius in Å (default 15.0).

   :returns: Scalar loss = 1 − lddt_smooth  (averaged over batch).



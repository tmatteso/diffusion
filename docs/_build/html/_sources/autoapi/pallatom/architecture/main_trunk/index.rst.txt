pallatom.architecture.main_trunk
================================

.. py:module:: pallatom.architecture.main_trunk

.. autoapi-nested-parse::

   MainTrunk — pure PyTorch implementation
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



Classes
-------

.. autoapisummary::

   pallatom.architecture.main_trunk.AtomDistogramHead
   pallatom.architecture.main_trunk.MainTrunk
   pallatom.architecture.main_trunk.RelativePositionEncoding
   pallatom.architecture.main_trunk.ResidueDistogramHead
   pallatom.architecture.main_trunk.TimeFourierEmbedding


Functions
---------

.. autoapisummary::

   pallatom.architecture.main_trunk.sinusoidal_encoding


Module Contents
---------------

.. py:class:: AtomDistogramHead(c_atompair: int, n_bins: int = 22, d_min: float = 0.0, d_max: float = 10.0, atoms_per_res: int = 3)

   Bases: :py:obj:`torch.nn.Module`


   Projects atom-pair embeddings p_lm → 22 distance-bin logits,
   restricted to a local 5L × 5L window.

   :param c_atompair:
   :type c_atompair: input atom-pair embedding dim
   :param n_bins:
   :type n_bins: number of distance bins (default 22)
   :param d_min:
   :type d_min: minimum distance in Å   (default 0.0)
   :param d_max:
   :type d_max: maximum distance in Å   (default 10.0)
   :param atoms_per_res:
   :type atoms_per_res: L — average atoms per residue, used to define window size
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]

      :returns: * **logits** (*(N_atom, N_atom, n_bins)  — full grid (unmasked pairs = 0 logit)*)
                * **mask** (*(N_atom, N_atom)          — True for pairs inside the local window*)



   .. py:method:: loss(p: torch.Tensor, targets: torch.Tensor) -> torch.Tensor

      Cross-entropy loss over the local 5L × 5L window only.



   .. py:attribute:: d_max
      :value: 10.0



   .. py:attribute:: d_min
      :value: 0.0



   .. py:attribute:: n_bins
      :value: 22



   .. py:attribute:: norm


   .. py:attribute:: proj1


   .. py:attribute:: proj2


   .. py:attribute:: window
      :value: 15



.. py:class:: MainTrunk(f_ref_dim: int = 35, n_bins: int = 38, n_atom_bins: int = 22, c_atom: int = 128, c_pair: int = 128, c_res: int = 256, c_atompair: int = 16, n_blocks: int = 2, n_heads: int = 4, sigma_data: float = 16.0, K_unit: int = 3, n_amino: int = 20)

   Bases: :py:obj:`torch.nn.Module`


   :param f_ref_dim:
   :type f_ref_dim: per-atom f^ref feature size (3 + element_dim after tile)
   :param n_bins:
   :type n_bins: distogram bins for TemplateEmbedder
   :param c_atom:
   :type c_atom: atom single dim        (default 128)
   :param c_pair:
   :type c_pair: trunk pair dim         (default 128)
   :param c_res:
   :type c_res: trunk single/residue dim (default 256)
   :param c_atompair:
   :type c_atompair: atom-pair dim          (default 16)
   :param sigma_data:
   :type sigma_data: data noise level       (default 16)
   :param K_unit:
   :type K_unit: number of decoder units (default 3)
   :param n_amino:
   :type n_amino: amino-acid vocabulary  (default 20)
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(batch: helpers.featurize.FeaturizedBatch) -> tuple[jaxtyping.Float[torch.Tensor, B N_atom 3], jaxtyping.Float[torch.Tensor, B N_res n_amino], jaxtyping.Float[torch.Tensor, B N_res N_res n_bins], jaxtyping.Float[torch.Tensor, B N_atom K n_atom_bins], list[jaxtyping.Float[torch.Tensor, B N_atom 3]], list[jaxtyping.Float[torch.Tensor, B N_res n_amino]]]


   .. py:attribute:: K_unit
      :value: 3



   .. py:attribute:: atom_decoders


   .. py:attribute:: atom_distogram_head


   .. py:attribute:: atom_encoder


   .. py:attribute:: inter_proj_seq


   .. py:attribute:: inter_seq_logits


   .. py:attribute:: node_updates


   .. py:attribute:: norm_s_init


   .. py:attribute:: pair_updates


   .. py:attribute:: proj_residue_idx


   .. py:attribute:: proj_s_init


   .. py:attribute:: proj_seq


   .. py:attribute:: rel_pos_enc


   .. py:attribute:: residue_distogram_head


   .. py:attribute:: seq_logits


   .. py:attribute:: sigma_data
      :value: 16.0



   .. py:attribute:: template_embedder


   .. py:attribute:: time_fourier


.. py:class:: RelativePositionEncoding(c_pair: int, max_rel: int = 32)

   Bases: :py:obj:`torch.nn.Module`


   Standard clipped relative position encoding.
   Produces z_ij^init ∈ R^{c_pair} from residue index differences.

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(N_token: int, device: torch.device) -> torch.Tensor


   .. py:attribute:: max_rel
      :value: 32



   .. py:attribute:: proj


.. py:class:: ResidueDistogramHead(c_pair: int, n_bins: int = 64, d_min: float = 2.0, d_max: float = 22.0)

   Bases: :py:obj:`torch.nn.Module`


   Projects symmetrised pair embeddings z_ij → 64 distance-bin logits.

   :param c_pair:
   :type c_pair: input pair embedding dim
   :param n_bins:
   :type n_bins: number of distance bins (default 64)
   :param d_min:
   :type d_min: minimum distance in Å   (default 2.0)
   :param d_max:
   :type d_max: maximum distance in Å   (default 22.0)
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(z: torch.Tensor) -> torch.Tensor

      z : (..., N_token, N_token, c_pair) — accepts unbatched or batched input
      returns logits : (..., N_token, N_token, n_bins)



   .. py:method:: loss(z: torch.Tensor, targets: torch.Tensor) -> torch.Tensor

      Convenience: compute cross-entropy loss against ground-truth positions.



   .. py:attribute:: d_max
      :value: 22.0



   .. py:attribute:: d_min
      :value: 2.0



   .. py:attribute:: n_bins
      :value: 64



   .. py:attribute:: norm


   .. py:attribute:: proj1


   .. py:attribute:: proj2


.. py:class:: TimeFourierEmbedding(c_res: int)

   Bases: :py:obj:`torch.nn.Module`


   Maps scalar x = ¼·log(t̂/σ_data) to a Fourier feature vector ∈ R^{c_res}.
   Uses learnable frequencies (as in AF3 / common diffusion practice).

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


   .. py:attribute:: freqs


   .. py:attribute:: proj


.. py:function:: sinusoidal_encoding(positions: jaxtyping.Float[torch.Tensor, batch N_res], dim: int = 32) -> jaxtyping.Float[torch.Tensor, batch N_res dim]

   Sinusoidal positional encoding. positions: (..., Nres,) → (..., Nres, dim)



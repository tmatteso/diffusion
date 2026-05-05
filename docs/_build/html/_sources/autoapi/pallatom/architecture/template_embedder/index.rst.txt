pallatom.architecture.template_embedder
=======================================

.. py:module:: pallatom.architecture.template_embedder


Classes
-------

.. autoapisummary::

   pallatom.architecture.template_embedder.TemplateEmbedder


Module Contents
---------------

.. py:class:: TemplateEmbedder(n_bins: int, c_z: int, c: int = 64, d: int = 128, n_blocks: int = 2, n_heads: int = 4)

   Bases: :py:obj:`torch.nn.Module`


   :param n_bins:
   :type n_bins: number of distogram bins  (f_distogram last dim)
   :param c_z:
   :type c_z: trunk pair embedding dim  (z_ij last dim)
   :param c:
   :type c: internal pair dim         (default 64)
   :param d:
   :type d: output pair dim           (default 128)
   :param n_blocks:
   :type n_blocks: PairformerStack depth     (default 2)
   :param n_heads:
   :type n_heads: attention heads           (default 4)
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(f_distogram: jaxtyping.Float[torch.Tensor, B N_res N_res n_bins], f_pseudo_beta_mask: jaxtyping.Float[torch.Tensor, B N_res], z_ij: jaxtyping.Float[torch.Tensor, B N_res N_res c_z], t: float) -> jaxtyping.Float[torch.Tensor, B N_res N_res d]


   .. py:attribute:: norm_v


   .. py:attribute:: norm_z


   .. py:attribute:: pairformer


   .. py:attribute:: proj_a


   .. py:attribute:: proj_out


   .. py:attribute:: proj_z



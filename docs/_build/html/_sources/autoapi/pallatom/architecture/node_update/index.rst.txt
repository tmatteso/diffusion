pallatom.architecture.node_update
=================================

.. py:module:: pallatom.architecture.node_update


Classes
-------

.. autoapisummary::

   pallatom.architecture.node_update.AttentionPairBias
   pallatom.architecture.node_update.NodeUpdate


Module Contents
---------------

.. py:class:: AttentionPairBias(c: int, c_pair: int, n_heads: int = 8)

   Bases: :py:obj:`torch.nn.Module`


   Self-attention on node embeddings s_i biased by pair embeddings z_ij.

   :param c:
   :type c: single embedding dim  (default 256)
   :param c_pair:
   :type c_pair: pair   embedding dim
   :param n_heads:
   :type n_heads: number of attention heads (default 8 per Algorithm 6)
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(s: jaxtyping.Float[torch.Tensor, B N_res c_res], t: jaxtyping.Float[torch.Tensor, B N_res c_res], z: jaxtyping.Float[torch.Tensor, B N_res N_res c_pair]) -> jaxtyping.Float[torch.Tensor, B N_res c_res]


   .. py:attribute:: head_dim


   .. py:attribute:: n_heads
      :value: 8



   .. py:attribute:: norm_s


   .. py:attribute:: norm_t


   .. py:attribute:: norm_z


   .. py:attribute:: proj_t


   .. py:attribute:: to_bias


   .. py:attribute:: to_g


   .. py:attribute:: to_k


   .. py:attribute:: to_out


   .. py:attribute:: to_q


   .. py:attribute:: to_v


.. py:class:: NodeUpdate(c: int = 256, c_pair: int = 128, n_heads: int = 8, dropout: float = 0.25)

   Bases: :py:obj:`torch.nn.Module`


   :param c:
   :type c: single embedding dim  (default 256)
   :param c_pair:
   :type c_pair: pair   embedding dim
   :param n_heads:
   :type n_heads: attention heads       (default 8, per Algorithm 6)
   :param dropout:
   :type dropout: rowwise dropout prob  (default 0.25, per Algorithm 6)
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(s: jaxtyping.Float[torch.Tensor, B N_res c_res], t: jaxtyping.Float[torch.Tensor, B N_res c_res], z: jaxtyping.Float[torch.Tensor, B N_res N_res c_pair]) -> jaxtyping.Float[torch.Tensor, B N_res c_res]


   .. py:attribute:: attn_pair_bias


   .. py:attribute:: dropout_row


   .. py:attribute:: transition



pallatom.architecture.atom_transformers
=======================================

.. py:module:: pallatom.architecture.atom_transformers


Attributes
----------

.. autoapisummary::

   pallatom.architecture.atom_transformers.WINDOW_SIZE


Classes
-------

.. autoapisummary::

   pallatom.architecture.atom_transformers.AtomAttentionDecoder
   pallatom.architecture.atom_transformers.AtomFeatureEncoder
   pallatom.architecture.atom_transformers.AtomTransformer
   pallatom.architecture.atom_transformers.AtomTransformerBlock
   pallatom.architecture.atom_transformers.LinearNoBias


Functions
---------

.. autoapisummary::

   pallatom.architecture.atom_transformers.build_sparse_pairs


Module Contents
---------------

.. py:class:: AtomAttentionDecoder(c_token: int, c_pair: int, c_atom: int, c_atompair: int, n_blocks: int = 3, n_heads: int = 4, window_size: int = WINDOW_SIZE)

   Bases: :py:obj:`torch.nn.Module`


   Decodes trunk embeddings back to per-atom position updates.

   Like the encoder, all pair tensors are sparse [B, N, K, *].
   Builds its own neighbour index from tok_idx; no dense mask required.

   :param c_token:
   :type c_token: trunk single dim  (s)
   :param c_pair:
   :type c_pair: trunk pair dim    (z)
   :param c_atom:
   :type c_atom: atom single dim   (q, c)
   :param c_atompair:
   :type c_atompair: atom-pair dim     (p)
   :param n_blocks:
   :type n_blocks: AtomTransformer blocks (default 3)
   :param n_heads:
   :type n_heads: attention heads        (default 4)
   :param window_size:
   :type window_size: local window in residues (default 32)
   :param Weight shapes: proj_s_q / proj_s_c : [c_token, c_atom]      (no bias)
                         proj_z              : [c_pair, c_atompair]    (no bias)
                         proj_r              : [c_atom, 3]             (no bias)
                         mlp_p layers        : [c_atompair, c_atompair] each (no bias)
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(q_skip: jaxtyping.Float[torch.Tensor, B N_atom c_atom], p_skip: jaxtyping.Float[torch.Tensor, B N_atom K c_atompair], c_skip: jaxtyping.Float[torch.Tensor, B N_atom c_atom], c: jaxtyping.Float[torch.Tensor, B N_atom c_atom], s: jaxtyping.Float[torch.Tensor, B N_res c_token], z: jaxtyping.Float[torch.Tensor, B N_res N_res c_pair], tok_idx: jaxtyping.Int[torch.Tensor, B N_atom]) -> tuple[jaxtyping.Float[torch.Tensor, B N_atom 3], jaxtyping.Float[torch.Tensor, B N_atom c_atom]]

      :param q_skip: atom skip queries      [B, N_atom, c_atom]
      :param p_skip: sparse pair skip       [B, N_atom, K, c_atompair]  (from encoder)
      :param c_skip: atom skip context      [B, N_atom, c_atom]
      :param c: atom context           [B, N_atom, c_atom]
      :param s: trunk single embeds    [B, N_res, c_token]
      :param z: trunk pair embeds      [B, N_res, N_res, c_pair]
      :param tok_idx: residue index per atom [B, N_atom]

      :returns: per-atom position update  [B, N_atom, 3]
                c_out    : updated atom context      [B, N_atom, c_atom]
      :rtype: r_update



   .. py:attribute:: mlp_p


   .. py:attribute:: norm_q_out


   .. py:attribute:: norm_s_c


   .. py:attribute:: norm_s_q


   .. py:attribute:: norm_z


   .. py:attribute:: proj_r


   .. py:attribute:: proj_s_c


   .. py:attribute:: proj_s_q


   .. py:attribute:: proj_z


   .. py:attribute:: transformer


   .. py:attribute:: window_size
      :value: 32



.. py:class:: AtomFeatureEncoder(f_ref_dim: int, c_token: int, c_pair: int, c: int, d: int, m: int, n_blocks: int, n_heads: int, window_size: int = WINDOW_SIZE)

   Bases: :py:obj:`torch.nn.Module`


   Encodes per-atom reference features into per-residue embeddings.

   All N×N pair tensors (d_lm, p_lm, z_gathered) are replaced by sparse
   [B, N, K, *] tensors indexed over the N × K live pairs within the 32-residue
   window. K is the maximum number of window-neighbours any atom has.

   :param f_ref_dim:
   :type f_ref_dim: per-atom f^ref feature size (ref_pos_dim + ref_element_dim)
   :param c_token:
   :type c_token: trunk single dim   (s_input)
   :param c_pair:
   :type c_pair: trunk pair dim     (z_input)
   :param c:
   :type c: output per-residue dim
   :param d:
   :type d: atom-pair embedding dim
   :param m:
   :type m: atom single embedding dim
   :param n_blocks:
   :type n_blocks: AtomTransformer blocks
   :param n_heads:
   :type n_heads: AtomTransformer heads
   :param window_size:
   :type window_size: local attention window in residues (default 32)
   :param Weight shapes: proj_fref_c              : [f_ref_dim, m]
                         proj_d_vec               : [3, d]
                         proj_inv_sq              : [1, d]
                         proj_v                   : [1, d]
                         proj_cl_pair / cm_pair   : [m, d]
                         proj_r_scaled            : [3, m]
                         proj_s_init              : [c_token, m]
                         proj_z_init              : [c_pair, d]
                         mlp_p layers             : [d, d] each
                         proj_agg                 : [m, c]
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(ref_pos: jaxtyping.Float[torch.Tensor, B N_atom 3], ref_element: jaxtyping.Float[torch.Tensor, B N_atom E], ref_space_uid: jaxtyping.Int[torch.Tensor, B N_atom], s_input: jaxtyping.Float[torch.Tensor, B N_res c_token], z_input: jaxtyping.Float[torch.Tensor, B N_res N_res c_pair], r_scaled: jaxtyping.Float[torch.Tensor, B N_atom 3], tok_idx: jaxtyping.Int[torch.Tensor, B N_atom]) -> tuple[jaxtyping.Float[torch.Tensor, B N_res c_res], jaxtyping.Float[torch.Tensor, B N_atom c_atom], jaxtyping.Float[torch.Tensor, B N_atom c_atom], jaxtyping.Float[torch.Tensor, B N_atom K c_atompair], jaxtyping.Float[torch.Tensor, B N_atom c_atom]]

      :param ref_pos: [B, N_atom, 3]              reference atom positions
      :param ref_element: [B, N_atom, E]              element one-hot features
      :param ref_space_uid: [B, N_atom]                 chain/space identifier per atom
      :param s_input: [B, N_res, c_token]         trunk single embeddings
      :param z_input: [B, N_res, N_res, c_pair]   trunk pair embeddings
      :param r_scaled: [B, N_atom, 3]              scaled noisy positions
      :param tok_idx: [B, N_atom]                 residue index per atom (0-based)

      Returns (all sparse where applicable):
          s_i    : [B, N_res, c]              aggregated residue embeddings
          q_skip : [B, N_atom, m]             atom skip queries (post-transformer)
          c_skip : [B, N_atom, m]             atom skip context
          p_skip : [B, N_atom, K, d]          sparse atom-pair skip embeddings
          c_l    : [B, N_atom, m]             updated atom context



   .. py:attribute:: m


   .. py:attribute:: mlp_p


   .. py:attribute:: norm_s_init


   .. py:attribute:: norm_z_init


   .. py:attribute:: proj_agg


   .. py:attribute:: proj_cl_pair


   .. py:attribute:: proj_cm_pair


   .. py:attribute:: proj_d_vec


   .. py:attribute:: proj_fref_c


   .. py:attribute:: proj_inv_sq


   .. py:attribute:: proj_r_scaled


   .. py:attribute:: proj_s_init


   .. py:attribute:: proj_v


   .. py:attribute:: proj_z_init


   .. py:attribute:: transformer


   .. py:attribute:: window_size
      :value: 32



.. py:class:: AtomTransformer(c_atom: int, c_atompair: int, n_blocks: int = 3, n_heads: int = 4, window_size: int = WINDOW_SIZE)

   Bases: :py:obj:`torch.nn.Module`


   Stack of n_blocks AtomTransformerBlocks, all operating on sparse [B, N, K, *] pairs.

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(q: jaxtyping.Float[torch.Tensor, B N c_atom], c: jaxtyping.Float[torch.Tensor, B N c_atom], p: jaxtyping.Float[torch.Tensor, B N K c_atompair], neighbor_idx: jaxtyping.Int[torch.Tensor, N K], valid_mask: jaxtyping.Bool[torch.Tensor, B N K]) -> jaxtyping.Float[torch.Tensor, B N c_atom]

      :param q: atom query embeddings   [B, N, c_atom]
      :param c: atom context embeddings [B, N, c_atom]
      :param p: sparse pair embeddings  [B, N, K, c_atompair]
      :param neighbor_idx: neighbour atom indices  [N, K]
      :param valid_mask: live-pair mask          [B, N, K]

      :returns: updated atom embeddings            [B, N, c_atom]
      :rtype: q



   .. py:attribute:: blocks


   .. py:attribute:: window_size
      :value: 32



.. py:class:: AtomTransformerBlock(c_atom: int, c_atompair: int, n_heads: int, window_size: int = WINDOW_SIZE)

   Bases: :py:obj:`torch.nn.Module`


   Single block of the AtomTransformer with a 32-residue local attention window.

   All pair tensors are sparse [B, N, K, *] — no N² allocation.

   Weight shapes:
       to_q / to_k / to_v / to_out : [c_atom, c_atom]       (no bias)
       pair_bias                    : [c_atompair, n_heads]   (no bias)
       ff1                          : [c_atom, c_atom * 4]    (no bias)
       ff2                          : [c_atom * 4, c_atom]    (no bias)

   Sparse attention shapes (window_size=32, K = live neighbours per atom):
       Q, K_proj, V : [B, N, n_heads, head_dim]
       K_nbr, V_nbr : [B, N, K, n_heads, head_dim]  — gathered over K neighbours
       scores       : [B, N, K, n_heads]             — O(N × K), not O(N²)
       pair bias    : [B, N, K, n_heads]

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(q: jaxtyping.Float[torch.Tensor, B N c_atom], c: jaxtyping.Float[torch.Tensor, B N c_atom], p: jaxtyping.Float[torch.Tensor, B N K c_atompair], neighbor_idx: jaxtyping.Int[torch.Tensor, N K], valid_mask: jaxtyping.Bool[torch.Tensor, B N K]) -> jaxtyping.Float[torch.Tensor, B N c_atom]

      :param q: atom query embeddings      [B, N, c_atom]
      :param c: atom context embeddings    [B, N, c_atom]
      :param p: sparse pair embeddings     [B, N, K, c_atompair]
                p[b, l, k] = features for pair (l, neighbor_idx[l, k])
      :param neighbor_idx: window neighbour indices   [N, K]
      :param valid_mask: True = live pair           [B, N, K]
                         encodes both window and chain constraints

      :returns: updated atom embeddings               [B, N, c_atom]
      :rtype: q



   .. py:attribute:: ff1


   .. py:attribute:: ff2


   .. py:attribute:: head_dim


   .. py:attribute:: n_heads


   .. py:attribute:: norm_c


   .. py:attribute:: norm_ff


   .. py:attribute:: norm_pair


   .. py:attribute:: norm_q


   .. py:attribute:: pair_bias


   .. py:attribute:: to_k


   .. py:attribute:: to_out


   .. py:attribute:: to_q


   .. py:attribute:: to_v


   .. py:attribute:: window_size
      :value: 32



.. py:class:: LinearNoBias(in_features: int, out_features: int)

   Bases: :py:obj:`torch.nn.Linear`


   Linear layer with no bias term.

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


.. py:function:: build_sparse_pairs(tok_idx: jaxtyping.Int[torch.Tensor, build_sparse_pairs.N], window_size: int = WINDOW_SIZE) -> tuple[jaxtyping.Int[torch.Tensor, N K], jaxtyping.Bool[torch.Tensor, N K]]

   For each atom l, collect indices of all atoms m within the 32-residue window.

   Uses an O(N²) boolean intermediate (one byte per pair) to build the index,
   then returns O(N × K) long + bool tensors where K << N for large proteins.

   :param tok_idx: [N]  residue index (0-based) for each atom
   :param window_size: total window span in residues (default 32)
                       atom l attends to m iff ``|tok_idx[l] - tok_idx[m]|`` < window_size // 2

   :returns: [N, K]  atom indices of each neighbour; padding slots → 0
             valid_mask   : [N, K]  True where the slot is a real neighbour (not padding)
   :rtype: neighbor_idx

   K = maximum neighbours any single atom has; varies with sequence length and
   atoms-per-residue. For a 32-residue window with ~14 atoms/residue, K ≈ 448.


.. py:data:: WINDOW_SIZE
   :type:  int
   :value: 32



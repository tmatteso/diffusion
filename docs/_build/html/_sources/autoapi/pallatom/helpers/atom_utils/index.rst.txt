pallatom.helpers.atom_utils
===========================

.. py:module:: pallatom.helpers.atom_utils


Attributes
----------

.. autoapisummary::

   pallatom.helpers.atom_utils.ATOM37_C
   pallatom.helpers.atom_utils.ATOM37_CA
   pallatom.helpers.atom_utils.ATOM37_CB
   pallatom.helpers.atom_utils.ATOM37_N
   pallatom.helpers.atom_utils.ATOM37_O
   pallatom.helpers.atom_utils.ATOM5_C
   pallatom.helpers.atom_utils.ATOM5_CA
   pallatom.helpers.atom_utils.ATOM5_CB
   pallatom.helpers.atom_utils.ATOM5_ELEMENTS
   pallatom.helpers.atom_utils.ATOM5_N
   pallatom.helpers.atom_utils.ATOM5_NAMES
   pallatom.helpers.atom_utils.ATOM5_O
   pallatom.helpers.atom_utils.PDB_CHAIN_IDS
   pallatom.helpers.atom_utils.PDB_MAX_CHAINS
   pallatom.helpers.atom_utils.atom_types
   pallatom.helpers.atom_utils.restype_1to3
   pallatom.helpers.atom_utils.restype_num
   pallatom.helpers.atom_utils.restype_order
   pallatom.helpers.atom_utils.restypes
   pallatom.helpers.atom_utils.rigid_group_atom_positions


Classes
-------

.. autoapisummary::

   pallatom.helpers.atom_utils.Protein


Functions
---------

.. autoapisummary::

   pallatom.helpers.atom_utils.atom37_to_atom5
   pallatom.helpers.atom_utils.atom37_to_cb
   pallatom.helpers.atom_utils.center_positions
   pallatom.helpers.atom_utils.get_cb_coords
   pallatom.helpers.atom_utils.make_fixed_size
   pallatom.helpers.atom_utils.make_np_example
   pallatom.helpers.atom_utils.pseudo_cb
   pallatom.helpers.atom_utils.to_pdb


Module Contents
---------------

.. py:class:: Protein

   Protein structure representation.


   .. py:attribute:: aatype
      :type:  jaxtyping.Int[numpy.ndarray, num_res]


   .. py:attribute:: atom_mask
      :type:  jaxtyping.Float[numpy.ndarray, num_res num_atom_type]


   .. py:attribute:: atom_positions
      :type:  jaxtyping.Float[numpy.ndarray, num_res num_atom_type 3]


   .. py:attribute:: b_factors
      :type:  jaxtyping.Float[numpy.ndarray, num_res num_atom_type]


   .. py:attribute:: chain_index
      :type:  jaxtyping.Int[numpy.ndarray, num_res]


   .. py:attribute:: residue_index
      :type:  jaxtyping.Int[numpy.ndarray, num_res]


.. py:function:: atom37_to_atom5(atom37_positions: jaxtyping.Float[torch.Tensor, B N_res 37 3], atom37_mask: jaxtyping.Float[torch.Tensor, B N_res 37]) -> tuple[jaxtyping.Float[torch.Tensor, B N_res 5 3], jaxtyping.Float[torch.Tensor, B N_res 5]]

   Extract the 5 backbone+Cβ atoms from atom37 representation.

   :returns: * **atom5_positions** (*(B, N_res, 5, 3)*)
             * **atom5_mask** (*(B, N_res, 5)*)


.. py:function:: atom37_to_cb(atom37_positions: jaxtyping.Float[torch.Tensor, B N_res 37 3], atom37_mask: jaxtyping.Float[torch.Tensor, B N_res 37]) -> tuple[jaxtyping.Float[torch.Tensor, B N_res 3], jaxtyping.Bool[torch.Tensor, B N_res]]

   Full pipeline: atom37 → atom5 → Cβ / pseudo-Cβ.

   :returns: * **cb** (*(B, N_res, 3)  — real Cβ where available, pseudo-Cβ otherwise*)
             * **pseudo_beta_mask** (*(B, N_res)     — True where real Cβ is present, False where pseudo-Cβ was used*)


.. py:function:: center_positions(np_example)

   Center 'atom_positions' on CA center of mass.


.. py:function:: get_cb_coords(atom5_positions: jaxtyping.Float[torch.Tensor, B N_res 5 3], atom5_mask: jaxtyping.Float[torch.Tensor, B N_res 5], fill_pseudo: bool = True) -> tuple[jaxtyping.Float[torch.Tensor, B N_res 3], jaxtyping.Bool[torch.Tensor, B N_res]]

   Extract Cβ (slot 4) from atom5, replacing missing Cβ (Gly) with pseudo-Cβ.

   :param atom5_positions:
   :type atom5_positions: (B, N_res, 5, 3)
   :param atom5_mask:
   :type atom5_mask: (B, N_res, 5)  — 1 where atom is present
   :param fill_pseudo:
   :type fill_pseudo: if True, compute pseudo-Cβ wherever Cβ is absent

   :returns: * **cb** (*(B, N_res, 3)  — real Cβ where available, pseudo-Cβ otherwise*)
             * **pseudo_beta_mask** (*(B, N_res)     — True where real Cβ is present, False where pseudo-Cβ was used*)


.. py:function:: make_fixed_size(np_example, max_seq_length=500)

   Pad features to fixed sequence length, i.e. currently axis=0.


.. py:function:: make_np_example(coords_dict)

   Make a dictionary of non-batched numpy protein features.


.. py:function:: pseudo_cb(n: jaxtyping.Float[torch.Tensor, ... 3], ca: jaxtyping.Float[torch.Tensor, ... 3], c: jaxtyping.Float[torch.Tensor, ... 3]) -> jaxtyping.Float[torch.Tensor, ... 3]

   Compute a virtual Cβ from backbone geometry (Gly-safe).

   Uses the standard ideal-geometry recipe:
     b = Cα - N        (N→Cα bond vector)
     d = C  - Cα       (Cα→C  bond vector)
     Cross them, then combine with ideal tetrahedral offsets.

   This matches the AlphaFold2 / ESMFold convention exactly.


.. py:function:: to_pdb(prot: Protein) -> str

   Converts a `Protein` instance to a PDB string.

   :param prot: The protein to convert to PDB.

   :returns: PDB string.


.. py:data:: ATOM37_C
   :value: 2


.. py:data:: ATOM37_CA
   :value: 1


.. py:data:: ATOM37_CB
   :value: 4


.. py:data:: ATOM37_N
   :value: 0


.. py:data:: ATOM37_O
   :value: 3


.. py:data:: ATOM5_C
   :value: 2


.. py:data:: ATOM5_CA
   :value: 1


.. py:data:: ATOM5_CB
   :value: 4


.. py:data:: ATOM5_ELEMENTS

.. py:data:: ATOM5_N
   :value: 0


.. py:data:: ATOM5_NAMES
   :value: ['N', 'CA', 'C', 'O', 'CB']


.. py:data:: ATOM5_O
   :value: 3


.. py:data:: PDB_CHAIN_IDS
   :value: 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


.. py:data:: PDB_MAX_CHAINS
   :value: 62


.. py:data:: atom_types
   :value: ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1',...


.. py:data:: restype_1to3

.. py:data:: restype_num
   :value: 20


.. py:data:: restype_order

.. py:data:: restypes
   :value: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


.. py:data:: rigid_group_atom_positions


pallatom.helpers.featurize
==========================

.. py:module:: pallatom.helpers.featurize


Classes
-------

.. autoapisummary::

   pallatom.helpers.featurize.Distogram
   pallatom.helpers.featurize.FeaturizedBatch
   pallatom.helpers.featurize.ProteinBatch
   pallatom.helpers.featurize.ResidueIndexEmbedding


Functions
---------

.. autoapisummary::

   pallatom.helpers.featurize.featurize_batch


Module Contents
---------------

.. py:class:: Distogram(n_bins: int, min_dist: float = 2.0, max_dist: float = 22.0, overflow_bin: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   Pairwise distogram module.

   Precomputes bin edges once at init; forward() maps per-residue coordinates
   → one-hot distogram + validity mask.  Accepts either:

   - ``(..., N, 3)``    — single atom per residue (e.g. pseudo-Cβ), auto-expanded to ``(..., N, 1, 3)``
   - ``(..., N, A, 3)`` — A atoms per residue (e.g. atom5, atom37)

   :param n_bins: Number of distance bins.
   :param min_dist: Lower edge of first bin in Ångströms (default 2.0).
   :param max_dist: Upper edge of last bin in Ångströms (default 22.0).
   :param overflow_bin: If True, adds one extra bin capturing distances > max_dist,
                        making the output shape (..., n_bins + 1) instead of (..., n_bins).

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: extra_repr() -> str

      Return the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



   .. py:method:: forward(coords: jaxtyping.Float[torch.Tensor, ... total_atom_count 3], coords_mask: Optional[jaxtyping.Bool[torch.Tensor, ... total_atom_count]] = None) -> tuple[jaxtyping.Float[torch.Tensor, ... total_atom_count total_atom_count n_bins], jaxtyping.Bool[torch.Tensor, ... total_atom_count total_atom_count]]

      :param coords: (..., total_atom_count, 3)
      :param coords_mask: (..., total_atom_count) — 1 where valid; all-ones if None.

      :returns:

                (..., total_atom_count, total_atom_count, n_bins [+1]) — one-hot bin assignment;
                             last bin is the overflow bin when overflow_bin=True.
                f_pair_mask: (..., total_atom_count, total_atom_count) bool — True where pair is valid.
                             overflow_bin=True:  valid atom pairs only.
                             overflow_bin=False: valid atom pairs AND dist <= max_dist.
      :rtype: f_distogram



   .. py:attribute:: max_dist
      :value: 22.0



   .. py:attribute:: min_dist
      :value: 2.0



   .. py:attribute:: n_bins


   .. py:attribute:: overflow_bin
      :value: False



.. py:class:: FeaturizedBatch

   .. py:attribute:: aa_indices
      :type:  jaxtyping.Int[torch.Tensor, B N_res]


   .. py:attribute:: atom5_mask
      :type:  jaxtyping.Bool[torch.Tensor, B N_atom]


   .. py:attribute:: center_uid
      :type:  jaxtyping.Int[torch.Tensor, B N_res]


   .. py:attribute:: f_pseudo_beta_mask
      :type:  jaxtyping.Int[torch.Tensor, B N_res]


   .. py:attribute:: f_residue_idx
      :type:  jaxtyping.Float[torch.Tensor, B N_res c_res]


   .. py:attribute:: gt_atom_distogram_mask_sparse
      :type:  jaxtyping.Bool[torch.Tensor, B N_atom K]


   .. py:attribute:: gt_atom_distogram_sparse
      :type:  jaxtyping.Float[torch.Tensor, B N_atom K n_atom_bins]


   .. py:attribute:: gt_res_distogram
      :type:  jaxtyping.Int[torch.Tensor, B N_res N_res n_templ_bins]


   .. py:attribute:: r_gt
      :type:  jaxtyping.Float[torch.Tensor, B N_atom 3]


   .. py:attribute:: r_input
      :type:  jaxtyping.Float[torch.Tensor, B N_atom 3]


   .. py:attribute:: ref_element
      :type:  jaxtyping.Float[torch.Tensor, B N_atom 4]


   .. py:attribute:: ref_pos
      :type:  jaxtyping.Float[torch.Tensor, B N_atom 3]


   .. py:attribute:: ref_space_uid
      :type:  jaxtyping.Int[torch.Tensor, B N_atom]


   .. py:attribute:: residue_mask
      :type:  jaxtyping.Bool[torch.Tensor, B N_res]


   .. py:attribute:: t_hat
      :type:  float


   .. py:attribute:: t_normalized
      :type:  float


   .. py:attribute:: tok_idx
      :type:  jaxtyping.Int[torch.Tensor, B N_atom]


.. py:class:: ProteinBatch

   .. py:attribute:: atom_mask
      :type:  jaxtyping.Float[torch.Tensor, B N_res 37]


   .. py:attribute:: atom_positions
      :type:  jaxtyping.Float[torch.Tensor, B N_res 37 3]


   .. py:attribute:: residue_index
      :type:  jaxtyping.Float[torch.Tensor, B N_res]


   .. py:attribute:: seq
      :type:  list[str]


.. py:class:: ResidueIndexEmbedding(max_residues: int = 1024, c_res: int = 256)

   Bases: :py:obj:`torch.nn.Module`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing them to be nested in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F


       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will also have their
   parameters converted when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(residue_indices: jaxtyping.Int[torch.Tensor, N_res]) -> jaxtyping.Float[torch.Tensor, N_res c_res]


   .. py:attribute:: embed


.. py:function:: featurize_batch(batch: ProteinBatch, tcfg: train.train_config.TrainConfig, c_beta_distogram_fn: Distogram, atom_distogram_fn: Distogram, index_embedding: torch.nn.Embedding, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> FeaturizedBatch


pallatom.architecture.pair_update
=================================

.. py:module:: pallatom.architecture.pair_update

.. autoapi-nested-parse::

   PairUpdate — pure PyTorch implementation
   Based on Algorithm 7 from the AlphaFold 3 paper.

   Replaces the simplified PairUpdate stub in main_trunk.py.

   Steps
   -----
   1. d_ij = ||r_i^center − r_j^center||           scalar pairwise distances
   2. b_ij = LinearNoBias(Transform_RBF(d_ij))      RBF-discretized distance bias  ∈ R^c
   3. z_ij += DropoutRowwise_0.25(TriangleAttentionStartingNodeWithBias(z_ij, b_ij))
   4. z_ij += DropoutColumnwise_0.25(TriangleAttentionEndingNodeWithBias(z_ij, b_ij))
   5. z_ij += Transition(z_ij)



Classes
-------

.. autoapisummary::

   pallatom.architecture.pair_update.DropoutColumnwise
   pallatom.architecture.pair_update.DropoutRowwise
   pallatom.architecture.pair_update.PairUpdate
   pallatom.architecture.pair_update.TransformRBF
   pallatom.architecture.pair_update.Transition
   pallatom.architecture.pair_update.TriangleAttentionEndingNodeWithBias
   pallatom.architecture.pair_update.TriangleAttentionStartingNodeWithBias


Module Contents
---------------

.. py:class:: DropoutColumnwise(p: float = 0.25)

   Bases: :py:obj:`torch.nn.Module`


   Drops entire columns (dim 2) with probability p during training.

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


   .. py:attribute:: p
      :value: 0.25



.. py:class:: DropoutRowwise(p: float = 0.25)

   Bases: :py:obj:`torch.nn.Module`


   Drops entire rows (dim 1) with probability p during training.

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor


   .. py:attribute:: p
      :value: 0.25



.. py:class:: PairUpdate(c: int = 128, n_rbf: int = 16, n_heads: int = 4, dropout: float = 0.25)

   Bases: :py:obj:`torch.nn.Module`


   :param c:
   :type c: pair embedding dim  (default 128)
   :param n_rbf:
   :type n_rbf: number of RBF centres
   :param n_heads:
   :type n_heads: attention heads
   :param dropout:
   :type dropout: rowwise/columnwise dropout probability (0.25 per paper)
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(z: jaxtyping.Float[torch.Tensor, B N_res N_res c_pair], r_center: jaxtyping.Float[torch.Tensor, B N_res 3]) -> jaxtyping.Float[torch.Tensor, B N_res N_res c_pair]


   .. py:attribute:: drop_col


   .. py:attribute:: drop_row


   .. py:attribute:: rbf


   .. py:attribute:: transition


   .. py:attribute:: tri_end


   .. py:attribute:: tri_start


.. py:class:: TransformRBF(c: int, n_rbf: int = 16, d_min: float = 0.0, d_max: float = 22.0)

   Bases: :py:obj:`torch.nn.Module`


   Converts a scalar distance d into a fixed-size RBF feature vector,
   then projects to R^c via LinearNoBias.

   Centers are evenly spaced in [d_min, d_max]; width = spacing.

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(d: jaxtyping.Float[torch.Tensor, B N_res N_res]) -> jaxtyping.Float[torch.Tensor, B N_res N_res c_pair]


   .. py:attribute:: centers
      :type:  torch.Tensor


   .. py:attribute:: proj


   .. py:attribute:: sigma
      :value: 1.375



.. py:class:: Transition(c: int, expansion: int = 4)

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


   .. py:method:: forward(z: torch.Tensor) -> torch.Tensor


   .. py:attribute:: ff1


   .. py:attribute:: ff2


   .. py:attribute:: norm


.. py:class:: TriangleAttentionEndingNodeWithBias(c: int, n_heads: int = 4)

   Bases: :py:obj:`torch.nn.Module`


   For each column j, attend over all i using queries/keys/values from z_ij,
   biased by b_ij.

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(z: jaxtyping.Float[torch.Tensor, B N_res N_res c_pair], b: jaxtyping.Float[torch.Tensor, B N_res N_res c_pair]) -> jaxtyping.Float[torch.Tensor, B N_res N_res c_pair]


   .. py:attribute:: head_dim


   .. py:attribute:: n_heads
      :value: 4



   .. py:attribute:: norm


   .. py:attribute:: norm_b


   .. py:attribute:: to_b


   .. py:attribute:: to_g


   .. py:attribute:: to_k


   .. py:attribute:: to_out


   .. py:attribute:: to_q


   .. py:attribute:: to_v


.. py:class:: TriangleAttentionStartingNodeWithBias(c: int, n_heads: int = 4)

   Bases: :py:obj:`torch.nn.Module`


   For each row i, attend over all j using queries/keys/values from z_ij,
   with an additive pair bias b_ij projected to per-head scalars.
   Gate with a sigmoid on z_ij.

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(z: jaxtyping.Float[torch.Tensor, B N_res N_res c_pair], b: jaxtyping.Float[torch.Tensor, B N_res N_res c_pair]) -> jaxtyping.Float[torch.Tensor, B N_res N_res c_pair]


   .. py:attribute:: head_dim


   .. py:attribute:: n_heads
      :value: 4



   .. py:attribute:: norm


   .. py:attribute:: norm_b


   .. py:attribute:: to_b


   .. py:attribute:: to_g


   .. py:attribute:: to_k


   .. py:attribute:: to_out


   .. py:attribute:: to_q


   .. py:attribute:: to_v



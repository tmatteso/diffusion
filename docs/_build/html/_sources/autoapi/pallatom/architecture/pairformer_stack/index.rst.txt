pallatom.architecture.pairformer_stack
======================================

.. py:module:: pallatom.architecture.pairformer_stack


Classes
-------

.. autoapisummary::

   pallatom.architecture.pairformer_stack.PairformerBlock
   pallatom.architecture.pairformer_stack.PairformerStack


Module Contents
---------------

.. py:class:: PairformerBlock(c: int, n_heads: int = 4)

   Bases: :py:obj:`torch.nn.Module`


   One block of the Pairformer operating purely on pair embeddings v_ij.

   Uses:

   - Row-wise gated self-attention  (triangle attention, rows)
   - Column-wise gated self-attention (triangle attention, cols)
   - Pair transition FFN

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: forward(v: jaxtyping.Float[torch.Tensor, B N_res N_res c]) -> jaxtyping.Float[torch.Tensor, B N_res N_res c]


   .. py:attribute:: ff1


   .. py:attribute:: ff2


   .. py:attribute:: g_col


   .. py:attribute:: g_row


   .. py:attribute:: head_dim


   .. py:attribute:: k_col


   .. py:attribute:: k_row


   .. py:attribute:: n_heads
      :value: 4



   .. py:attribute:: norm_col


   .. py:attribute:: norm_ff


   .. py:attribute:: norm_row


   .. py:attribute:: out_col


   .. py:attribute:: out_row


   .. py:attribute:: q_col


   .. py:attribute:: q_row


   .. py:attribute:: v_col


   .. py:attribute:: v_row


.. py:class:: PairformerStack(c: int, n_blocks: int = 2, n_heads: int = 4)

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


   .. py:method:: forward(v: jaxtyping.Float[torch.Tensor, B N_res N_res c]) -> jaxtyping.Float[torch.Tensor, B N_res N_res c]


   .. py:attribute:: blocks



train_loop
==========

.. py:module:: train_loop


Attributes
----------

.. autoapisummary::

   train_loop.log
   train_loop.parser


Functions
---------

.. autoapisummary::

   train_loop.evaluate
   train_loop.train


Module Contents
---------------

.. py:function:: evaluate(model: architecture.main_trunk.MainTrunk, loader, tcfg: train.train_config.TrainConfig, distogram_res: helpers.featurize.Distogram, distogram_atom: helpers.featurize.Distogram, index_embedding: torch.nn.Embedding, device: str) -> dict[str, float]

   Full-dataset evaluation pass. Returns mean loss per metric.


.. py:function:: train(model: architecture.main_trunk.MainTrunk, tcfg: train.train_config.TrainConfig, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, distogram_res: helpers.featurize.Distogram, distogram_atom: helpers.featurize.Distogram, index_embedding: torch.nn.Embedding, device: str) -> None

.. py:data:: log

.. py:data:: parser


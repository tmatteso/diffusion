pallatom.helpers.data
=====================

.. py:module:: pallatom.helpers.data


Classes
-------

.. autoapisummary::

   pallatom.helpers.data.ProteinDataset


Functions
---------

.. autoapisummary::

   pallatom.helpers.data.make_data_loaders


Module Contents
---------------

.. py:class:: ProteinDataset(jsonl_path: str | pathlib.Path, names: list[str], max_seq_length: int = 256)

   Bases: :py:obj:`torch.utils.data.Dataset`


   Lazy-loading Dataset backed by a JSONL file.

   Scans the file once at construction to build a name→byte-offset index
   (only offsets are kept in RAM, not the protein data).  Each __getitem__
   seeks to the relevant line and parses only that entry.

   Compatible with num_workers > 0: the open file handle is excluded from
   pickling and re-opened lazily inside each worker process.

   JSONL format expected per line:
       {"name": "1abc.A", "seq": "ACDEF...", "coords": {"N": [[x,y,z],...],
        "CA": [...], "C": [...], "O": [...]}, ...}

   :param jsonl_path: Path to the JSONL file.
   :param names: List of entry names (e.g. "1abc.A") to include.
   :param max_seq_length: Sequences longer than this are truncated; shorter ones
                          are zero-padded to this length.


   .. py:method:: __del__() -> None


   .. py:method:: __getitem__(idx: int) -> dict


   .. py:method:: __getstate__() -> dict


   .. py:method:: __len__() -> int


   .. py:attribute:: jsonl_path


   .. py:attribute:: max_seq_length
      :value: 256



.. py:function:: make_data_loaders(cfg: train.train_config.TrainConfig, jsonl_path: str | pathlib.Path, splits_path: str | pathlib.Path, num_workers: int = 0, debug_run: bool = True) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]

   Build train, validation, and test DataLoaders from a JSONL file and a
   splits JSON.

   :param cfg: TrainConfig — batch_size and max_seq_length are read from
               cfg.loader.
   :param jsonl_path: Path to the JSONL protein dataset.
   :param splits_path: Path to a JSON file with keys "train", "validation", and
                       "test", each a list of entry names.
   :param num_workers: DataLoader worker processes (0 = main process only).

   :returns: (train_loader, val_loader, test_loader)



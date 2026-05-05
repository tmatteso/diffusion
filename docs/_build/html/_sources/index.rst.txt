pallatom
========

**pallatom** is a protein structure diffusion model implementing EDM-based
denoising over atomic coordinates, built on AlphaFold 3–style architecture
components.

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   autoapi/index

Overview
--------

The library is divided into four subpackages:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Package
     - Description
   * - :mod:`pallatom.architecture`
     - Core neural network modules — ``MainTrunk``, attention blocks,
       pair updates, template embedder, and loss functions.
   * - :mod:`pallatom.helpers`
     - Data loading (``ProteinDataset``), featurization (``Distogram``,
       ``ProteinBatch``), Kabsch alignment, and atom/residue constants.
   * - :mod:`pallatom.train`
     - Pydantic training configuration (``TrainingParams``, ``ModelParams``,
       ``NoiseScheduleParams``, …) and the training/evaluation loop.
   * - :mod:`pallatom.sample`
     - Pydantic sampling configuration and ``EDMPrecond`` — the
       EDM-compatible denoiser wrapper for inference.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

pallatom.helpers.compute_EDM_data_params
========================================

.. py:module:: pallatom.helpers.compute_EDM_data_params


Attributes
----------

.. autoapisummary::

   pallatom.helpers.compute_EDM_data_params.parser


Functions
---------

.. autoapisummary::

   pallatom.helpers.compute_EDM_data_params.compute_edm_noise_params
   pallatom.helpers.compute_EDM_data_params.compute_sigma_data


Module Contents
---------------

.. py:function:: compute_edm_noise_params(train_loader) -> dict

   Fit sigma_max, sigma_min, P_mean, P_std from the training coordinate distribution.

   - sigma_max : 99th-percentile per-atom distance from the CA centroid
   - sigma_min : max(1st-percentile pairwise CA distance, 2e-3)
   - P_mean    : midpoint of [log sigma_min, log sigma_max]
   - P_std     : half-width of that interval (±1 std spans the full range)


.. py:function:: compute_sigma_data(train_loader) -> float

   RMS displacement of valid atom positions from the CA centroid.


.. py:data:: parser


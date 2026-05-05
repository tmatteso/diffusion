sampling
========

.. py:module:: sampling

.. autoapi-nested-parse::

   EDM sampling from a trained MainTrunk denoising network.



Attributes
----------

.. autoapisummary::

   sampling.ATOM5_TO_ATOM37
   sampling.NATOM
   sampling.log
   sampling.parser


Classes
-------

.. autoapisummary::

   sampling.EDMPrecond
   sampling.EDMSampler


Functions
---------

.. autoapisummary::

   sampling.atom5_to_atom37
   sampling.build_sampling_context
   sampling.build_template_context


Module Contents
---------------

.. py:class:: EDMPrecond(model: architecture.main_trunk.MainTrunk, context: helpers.featurize.FeaturizedBatch, sigma_min: float = 0.002, sigma_max: float = 80.0)

   Bases: :py:obj:`torch.nn.Module`


   Wraps MainTrunk as an EDM-compatible denoiser D_θ(r_noisy, σ) → r_denoised.

   :param model:
   :type model: trained MainTrunk
   :param context: t_hat are replaced at every forward call
   :type context: FeaturizedBatch with static fields filled in; r_input and
   :param sigma_min:
   :type sigma_min: lower σ bound, used only to compute t_normalized
   :param sigma_max:
   :type sigma_max: upper σ bound, used only to compute t_normalized
   :param Initialize internal Module state:
   :param shared by both nn.Module and ScriptModule.:


   .. py:method:: forward(r_input: jaxtyping.Float[torch.Tensor, B N_atom 3], t_hat: float) -> jaxtyping.Float[torch.Tensor, B N_atom 3]


   .. py:attribute:: context


   .. py:attribute:: model


   .. py:attribute:: sigma_max
      :value: 80.0



   .. py:attribute:: sigma_min
      :value: 0.002



.. py:class:: EDMSampler(denoiser: EDMPrecond, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0, S_churn: float = 0.0, S_tmin: float = 0.0, S_tmax: float = float('inf'), S_noise: float = 1.003)

   Karras et al. 2022 deterministic (Heun) sampler.

   :param denoiser:
   :type denoiser: EDMPrecond wrapping a trained MainTrunk
   :param sigma_min:
   :type sigma_min: float  smallest noise level  (paper: 0.002)
   :param sigma_max:
   :type sigma_max: float  largest  noise level  (paper: 80.0)
   :param rho:
   :type rho: float  schedule exponent     (paper: 7.0)
   :param S_churn:
   :type S_churn: float  stochastic noise injected per step (0 = deterministic)
   :param S_tmin:
   :type S_tmin: float  only inject noise in [S_tmin, S_tmax]
   :param S_tmax:
   :type S_tmax: float
   :param S_noise:
   :type S_noise: float  scaling of injected noise


   .. py:method:: sample(shape: tuple[int, int, int], steps: int = 40, device: torch.device | str = 'cpu') -> jaxtyping.Float[torch.Tensor, B N_atom 3]

      shape : (B, N_atom, 3)  — batch of atom coordinate tensor shapes
      steps : number of ODE steps  (40 is usually plenty)



   .. py:attribute:: S_churn
      :value: 0.0



   .. py:attribute:: S_noise
      :value: 1.003



   .. py:attribute:: S_tmax


   .. py:attribute:: S_tmin
      :value: 0.0



   .. py:attribute:: denoiser


   .. py:attribute:: rho
      :value: 7.0



   .. py:attribute:: sigma_max
      :value: 80.0



   .. py:attribute:: sigma_min
      :value: 0.002



.. py:function:: atom5_to_atom37(coords_5: jaxtyping.Float[numpy.ndarray, N_res 5 3], mask_5: Optional[jaxtyping.Float[numpy.ndarray, N_res 5]] = None) -> tuple[jaxtyping.Float[numpy.ndarray, N_res 37 3], jaxtyping.Float[numpy.ndarray, N_res 37]]

   Map atom5 coordinates back into the full atom37 layout.

   :returns: * **x_37** (*(N_res, 37, 3)*)
             * **mask_37** (*(N_res, 37)*)


.. py:function:: build_sampling_context(N_res: int, index_embedding: torch.nn.Embedding, batch_size: int = 1, n_templ_bins: int = 38, n_atom_bins: int = 22, c_res_embed: int = 32, device: str = 'cpu') -> helpers.featurize.FeaturizedBatch

   Build the static context FeaturizedBatch for unconditional backbone generation.

   All batch items share identical static context (same reference conformer, same
   indices). r_input and t_hat are placeholder zeros; EDMPrecond.forward replaces
   them at each denoising step via dataclasses.replace.

   :param N_res:
   :type N_res: number of residues
   :param batch_size:
   :type batch_size: B — number of protein samples to generate in parallel


.. py:function:: build_template_context(N_res: int, index_embedding: torch.nn.Embedding, template_cb: jaxtyping.Float[torch.Tensor, N_res 3], template_mask: jaxtyping.Float[torch.Tensor, build_template_context.N_res], batch_size: int = 1, n_templ_bins: int = 39, n_atom_bins: int = 22, c_res_embed: int = 32, device: str = 'cpu') -> helpers.featurize.FeaturizedBatch

   Build the static context FeaturizedBatch for template-guided backbone generation.

   For full template conditioning pass template_mask=torch.ones(N_res).
   For partial template (motif scaffolding) set template_mask to 1 for residues
   with known coordinates and 0 for residues to be designed de novo.

   :param N_res:
   :type N_res: number of residues
   :param index_embedding:
   :type index_embedding: trained residue index embedding
   :param template_cb:
   :type template_cb: pseudo-Cβ coordinates, shape (N_res, 3)
   :param template_mask:
   :type template_mask: 1.0 = residue has valid template coords, 0.0 = unknown
   :param batch_size:
   :type batch_size: B — number of samples to generate in parallel
   :param n_templ_bins:
   :type n_templ_bins: total distogram bins including overflow (default 39 = 38 + 1)


.. py:data:: ATOM5_TO_ATOM37

.. py:data:: NATOM
   :value: 5


.. py:data:: log

.. py:data:: parser


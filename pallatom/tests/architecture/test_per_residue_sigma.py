"""Tests for per-residue noise levels (``t_hat`` of shape ``[B, N_res]``).

The noise level sigma is carried per residue through featurization, the
trunk's EDM preconditioning and the EDM loss weight. Training and sampling
still use one sigma per protein, so these tests check two things: that the
per-residue path reduces to the per-protein one when sigma is uniform, and
that it acts residue by residue when sigma is not.
"""

import dataclasses
import enum

import numpy as np
import pytest
import torch
from architecture.losses import atom_loss
from architecture.main_trunk import MainTrunk
from einops import rearrange, repeat
from helpers.atom_utils import Protein
from helpers.data import (
    Distogram,
    FeaturizedBatch,
    build_distogram_module,
    featurize_batch,
)
from helpers.useful_objects import manual_seed
from jaxtyping import Float, Int
from sample.sampling import NATOM, build_sampling_context
from train.train_config import (
    AtomDistogramParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
    TemplateDistogramParams,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# atom_loss with per-point weights
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_points() -> Float[torch.Tensor, "B N 3"]:
    """Two samples of four non-coplanar ground-truth points.

    A hard-coded data fixture rather than a factory: every ``atom_loss``
    test below uses the same structures and varies only the weights. The
    second sample is the first scaled by 2, so the two samples have
    different errors.

    Returns:
        Points at the origin and one unit along each axis, and the same
        points scaled by 2.
    """
    unit = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    return torch.stack([unit, 2.0 * unit])


@pytest.fixture
def perturbed_points(
    reference_points: Float[torch.Tensor, "B N 3"],
) -> Float[torch.Tensor, "B N 3"]:
    """Reference points moved non-rigidly, so Kabsch leaves an error.

    Args:
        reference_points: Ground-truth points to perturb.

    Returns:
        Each point moved by a different small offset.
    """
    offsets = torch.tensor(
        [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3], [0.1, 0.1, 0.0]],
    )
    return reference_points + offsets


@pytest.fixture
def point_mask() -> Float[torch.Tensor, "B N"]:
    """Every point of both samples is valid.

    Returns:
        A mask of ones.
    """
    return torch.ones(2, 4)


@pytest.fixture
def residue_types() -> Int[torch.Tensor, "B N_aa"]:
    """Two residues per sample, two points each, one polar and one not.

    Returns:
        Alanine (index 0, nonpolar) and asparagine (index 2, polar), in
        opposite orders for the two samples, so residue-type weights vary.
    """
    return torch.tensor([[0, 2], [2, 0]])


@pytest.fixture
def unit_point_weight() -> Float[torch.Tensor, "B N"]:
    """EDM weight of 1 on every point.

    Returns:
        A weight tensor of ones.
    """
    return torch.ones(2, 4)


@pytest.fixture
def unit_weight_loss(
    perturbed_points: Float[torch.Tensor, "B N 3"],
    reference_points: Float[torch.Tensor, "B N 3"],
    point_mask: Float[torch.Tensor, "B N"],
    residue_types: Int[torch.Tensor, "B N_aa"],
    unit_point_weight: Float[torch.Tensor, "B N"],
) -> Float[torch.Tensor, "B"]:
    """Per-sample ``atom_loss`` with a weight of 1 on every point.

    Args:
        perturbed_points: Predicted points.
        reference_points: Ground-truth points.
        point_mask: Valid-point mask.
        residue_types: Residue amino-acid indices.
        unit_point_weight: Weight of 1 on every point.

    Returns:
        The unweighted loss of each sample.
    """
    return atom_loss(
        perturbed_points,
        reference_points,
        point_mask,
        aa_indices=residue_types,
        lambda_sigma_weight=unit_point_weight,
    )


@dataclasses.dataclass(frozen=True)
class SampleScaleCase:
    """One EDM weight per sample, constant over the sample's points.

    Attributes:
        scales: The weight of each sample.
    """

    scales: tuple[float, float]


class SampleScale(enum.Enum):
    """Per-sample weights that must factor out of ``atom_loss``.

    Members:
        distinct: Different weights for the two samples.
        shared: The same weight for both samples.
        small: A weight below 1, as at high noise levels.
    """

    distinct = SampleScaleCase(scales=(0.5, 2.0))
    shared = SampleScaleCase(scales=(3.0, 3.0))
    small = SampleScaleCase(scales=(1.0, 0.25))


@pytest.fixture
def sample_scale(
    sample_scale_scenario: SampleScale,
) -> Float[torch.Tensor, "B"]:
    """The scenario's per-sample weights as a tensor.

    Args:
        sample_scale_scenario: Per-sample weights; the requesting test
            parametrizes it.

    Returns:
        One weight per sample.
    """
    return torch.tensor(sample_scale_scenario.value.scales)


@pytest.fixture
def per_sample_weight_loss(
    sample_scale: Float[torch.Tensor, "B"],
    perturbed_points: Float[torch.Tensor, "B N 3"],
    reference_points: Float[torch.Tensor, "B N 3"],
    point_mask: Float[torch.Tensor, "B N"],
    residue_types: Int[torch.Tensor, "B N_aa"],
) -> Float[torch.Tensor, "B"]:
    """``atom_loss`` with each sample's weight repeated over its points.

    Args:
        sample_scale: One weight per sample.
        perturbed_points: Predicted points.
        reference_points: Ground-truth points.
        point_mask: Valid-point mask.
        residue_types: Residue amino-acid indices.

    Returns:
        The weighted loss of each sample.
    """
    return atom_loss(
        perturbed_points,
        reference_points,
        point_mask,
        aa_indices=residue_types,
        lambda_sigma_weight=repeat(sample_scale, "b -> b n", n=4),
    )


@pytest.mark.parametrize(
    "sample_scale_scenario",
    list(SampleScale),
    ids=[case.name for case in SampleScale],
)
def test_atom_loss_constant_weight_factors_out_of_the_sum(
    per_sample_weight_loss: Float[torch.Tensor, "B"],
    sample_scale: Float[torch.Tensor, "B"],
    unit_weight_loss: Float[torch.Tensor, "B"],
) -> None:
    """A weight constant within a sample scales that sample's loss.

    Arrange: the same structures scored with a per-sample weight repeated
    over every point, and with a weight of 1.
    Act: compare the two losses.
    Assert: the weighted loss is the per-sample weight times the unit-weight
    loss. This is the per-sample form ``main`` used before weights became
    per point, so uniform-sigma training is unchanged.

    Args:
        per_sample_weight_loss: Loss with the repeated per-sample weight.
        sample_scale: The per-sample weights.
        unit_weight_loss: Loss with a weight of 1.
    """
    torch.testing.assert_close(
        per_sample_weight_loss,
        sample_scale * unit_weight_loss,
    )


@dataclasses.dataclass(frozen=True)
class WeightPairCase:
    """Two per-point weight tables for one batch of two samples.

    Attributes:
        first: Weight of each point in the first table.
        second: Weight of each point in the second table.
    """

    first: tuple[tuple[float, ...], ...]
    second: tuple[tuple[float, ...], ...]


class WeightPair(enum.Enum):
    """Pairs of non-uniform per-point weights.

    Members:
        disjoint: The two tables weight different points.
        overlapping: The two tables share some weighted points.
    """

    disjoint = WeightPairCase(((1, 0, 0, 2), (0, 1, 1, 0)), ((0, 3, 1, 0),) * 2)
    overlapping = WeightPairCase(((1, 1, 0, 0),) * 2, ((0, 2, 2, 0),) * 2)


@pytest.fixture
def fixed_alignment_weight(
    weight_pair_scenario: WeightPair,
) -> Float[torch.Tensor, "B N"]:
    """Unit Kabsch weight shared by every loss in a linearity check.

    ``atom_loss`` aligns by its loss weight by default, so the alignment
    would change between the two tables and their sum. Linearity in the
    per-point weight holds at a fixed alignment, which is how
    ``noise_group_shares`` uses it, so all three losses take this one.

    Args:
        weight_pair_scenario: Weight tables, used only for their shape; the
            requesting test parametrizes it.

    Returns:
        Float tensor of ones, shaped like the weight tables.
    """
    return torch.ones_like(
        torch.tensor(weight_pair_scenario.value.first, dtype=torch.float32),
    )


@pytest.fixture
def first_weight_loss(
    weight_pair_scenario: WeightPair,
    perturbed_points: Float[torch.Tensor, "B N 3"],
    reference_points: Float[torch.Tensor, "B N 3"],
    point_mask: Float[torch.Tensor, "B N"],
    residue_types: Int[torch.Tensor, "B N_aa"],
    fixed_alignment_weight: Float[torch.Tensor, "B N"],
) -> Float[torch.Tensor, "B"]:
    """``atom_loss`` with the scenario's first weight table.

    Args:
        weight_pair_scenario: Weight tables; the requesting test
            parametrizes it.
        perturbed_points: Predicted points.
        reference_points: Ground-truth points.
        point_mask: Valid-point mask.
        residue_types: Residue amino-acid indices.
        fixed_alignment_weight: Kabsch weight shared by all three losses.

    Returns:
        The loss of each sample under the first table.
    """
    return atom_loss(
        perturbed_points,
        reference_points,
        point_mask,
        aa_indices=residue_types,
        alignment_lambda=fixed_alignment_weight,
        lambda_sigma_weight=torch.tensor(
            weight_pair_scenario.value.first,
            dtype=torch.float32,
        ),
    )


@pytest.fixture
def second_weight_loss(
    weight_pair_scenario: WeightPair,
    perturbed_points: Float[torch.Tensor, "B N 3"],
    reference_points: Float[torch.Tensor, "B N 3"],
    point_mask: Float[torch.Tensor, "B N"],
    residue_types: Int[torch.Tensor, "B N_aa"],
    fixed_alignment_weight: Float[torch.Tensor, "B N"],
) -> Float[torch.Tensor, "B"]:
    """``atom_loss`` with the scenario's second weight table.

    Args:
        weight_pair_scenario: Weight tables; the requesting test
            parametrizes it.
        perturbed_points: Predicted points.
        reference_points: Ground-truth points.
        point_mask: Valid-point mask.
        residue_types: Residue amino-acid indices.
        fixed_alignment_weight: Kabsch weight shared by all three losses.

    Returns:
        The loss of each sample under the second table.
    """
    return atom_loss(
        perturbed_points,
        reference_points,
        point_mask,
        aa_indices=residue_types,
        alignment_lambda=fixed_alignment_weight,
        lambda_sigma_weight=torch.tensor(
            weight_pair_scenario.value.second,
            dtype=torch.float32,
        ),
    )


@pytest.fixture
def summed_weight_loss(
    weight_pair_scenario: WeightPair,
    perturbed_points: Float[torch.Tensor, "B N 3"],
    reference_points: Float[torch.Tensor, "B N 3"],
    point_mask: Float[torch.Tensor, "B N"],
    residue_types: Int[torch.Tensor, "B N_aa"],
    fixed_alignment_weight: Float[torch.Tensor, "B N"],
) -> Float[torch.Tensor, "B"]:
    """``atom_loss`` with the sum of the scenario's two weight tables.

    Args:
        weight_pair_scenario: Weight tables; the requesting test
            parametrizes it.
        perturbed_points: Predicted points.
        reference_points: Ground-truth points.
        point_mask: Valid-point mask.
        residue_types: Residue amino-acid indices.
        fixed_alignment_weight: Kabsch weight shared by all three losses.

    Returns:
        The loss of each sample under the summed table.
    """
    case = weight_pair_scenario.value
    return atom_loss(
        perturbed_points,
        reference_points,
        point_mask,
        aa_indices=residue_types,
        alignment_lambda=fixed_alignment_weight,
        lambda_sigma_weight=torch.tensor(case.first, dtype=torch.float32)
        + torch.tensor(case.second, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "weight_pair_scenario",
    list(WeightPair),
    ids=[case.name for case in WeightPair],
)
def test_atom_loss_is_linear_in_per_point_weight(
    summed_weight_loss: Float[torch.Tensor, "B"],
    first_weight_loss: Float[torch.Tensor, "B"],
    second_weight_loss: Float[torch.Tensor, "B"],
) -> None:
    """The loss is additive in the per-point weights.

    Arrange: two non-uniform per-point weight tables.
    Act: score the same structures with each table and with their sum.
    Assert: at one fixed alignment, the summed-table loss is the sum of the
    two losses. So each
    weight multiplies its own point's squared error inside the weighted
    mean, which is what lets residues at different noise levels carry
    different weights in one sample.

    Args:
        summed_weight_loss: Loss under the summed table.
        first_weight_loss: Loss under the first table.
        second_weight_loss: Loss under the second table.
    """
    torch.testing.assert_close(
        summed_weight_loss,
        first_weight_loss + second_weight_loss,
    )


# ---------------------------------------------------------------------------
# MainTrunk per-atom EDM preconditioning
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CleanResidueCase:
    """Which residue of a four-residue protein is nearly noise-free.

    Attributes:
        index: Position of the near-clean residue.
        tolerance: Largest coordinate change the near-clean residue may
            show, and smallest change every other residue must show.
    """

    index: int
    tolerance: float


class CleanResidue(enum.Enum):
    """Position of the one residue at a tiny noise level.

    Members:
        first: The first residue of the chain.
        last: The last residue of the chain.
    """

    first = CleanResidueCase(index=0, tolerance=1e-2)
    last = CleanResidueCase(index=3, tolerance=1e-2)


@pytest.fixture
def template_distogram() -> Distogram:
    """Template distogram module built from the library's default params.

    Returns:
        The self-conditioning template distogram used in training.
    """
    return build_distogram_module(TemplateDistogramParams())


@pytest.fixture
def atom_distogram() -> Distogram:
    """Atom distogram module built from the library's default params.

    Returns:
        The sparse atom-pair distogram used in training.
    """
    return build_distogram_module(AtomDistogramParams())


@pytest.fixture
def residue_distogram() -> Distogram:
    """Residue distogram module built from the library's default params.

    Returns:
        The residue distogram whose bins are the training target.
    """
    return build_distogram_module(ResidueDistogramParams())


@pytest.fixture
def sampling_context(
    atom_distogram: Distogram,
    template_distogram: Distogram,
) -> FeaturizedBatch:
    """All-alanine four-residue batch whose input is unit-scale noise.

    Args:
        atom_distogram: Atom distogram module used to build the context.
        template_distogram: Template distogram module used to build the
            context.

    Returns:
        A batch-of-one context on CPU.
    """
    _ = manual_seed(0)
    with torch.no_grad():
        return build_sampling_context(
            pdb_file_path=None,
            atom_distogram_fn=atom_distogram,
            templ_distogram_fn=template_distogram,
            residue_number=4,
            batch_size=1,
            device="cpu",
        )


@pytest.fixture
def small_trunk() -> MainTrunk:
    """Untrained two-unit MainTrunk in eval mode, seeded for repeatability.

    Returns:
        A MainTrunk built from default params apart from ``K_unit``.
    """
    _ = manual_seed(0)
    return MainTrunk(
        model_params=ModelParams(K_unit=2),
        template_distogram_params=TemplateDistogramParams(),
        residue_distogram_params=ResidueDistogramParams(),
        atom_distogram_params=AtomDistogramParams(),
        noise_params=NoiseScheduleParams(),
    ).eval()


@pytest.fixture
def mixed_sigma_batch(
    clean_residue_scenario: CleanResidue,
    sampling_context: FeaturizedBatch,
) -> FeaturizedBatch:
    """Batch with one residue at sigma 1e-4 and the rest at sigma 10.

    Args:
        clean_residue_scenario: Which residue is near-clean; the requesting
            test parametrizes it.
        sampling_context: Batch to give per-residue noise levels.

    Returns:
        The context with a non-uniform per-residue ``t_hat``.
    """
    t_hat = torch.full((1, 4), 10.0)
    t_hat[0, clean_residue_scenario.value.index] = 1e-4
    return dataclasses.replace(sampling_context, t_hat=t_hat)


@pytest.fixture
def denoising_change(
    small_trunk: MainTrunk,
    mixed_sigma_batch: FeaturizedBatch,
) -> Float[torch.Tensor, "N_res"]:
    """Largest change from input to denoised coordinates, per residue.

    Args:
        small_trunk: Model to run.
        mixed_sigma_batch: Batch with per-residue noise levels.

    Returns:
        For each residue, the largest absolute coordinate difference between
        the denoised output and the noisy input over the residue's atoms.
    """
    with torch.no_grad():
        denoised = small_trunk(mixed_sigma_batch).r_denoised
    change = (denoised - mixed_sigma_batch.r_gt_noised).abs()
    per_atom: Float[torch.Tensor, "N_atom"] = change[0].amax(dim=-1)
    return rearrange(per_atom, "(n a) -> n a", a=NATOM).amax(dim=-1)


@pytest.mark.parametrize(
    "clean_residue_scenario",
    list(CleanResidue),
    ids=[case.name for case in CleanResidue],
)
def test_trunk_applies_edm_skip_per_residue(
    denoising_change: Float[torch.Tensor, "N_res"],
    clean_residue_scenario: CleanResidue,
) -> None:
    """A near-clean residue passes through; noisy residues are rewritten.

    Arrange: a four-residue batch with one residue at sigma 1e-4 and the
    others at sigma 10.
    Act: one trunk forward pass.
    Assert: at sigma 1e-4, EDM's ``c_skip`` is 1 to within 4e-11 and
    ``c_out`` is about 1e-4, so that residue's atoms come out within 1e-2 of
    the input. At sigma 10, ``c_skip`` is about 0.72, so the other residues
    move by more than 1e-2. Both hold only if ``c_skip`` and ``c_out`` are
    applied per atom from the atom's own residue.

    Args:
        denoising_change: Largest input-to-output change per residue.
        clean_residue_scenario: Which residue is near-clean; parametrized.
    """
    case = clean_residue_scenario.value
    clean = torch.zeros(4, dtype=torch.bool)
    clean[case.index] = True
    assert bool((denoising_change[clean] < case.tolerance).all())
    assert bool((denoising_change[~clean] > case.tolerance).all())


# ---------------------------------------------------------------------------
# Featurization
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BatchLengthsCase:
    """Residue counts of the proteins in one featurized batch.

    Attributes:
        lengths: Number of residues of each protein.
    """

    lengths: tuple[int, int]


class BatchLengths(enum.Enum):
    """Protein lengths for a two-protein batch.

    Members:
        equal: Both proteins have four residues; nothing is padded.
        padded: Five and three residues; the second is padded by two.
    """

    equal = BatchLengthsCase(lengths=(4, 4))
    padded = BatchLengthsCase(lengths=(5, 3))


@pytest.fixture
def length_batch(
    batch_lengths_scenario: BatchLengths,
    template_distogram: Distogram,
    residue_distogram: Distogram,
    atom_distogram: Distogram,
) -> FeaturizedBatch:
    """Featurized batch of two all-alanine proteins of the given lengths.

    Atom positions are an evenly spaced ramp, so the input is deterministic
    apart from the sampled noise.

    Args:
        batch_lengths_scenario: Protein lengths; the requesting test
            parametrizes it.
        template_distogram: Template distogram module.
        residue_distogram: Residue distogram module.
        atom_distogram: Atom distogram module.

    Returns:
        The featurized batch.
    """
    proteins = [
        Protein(
            atom_positions=rearrange(
                0.5 * np.arange(n_res * 37 * 3, dtype=np.float64),
                "(n a d) -> n a d",
                a=37,
                d=3,
            ),
            aatype=np.zeros(n_res, dtype=np.intp),
            atom_mask=np.ones((n_res, 37), dtype=np.float64),
            residue_index=np.arange(n_res, dtype=np.intp),
            chain_index=np.zeros(n_res, dtype=np.intp),
            b_factors=np.zeros((n_res, 37), dtype=np.float64),
        )
        for n_res in batch_lengths_scenario.value.lengths
    ]
    _ = manual_seed(0)
    return featurize_batch(
        proteins,
        TrainConfig(),
        template_distogram,
        residue_distogram,
        atom_distogram,
    )


@pytest.mark.parametrize(
    "batch_lengths_scenario",
    list(BatchLengths),
    ids=[case.name for case in BatchLengths],
)
def test_featurization_shares_one_sigma_across_a_protein(
    length_batch: FeaturizedBatch,
) -> None:
    """Training gives every residue of a protein the same noise level.

    Arrange: two proteins featurized together, with and without padding.
    Act: read the batch's per-residue ``t_hat``.
    Assert: within each protein, padding included, every residue has the
    protein's single sampled sigma, and that sigma is positive. This keeps
    training identical to the per-protein schedule until a later change
    deliberately varies sigma between residues.

    Args:
        length_batch: The featurized batch.
    """
    t_hat = length_batch.t_hat
    assert bool((t_hat == t_hat[:, :1]).all())
    assert bool((t_hat > 0.0).all())

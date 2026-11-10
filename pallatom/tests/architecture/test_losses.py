"""Tests for diffusion model loss functions.

Covers atom_loss (zero on rigid copies, invariance to rigid motion of the
target, residue-polarity weighting, masking, the EDM weight, gradients),
med_loss against its closed form, smooth_lddt_loss against its closed form
with and without masking, pairwise_dist on a 3-4-5 triangle, both distogram
losses on uniform and perfect logits with one-hot or index targets, and
seq_ce_loss's handling of padding and the unknown token, plus every
function's shape contract.

Uniform logits give a cross-entropy of exactly ``log(n_classes)``, which
anchors most expected values. Coordinates come from
``tests.patterns.chiral_cloud``.
"""

import abc
import dataclasses
import enum
import math
from typing import override

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from architecture.errors import (
    AtomResidueCountMismatchError,
    BlockCountMismatchError,
    NoDenoiserBlockError,
)
from architecture.losses import (
    NONPOLAR_RESIDUE_WEIGHT,
    PAD_TOKEN,
    POLAR_RESIDUE_WEIGHT,
    POLAR_RESIDUES,
    RESIDUE_TYPE_WEIGHTS,
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    med_loss,
    pairwise_dist,
    residue_type_weight,
    seq_ce_loss,
    smooth_lddt_loss,
)
from einops import einsum, rearrange, repeat
from helpers.alignment import kabsch_align
from helpers.atom_utils import (
    RESTYPE_NUM_NO_X,
    RESTYPES_NO_X,
    Protein,
    restype_order,
)
from helpers.data import (
    FeaturizedBatch,
    build_distogram_module,
    featurize_single_item,
)
from jaxtyping import Bool, Float, Int, TypeCheckError
from tests.patterns import chiral_cloud, sine_pattern
from train.train_config import (
    AtomDistogramParams,
    LossParams,
    ResidueDistogramParams,
    TemplateDistogramParams,
    TrainConfig,
)

type Coords = Float[torch.Tensor, "B N 3"]


@dataclasses.dataclass(frozen=True)
class LossDims:
    """Sizes shared by the structures, logits and targets in this file.

    Attributes:
        batch: Batch size ``B``.
        n_res: Residues per structure.
        atoms_per_res: Atoms per residue.
        n_bins: Distogram bins.
        k: Sparse neighbours per atom for the atom distogram; smaller than
            the atom count so the two axes cannot be confused.
    """

    batch: int
    n_res: int
    atoms_per_res: int
    n_bins: int
    k: int

    @property
    def n_atom(self) -> int:
        """Atoms per structure.

        Returns:
            ``n_res * atoms_per_res``.
        """
        return self.n_res * self.atoms_per_res


@dataclasses.dataclass(frozen=True)
class RigidCase:
    """A rigid motion ``x -> x R + t``, with ``R`` a proper rotation.

    Attributes:
        rotation: Integer rotation matrix ``R``, row by row.
        translation: Offset ``t``.
    """

    rotation: tuple[tuple[int, int, int], ...]
    translation: tuple[float, float, float]

    def apply(self, x: Coords) -> Coords:
        """Rotate then translate every structure in ``x``.

        Args:
            x: Batched coordinates to move.

        Returns:
            ``x R + t``.
        """
        rotation = torch.tensor(self.rotation, dtype=torch.float32)
        moved = einsum(x, rotation, "b n d, d e -> b n e")
        return moved + torch.tensor(self.translation)


@dataclasses.dataclass(frozen=True)
class TargetCase:
    """How distogram targets are encoded.

    Attributes:
        one_hot: One-hot float targets when true, integer bin indices when
            false.
    """

    one_hot: bool


class TargetForm(enum.Enum):
    """Target encodings both distogram losses accept.

    Members:
        index: Integer bin per pair.
        one_hot: One-hot float vector over the bins per pair.
    """

    index = TargetCase(one_hot=False)
    one_hot = TargetCase(one_hot=True)


@dataclasses.dataclass(frozen=True)
class MaskCase:
    """Whether a distogram loss is given a mask, and whether that is buggy.

    Attributes:
        masked: Pass a mask when true, ``None`` when false.
        divisor_bug: The case hits the known ``distogram_loss_atom``
            normalisation bug, so its fixture marks it strict ``xfail``.
    """

    masked: bool
    divisor_bug: bool


class ResidueMask(enum.Enum):
    """Residue masks for ``distogram_loss_residue``.

    Members:
        unmasked: No mask.
        masked: The last residue is masked out.
    """

    unmasked = MaskCase(masked=False, divisor_bug=False)
    masked = MaskCase(masked=True, divisor_bug=False)


class AtomMask(enum.Enum):
    """Slot masks for ``distogram_loss_atom``.

    Members:
        masked: An all-valid slot mask.
        unmasked: No mask. Fails today: without a mask the loss divides
            by ``q.shape[-2] ** 2 = K**2`` instead of ``N_atom * K``.
    """

    masked = MaskCase(masked=True, divisor_bug=False)
    unmasked = MaskCase(masked=False, divisor_bug=True)


class RigidScenario(enum.Enum):
    """Rigid motions the Kabsch-aligned atom loss must not see.

    Members:
        identity: No motion.
        translate: Pure shift by ``(10, 5, -3)``.
        rotate_z: Quarter turn about the z axis.
        screw: Third of a turn about ``(1, 1, 1)`` plus the shift.
    """

    identity = RigidCase(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (0, 0, 0))
    translate = RigidCase(((1, 0, 0), (0, 1, 0), (0, 0, 1)), (10, 5, -3))
    rotate_z = RigidCase(((0, -1, 0), (1, 0, 0), (0, 0, 1)), (0, 0, 0))
    screw = RigidCase(((0, 0, 1), (1, 0, 0), (0, 1, 0)), (10, 5, -3))


@dataclasses.dataclass(frozen=True)
class MedCase:
    """Hyperparameters for a ``med_loss`` call over identical blocks.

    Attributes:
        lam: Structure-loss weight.
        alpha_0: Sequence-loss weight.
        gamma: Per-block decay.
        n_blocks: Number of decoder blocks ``K``.
    """

    lam: float
    alpha_0: float
    gamma: float
    n_blocks: int


class MedScenario(enum.Enum):
    """``med_loss`` settings that weight its two terms differently.

    ``LossParams`` requires ``lam > 0`` and ``gamma < 1``, so neither term
    can be switched off entirely and blocks always decay.

    Members:
        structure_only: ``alpha_0 = 0`` leaves only the structure term.
        sequence_heavy: ``lam = 0.01`` makes the sequence term dominate.
        both_decayed: Both terms, three blocks decayed by 0.5.
        slow_decay: ``gamma = 0.99`` over four blocks, almost uniform.
    """

    structure_only = MedCase(2.0, 0.0, 0.9, 2)
    sequence_heavy = MedCase(0.01, 2.0, 0.9, 2)
    both_decayed = MedCase(1.0, 0.1, 0.5, 3)
    slow_decay = MedCase(1.0, 1.0, 0.99, 4)


@dataclasses.dataclass(frozen=True)
class MalformedInputs:
    """Tensors with one wrong axis, and valid partners for them.

    Attributes:
        coords4: ``(B, N, 4)`` coordinates.
        coords3: ``(B, N, 3)`` coordinates.
        flat: ``(N, N)`` logits with no bin axis.
        idx: ``(N, N)`` integer targets.
        seq: ``(B, N, n_amino)`` sequence logits.
    """

    coords4: Float[torch.Tensor, "B N 4"]
    coords3: Coords
    flat: Float[torch.Tensor, "N N"]
    idx: Int[torch.Tensor, "N N"]
    seq: Float[torch.Tensor, "B N n_amino"]


@dataclasses.dataclass(frozen=True)
class BadCall(abc.ABC):
    """A loss call with one input of the wrong shape."""

    @abc.abstractmethod
    def call(self, bad: MalformedInputs) -> object:
        """Call one loss on inputs drawn from ``bad``.

        Args:
            bad: Valid and malformed tensors to pass.

        Returns:
            The loss's result; the test expects it to raise.
        """


@dataclasses.dataclass(frozen=True)
class PairwiseCall(BadCall):
    """Calls ``pairwise_dist`` on 4-coordinate atoms."""

    @override
    def call(self, bad: MalformedInputs) -> object:
        """Make the malformed call.

        Args:
            bad: Valid and malformed tensors to pass.

        Returns:
            The pairwise distances.
        """
        return pairwise_dist(bad.coords4)


@dataclasses.dataclass(frozen=True)
class SmoothLddtCall(BadCall):
    """Calls ``smooth_lddt_loss`` on 4-coordinate predictions."""

    @override
    def call(self, bad: MalformedInputs) -> object:
        """Make the malformed call.

        Args:
            bad: Valid and malformed tensors to pass.

        Returns:
            The loss.
        """
        return smooth_lddt_loss(bad.coords4, bad.coords3)


@dataclasses.dataclass(frozen=True)
class ResidueDistoCall(BadCall):
    """Calls ``distogram_loss_residue`` on logits with no bins."""

    @override
    def call(self, bad: MalformedInputs) -> object:
        """Make the malformed call.

        Args:
            bad: Valid and malformed tensors to pass.

        Returns:
            The loss.
        """
        return distogram_loss_residue(bad.flat, bad.idx)


@dataclasses.dataclass(frozen=True)
class AtomDistoCall(BadCall):
    """Calls ``distogram_loss_atom`` on logits with no bins."""

    @override
    def call(self, bad: MalformedInputs) -> object:
        """Make the malformed call.

        Args:
            bad: Valid and malformed tensors to pass.

        Returns:
            The loss.
        """
        return distogram_loss_atom(bad.flat, bad.idx)


@dataclasses.dataclass(frozen=True)
class SeqCeCall(BadCall):
    """Calls ``seq_ce_loss`` with 1-D targets for 3-D logits."""

    @override
    def call(self, bad: MalformedInputs) -> object:
        """Make the malformed call.

        Args:
            bad: Valid and malformed tensors to pass.

        Returns:
            The loss.
        """
        return seq_ce_loss(bad.seq, bad.idx[0])


class MalformedScenario(enum.Enum):
    """Calls each loss's jaxtyping contract must reject.

    Members:
        pairwise: ``pairwise_dist`` on 4-coordinate atoms.
        smooth_lddt: ``smooth_lddt_loss`` on 4-coordinate predictions.
        residue_disto: ``distogram_loss_residue`` on logits with no bins.
        atom_disto: ``distogram_loss_atom`` on logits with no bins.
        seq_ce: ``seq_ce_loss`` with 1-D targets for 3-D logits.
    """

    pairwise = PairwiseCall()
    smooth_lddt = SmoothLddtCall()
    residue_disto = ResidueDistoCall()
    atom_disto = AtomDistoCall()
    seq_ce = SeqCeCall()


@pytest.fixture
def dims() -> LossDims:
    """Return the sizes every structure, logit and target is built at.

    Returns:
        Small sizes with ``k < n_atom``.
    """
    return LossDims(batch=2, n_res=4, atoms_per_res=3, n_bins=8, k=5)


@pytest.fixture
def gt(dims: LossDims) -> Coords:
    """Return non-planar ground-truth structures spanning about 10 angstroms.

    Args:
        dims: Structure sizes.

    Returns:
        ``(B, N_atom, 3)`` coordinates.
    """
    return 5 * chiral_cloud(dims.batch, dims.n_atom)


@pytest.fixture
def perturbed(gt: Coords) -> Coords:
    """Return ``gt`` plus a fixed non-rigid perturbation of about 1 angstrom.

    The perturbation is a chiral cloud offset by one structure in the
    sequence, so it is unrelated to ``gt``.

    Args:
        gt: Ground truth.

    Returns:
        A prediction that no rigid motion maps onto ``gt``.
    """
    b, n, _ = gt.shape
    return gt + chiral_cloud(b + 1, n)[1:]


@pytest.fixture
def aa_indices(dims: LossDims) -> Int[torch.Tensor, "B N_res"]:
    """Return residue types ``(7 r + b) % 20`` for residue ``r`` of item ``b``.

    Args:
        dims: Supplies the batch and residue counts.

    Returns:
        Canonical amino-acid indices, mixing polar and nonpolar types.
    """
    r = rearrange(torch.arange(dims.n_res), "r -> 1 r")
    b = rearrange(torch.arange(dims.batch), "b -> b 1")
    return (7 * r + b) % RESTYPE_NUM_NO_X


@pytest.fixture
def ones_weight(dims: LossDims) -> Float[torch.Tensor, " B"]:
    """Return an EDM noise weight of 1 for every batch item.

    Args:
        dims: Supplies the batch size.

    Returns:
        ``(B,)`` ones.
    """
    return torch.ones(dims.batch)


@pytest.fixture
def residue_target(
    target_form: TargetForm,
    dims: LossDims,
    residue_bins: Int[torch.Tensor, "B N_res N_res"],
) -> (
    Int[torch.Tensor, "B N_res N_res"] | Float[torch.Tensor, "B N_res N_res n"]
):
    """Return ``residue_bins`` in the scenario's encoding.

    Args:
        target_form: Index or one-hot; the requesting test parametrizes
            it.
        dims: Supplies ``n_bins``.
        residue_bins: Target bins.

    Returns:
        The bins themselves, or their one-hot encoding.
    """
    if not target_form.value.one_hot:
        return residue_bins
    return F.one_hot(residue_bins, dims.n_bins).float()


@pytest.fixture
def atom_target(
    target_form: TargetForm,
    dims: LossDims,
    atom_bins: Int[torch.Tensor, "B N_atom K"],
) -> Int[torch.Tensor, "B N_atom K"] | Float[torch.Tensor, "B N_atom K n"]:
    """Return ``atom_bins`` in the scenario's encoding.

    Args:
        target_form: Index or one-hot; the requesting test parametrizes
            it.
        dims: Supplies ``n_bins``.
        atom_bins: Target bins.

    Returns:
        The bins themselves, or their one-hot encoding.
    """
    if not target_form.value.one_hot:
        return atom_bins
    return F.one_hot(atom_bins, dims.n_bins).float()


@pytest.fixture
def residue_mask(
    residue_mask_case: ResidueMask,
    dims: LossDims,
) -> Float[torch.Tensor, "B N_res"] | None:
    """Return the residue mask that ``residue_mask_case`` asks for.

    Args:
        residue_mask_case: Whether to mask; the requesting test
            parametrizes it.
        dims: Supplies the batch and residue counts.

    Returns:
        A mask hiding the last residue, or ``None``.
    """
    if not residue_mask_case.value.masked:
        return None
    mask = torch.ones(dims.batch, dims.n_res)
    mask[:, -1] = 0
    return mask


@pytest.fixture
def atom_mask(
    request: pytest.FixtureRequest,
    atom_mask_case: AtomMask,
    atom_bins: Int[torch.Tensor, "B N_atom K"],
) -> Bool[torch.Tensor, "B N_atom K"] | None:
    """Return the slot mask that ``atom_mask_case`` asks for.

    A case flagged ``divisor_bug`` gets a strict ``xfail`` marker, so it
    must fail today and the marker must come off once the loss is fixed.
    Marks cannot ride on enum members, so the fixture applies it.

    Args:
        request: Pytest request that receives the ``xfail`` marker.
        atom_mask_case: Whether to mask, and whether the case hits the
            known divisor bug; the requesting test parametrizes it.
        atom_bins: Supplies the slot layout.

    Returns:
        An all-valid slot mask, or ``None``.
    """
    if atom_mask_case.value.divisor_bug:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="unmasked distogram_loss_atom divides by K**2",
            ),
        )
    if not atom_mask_case.value.masked:
        return None
    return torch.ones(atom_bins.shape, dtype=torch.bool)


@pytest.fixture
def batch(
    dims: LossDims,
    gt: Coords,
    aa_indices: Int[torch.Tensor, "B N_res"],
) -> FeaturizedBatch:
    """Build a FeaturizedBatch carrying ``gt`` and ``aa_indices``.

    ``med_loss`` reads only ``r_gt``, ``atom5_mask`` and ``aa_indices``;
    the other fields are zero placeholders of the right shapes.

    Args:
        dims: Batch sizes.
        gt: Ground-truth coordinates.
        aa_indices: Ground-truth residue types.

    Returns:
        The batch.
    """
    b, r, n, k = dims.batch, dims.n_res, dims.n_atom, dims.k
    zeros_rr = torch.zeros(b, r, r, dtype=torch.long)
    return FeaturizedBatch(
        ref_pos=gt.clone(),
        ref_element=torch.zeros(b, n, 4),
        ref_space_uid=torch.zeros(b, n, dtype=torch.long),
        gt_res_distogram_indices=zeros_rr,
        noised_res_distogram=torch.zeros(b, r, r, dims.n_bins),
        f_pseudo_beta_mask=torch.ones(b, r, dtype=torch.long),
        f_residue_idx=repeat(torch.arange(r), "r -> b r", b=b),
        r_gt=gt,
        r_gt_noised=gt.clone(),
        atom5_mask=torch.ones(b, n, dtype=torch.bool),
        aa_indices=aa_indices,
        t_hat=torch.ones(b),
        t_normalized=torch.zeros(b, r, r),
        tok_idx=repeat(
            torch.repeat_interleave(torch.arange(r), dims.atoms_per_res),
            "n -> b n",
            b=b,
        ),
        center_uid=torch.zeros(b, n, dtype=torch.long),
        gt_atom_distogram_sparse=torch.zeros(b, n, k, dtype=torch.long),
        gt_atom_distogram_mask_sparse=torch.ones(b, n, k, dtype=torch.bool),
    )


@pytest.fixture
def residue_bins(dims: LossDims) -> Int[torch.Tensor, "B N_res N_res"]:
    """Return residue distogram targets ``(i + j + b) % n_bins``.

    Args:
        dims: Supplies the sizes.

    Returns:
        Integer bin per residue pair.
    """
    i = rearrange(torch.arange(dims.n_res), "i -> 1 i 1")
    j = rearrange(torch.arange(dims.n_res), "j -> 1 1 j")
    b = rearrange(torch.arange(dims.batch), "b -> b 1 1")
    return (i + j + b) % dims.n_bins


@pytest.fixture
def atom_bins(dims: LossDims) -> Int[torch.Tensor, "B N_atom K"]:
    """Return sparse atom distogram targets ``(l + k + b) % n_bins``.

    Args:
        dims: Supplies the sizes; the slot axis has ``k < n_atom`` entries.

    Returns:
        Integer bin per atom and neighbour slot.
    """
    atom = rearrange(torch.arange(dims.n_atom), "l -> 1 l 1")
    slot = rearrange(torch.arange(dims.k), "k -> 1 1 k")
    b = rearrange(torch.arange(dims.batch), "b -> b 1 1")
    return (atom + slot + b) % dims.n_bins


@pytest.fixture
def malformed_inputs(dims: LossDims, gt: Coords) -> MalformedInputs:
    """Build the inputs the malformed-shape scenarios cut from.

    Args:
        dims: Sizes.
        gt: Valid coordinates.

    Returns:
        One wrong-axis tensor per contract, and valid partners.
    """
    n = dims.n_atom
    return MalformedInputs(
        coords4=torch.cat([gt, gt[..., :1]], dim=-1),
        coords3=gt,
        flat=torch.zeros(n, n),
        idx=torch.zeros(n, n, dtype=torch.long),
        seq=torch.zeros(dims.batch, n, RESTYPE_NUM_NO_X),
    )


@pytest.fixture
def canonical_residues() -> Int[torch.Tensor, "1 n_amino"]:
    """Return every canonical residue index, 0 to 19, in one row.

    Returns:
        ``(1, 20)`` indices in ``RESTYPES_NO_X`` order.
    """
    return rearrange(torch.arange(RESTYPE_NUM_NO_X), "n -> 1 n")


@pytest.fixture
def triangle() -> Float[torch.Tensor, "3 3"]:
    """Return a 3-4-5 right triangle's corners.

    Returns:
        Points ``(0,0,0)``, ``(3,0,0)`` and ``(0,4,0)``.
    """
    return torch.tensor([[0.0, 0, 0], [3, 0, 0], [0, 4, 0]])


@pytest.fixture
def lddt_floor() -> Float[torch.Tensor, ""]:
    """Return the smooth-lDDT loss when every distance is exact.

    Each pair then scores the mean of ``sigmoid(t)`` over the thresholds
    ``t`` in ``(0.5, 1, 2, 4)``, so the loss is one minus that, about
    0.168.

    Returns:
        The scalar floor.
    """
    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0])
    return 1 - torch.sigmoid(thresholds).mean()


@pytest.mark.parametrize(
    "rigid_scenario",
    list(RigidScenario),
    ids=[case.name for case in RigidScenario],
)
def test_atom_loss_is_zero_for_a_rigid_copy(
    rigid_scenario: RigidScenario,
    gt: Coords,
    aa_indices: Int[torch.Tensor, "B N_res"],
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """Predicting a rigidly moved ground truth costs nothing after alignment.

    Args:
        rigid_scenario: Motion applied to the prediction.
        gt: Ground truth.
        aa_indices: Residue types.
        ones_weight: EDM weight of 1.
    """
    pred = rigid_scenario.value.apply(gt)
    loss = atom_loss(
        pred,
        gt,
        aa_indices=aa_indices,
        lambda_sigma_weight=ones_weight,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    "rigid_scenario",
    list(RigidScenario),
    ids=[case.name for case in RigidScenario],
)
def test_atom_loss_ignores_rigid_motion_of_the_target(
    rigid_scenario: RigidScenario,
    gt: Coords,
    perturbed: Coords,
    aa_indices: Int[torch.Tensor, "B N_res"],
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """Moving the ground truth rigidly leaves a nonzero loss unchanged.

    The ground truth is aligned onto the prediction before the residuals
    are taken, so its frame must not matter.

    Args:
        rigid_scenario: Motion applied to the ground truth.
        gt: Ground truth.
        perturbed: Non-rigidly perturbed prediction.
        aa_indices: Residue types.
        ones_weight: EDM weight of 1.
    """
    kwargs = {"aa_indices": aa_indices, "lambda_sigma_weight": ones_weight}
    base = atom_loss(perturbed, gt, **kwargs)
    moved = atom_loss(perturbed, rigid_scenario.value.apply(gt), **kwargs)
    assert bool((base > 0).all())
    torch.testing.assert_close(moved, base, atol=1e-4, rtol=1e-4)


def test_atom_loss_grows_with_deviation(
    gt: Coords,
    perturbed: Coords,
    aa_indices: Int[torch.Tensor, "B N_res"],
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """Scaling a non-rigid deviation by 0.1, 1 and 5 strictly raises the loss.

    Not parametrized: the property is an ordering across the three scales.

    Args:
        gt: Ground truth.
        perturbed: ``gt`` plus a non-rigid deviation.
        aa_indices: Residue types.
        ones_weight: EDM weight of 1.
    """
    deviation = perturbed - gt
    losses = [
        atom_loss(
            gt + scale * deviation,
            gt,
            aa_indices=aa_indices,
            lambda_sigma_weight=ones_weight,
        )
        for scale in (0.1, 1.0, 5.0)
    ]
    assert bool((losses[0] < losses[1]).all())
    assert bool((losses[1] < losses[2]).all())


def test_atom_loss_scales_linearly_with_the_noise_weight(
    gt: Coords,
    perturbed: Coords,
    aa_indices: Int[torch.Tensor, "B N_res"],
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """Each item's loss is multiplied by its own ``lambda_sigma_weight``.

    Not parametrized: two different weights in one batch show both the
    scaling and that it is per item.

    Args:
        gt: Ground truth.
        perturbed: Prediction.
        aa_indices: Residue types.
        ones_weight: EDM weight of 1, the baseline.
    """
    weight = torch.tensor([2.0, 0.5])
    unit = atom_loss(
        perturbed,
        gt,
        aa_indices=aa_indices,
        lambda_sigma_weight=ones_weight,
    )
    weighted = atom_loss(
        perturbed,
        gt,
        aa_indices=aa_indices,
        lambda_sigma_weight=weight,
    )
    torch.testing.assert_close(weighted, weight * unit)


def test_atom_loss_gradient_reaches_prediction_only(
    gt: Coords,
    perturbed: Coords,
    aa_indices: Int[torch.Tensor, "B N_res"],
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """The prediction gets a finite, nonzero gradient; the target gets none.

    The ground truth is aligned under ``no_grad`` so the SVD is not
    differentiated. Not parametrized: one backward pass shows both sides.

    Args:
        gt: Ground truth.
        perturbed: Prediction.
        aa_indices: Residue types.
        ones_weight: EDM weight of 1.
    """
    pred = perturbed.clone().requires_grad_()
    target = gt.clone().requires_grad_()
    loss = atom_loss(
        pred,
        target,
        aa_indices=aa_indices,
        lambda_sigma_weight=ones_weight,
    )
    torch.autograd.backward([loss.sum()])
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) > 0
    assert target.grad is None


@pytest.mark.parametrize(
    ("letter", "weight"),
    [
        pytest.param("S", POLAR_RESIDUE_WEIGHT, id="polar_serine"),
        pytest.param("D", POLAR_RESIDUE_WEIGHT, id="polar_aspartate"),
        pytest.param("K", POLAR_RESIDUE_WEIGHT, id="polar_lysine"),
        pytest.param("A", NONPOLAR_RESIDUE_WEIGHT, id="nonpolar_alanine"),
        pytest.param("V", NONPOLAR_RESIDUE_WEIGHT, id="nonpolar_valine"),
    ],
)
def test_atom_loss_weights_each_residue_by_polarity(
    gt: Coords,
    perturbed: Coords,
    dims: LossDims,
    letter: str,
    weight: float,
) -> None:
    """One residue of type ``letter`` among alanines carries its table weight.

    The reference recomputes the Kabsch-aligned residuals with
    ``kabsch_align`` and averages them with residue 0's three atoms
    weighted by ``weight`` and every other atom by 1.

    Args:
        gt: Ground truth.
        perturbed: Prediction.
        dims: Supplies the residue and atom counts.
        letter: One-letter code of residue 0.
        weight: Weight residue 0 must get.
    """
    aa = torch.full((dims.batch, dims.n_res), restype_order["A"])
    aa[:, 0] = restype_order[letter]
    with torch.no_grad():
        (aligned,) = kabsch_align(  # pylint: disable=unpacking-non-sequence
            gt,
            perturbed,
            return_transform=False,
        )
    diff = perturbed - aligned
    sq = einsum(diff, diff, "b n d, b n d -> b n")
    per_atom = torch.ones(dims.n_atom)
    per_atom[: dims.atoms_per_res] = weight
    expected = einsum(sq, per_atom, "b n, n -> b") / (3 * per_atom.sum())
    loss = atom_loss(
        perturbed,
        gt,
        aa_indices=aa,
        lambda_sigma_weight=torch.ones(dims.batch),
    )
    torch.testing.assert_close(loss, expected, atol=1e-4, rtol=1e-4)


def test_atom_loss_masked_atoms_count_for_nothing(
    gt: Coords,
    aa_indices: Int[torch.Tensor, "B N_res"],
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """A huge error on masked atoms leaves the loss at zero.

    The first residue's atoms are pushed 100 angstroms away and masked out;
    masking must override polarity weighting and keep them out of both the
    alignment and the residuals. Not parametrized: the all-ones mask case
    is the next test.

    Args:
        gt: Ground truth.
        aa_indices: Residue types.
        ones_weight: EDM weight of 1.
    """
    mask = torch.ones(gt.shape[:2])
    mask[:, :3] = 0
    pred = gt.clone()
    pred[:, :3] += 100
    loss = atom_loss(
        pred,
        gt,
        mask=mask,
        aa_indices=aa_indices,
        lambda_sigma_weight=ones_weight,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-4, rtol=0)


def test_atom_loss_all_ones_mask_matches_no_mask(
    gt: Coords,
    perturbed: Coords,
    aa_indices: Int[torch.Tensor, "B N_res"],
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """An all-ones mask gives the same loss as omitting the mask.

    Not parametrized: the masked case is the test above.

    Args:
        gt: Ground truth.
        perturbed: Prediction.
        aa_indices: Residue types.
        ones_weight: EDM weight of 1.
    """
    kwargs = {"aa_indices": aa_indices, "lambda_sigma_weight": ones_weight}
    unmasked = atom_loss(perturbed, gt, **kwargs)
    masked = atom_loss(perturbed, gt, torch.ones(gt.shape[:2]), **kwargs)
    torch.testing.assert_close(masked, unmasked)


def test_atom_loss_rejects_atoms_not_divisible_by_residues(
    gt: Coords,
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """12 atoms with 5 residue types cannot be split evenly and raise.

    Not parametrized: any indivisible count takes the same branch.

    Args:
        gt: 12-atom ground truth.
        ones_weight: EDM weight of 1.
    """
    aa = torch.zeros(gt.shape[0], 5, dtype=torch.long)
    with pytest.raises(AtomResidueCountMismatchError, match="multiple"):
        _ = atom_loss(gt, gt, aa_indices=aa, lambda_sigma_weight=ones_weight)


def test_atom_loss_rejects_four_coordinate_atoms(
    malformed_inputs: MalformedInputs,
    aa_indices: Int[torch.Tensor, "B N_res"],
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """A ``(B, N, 4)`` prediction fails the jaxtyping contract.

    Not parametrized: the other losses' contracts are tested together
    below.

    Args:
        malformed_inputs: Supplies the bad and good coordinates.
        aa_indices: Residue types.
        ones_weight: EDM weight of 1.
    """
    x = malformed_inputs
    with pytest.raises(TypeCheckError):
        _ = atom_loss(
            x.coords4,
            x.coords3,
            aa_indices=aa_indices,
            lambda_sigma_weight=ones_weight,
        )


@pytest.mark.parametrize(
    ("letters", "expected"),
    [
        pytest.param("ASDVKG", (1.0, 2.0, 2.0, 1.0, 2.0, 1.0), id="mixed"),
        pytest.param("XX", (1.0, 1.0), id="unknown_clamped_in_range"),
    ],
)
def test_residue_type_weight_looks_up_polarity(
    letters: str,
    expected: tuple[float, ...],
) -> None:
    """Each residue maps to its polarity weight; ``X`` is clamped into range.

    ``X`` (index 20) sits one past the table, so it must not raise.

    Args:
        letters: One-letter codes.
        expected: Weights in order.
    """
    aa = torch.tensor([[restype_order[c] for c in letters]])
    weight = residue_type_weight(aa)
    torch.testing.assert_close(weight, torch.tensor([expected]))


def test_residue_type_weight_classifies_every_canonical_residue(
    canonical_residues: Int[torch.Tensor, "1 n_amino"],
) -> None:
    """Every canonical residue gets 2 if listed as polar and 1 otherwise.

    Checks both the exported table and the lookup through it. Not
    parametrized: all twenty residues are checked at once.

    Args:
        canonical_residues: Indices 0 to 19 in one row.
    """
    expected = [
        POLAR_RESIDUE_WEIGHT if r in POLAR_RESIDUES else NONPOLAR_RESIDUE_WEIGHT
        for r in RESTYPES_NO_X
    ]
    assert expected == RESIDUE_TYPE_WEIGHTS
    weights = residue_type_weight(canonical_residues)
    torch.testing.assert_close(weights, torch.tensor([expected]))


@pytest.mark.parametrize(
    "med_scenario",
    list(MedScenario),
    ids=[case.name for case in MedScenario],
)
def test_med_loss_matches_closed_form(
    med_scenario: MedScenario,
    batch: FeaturizedBatch,
    perturbed: Coords,
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """Over identical blocks, ``med_loss`` is a decayed mean of the two terms.

    The loss is ``(1/K) sum_k gamma**(K-k) (lam S + alpha_0 C)``. With
    every block equal it factors into
    ``(sum_k gamma**(K-k) / K) (lam mean(S) + alpha_0 C)``, where ``S`` is
    ``atom_loss`` per item and ``C`` is ``seq_ce_loss``. Uniform logits
    make ``C = log 20`` exactly.

    Args:
        med_scenario: Hyperparameters.
        batch: Ground truth.
        perturbed: Prediction used by every block.
        ones_weight: EDM weight of 1.
    """
    case = med_scenario.value
    logits = torch.zeros(*batch.aa_indices.shape, RESTYPE_NUM_NO_X)
    structure = atom_loss(
        perturbed,
        batch.r_gt,
        batch.atom5_mask.float(),
        aa_indices=batch.aa_indices,
        lambda_sigma_weight=ones_weight,
    ).mean()
    decay = sum(case.gamma**k for k in range(case.n_blocks)) / case.n_blocks
    expected = decay * (
        case.lam * structure + case.alpha_0 * math.log(RESTYPE_NUM_NO_X)
    )
    loss = med_loss(
        [perturbed] * case.n_blocks,
        [logits] * case.n_blocks,
        batch,
        LossParams(lam=case.lam, alpha_0=case.alpha_0, gamma=case.gamma),
        lambda_sigma_weight=ones_weight,
    )
    torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("n_structure", "n_sequence", "error"),
    [
        pytest.param(0, 0, NoDenoiserBlockError, id="no_blocks"),
        pytest.param(3, 2, BlockCountMismatchError, id="mismatched_counts"),
    ],
)
def test_med_loss_rejects_bad_block_lists(
    batch: FeaturizedBatch,
    ones_weight: Float[torch.Tensor, " B"],
    n_structure: int,
    n_sequence: int,
    error: type[ValueError],
) -> None:
    """An empty block list, or lists of unequal length, raise.

    Args:
        batch: Ground truth.
        ones_weight: EDM weight of 1.
        n_structure: Structure blocks passed.
        n_sequence: Sequence blocks passed.
        error: Exception expected.
    """
    logits = torch.zeros(*batch.aa_indices.shape, RESTYPE_NUM_NO_X)
    with pytest.raises(error):
        _ = med_loss(
            [batch.r_gt] * n_structure,
            [logits] * n_sequence,
            batch,
            LossParams(),
            lambda_sigma_weight=ones_weight,
        )


def test_med_loss_gradient_reaches_the_earliest_block(
    batch: FeaturizedBatch,
    perturbed: Coords,
    ones_weight: Float[torch.Tensor, " B"],
) -> None:
    """Block 0, the most decayed, still gets a finite nonzero gradient.

    Not parametrized: later blocks are less decayed, so block 0 is the case
    most likely to lose its gradient.

    Args:
        batch: Ground truth.
        perturbed: Prediction for every block.
        ones_weight: EDM weight of 1.
    """
    first = perturbed.clone().requires_grad_()
    logits = torch.zeros(*batch.aa_indices.shape, RESTYPE_NUM_NO_X)
    loss = med_loss(
        [first, perturbed, perturbed],
        [logits] * 3,
        batch,
        LossParams(gamma=0.5),
        lambda_sigma_weight=ones_weight,
    )
    (grad,) = torch.autograd.grad(loss, first)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


def test_smooth_lddt_identical_structures_give_the_closed_form(
    gt: Coords,
    lddt_floor: Float[torch.Tensor, ""],
) -> None:
    """With zero distance error the loss is its floor.

    Not parametrized: the value does not depend on the structure.

    Args:
        gt: Any structure.
        lddt_floor: Loss at zero distance error.
    """
    torch.testing.assert_close(smooth_lddt_loss(gt, gt), lddt_floor)


def test_smooth_lddt_masked_atoms_count_for_nothing(
    gt: Coords,
    lddt_floor: Float[torch.Tensor, ""],
) -> None:
    """Corrupting masked atoms leaves the loss at its zero-error floor.

    Pairs are counted only when both atoms are valid. Not parametrized:
    one masked residue is enough.

    Args:
        gt: Ground truth.
        lddt_floor: Loss at zero distance error.
    """
    mask = torch.ones(gt.shape[:2], dtype=torch.bool)
    mask[:, :3] = False
    pred = gt.clone()
    pred[:, :3] *= 7
    loss = smooth_lddt_loss(pred, gt, mask=mask)
    torch.testing.assert_close(loss, lddt_floor)


def test_smooth_lddt_rises_toward_one_with_distance_error(
    gt: Coords,
    perturbed: Coords,
) -> None:
    """Scaling a deviation up raises the loss, which stays at most 1.

    Not parametrized: the property is an ordering across the scales.

    Args:
        gt: Ground truth.
        perturbed: ``gt`` plus a non-rigid deviation.
    """
    deviation = perturbed - gt
    losses = [
        smooth_lddt_loss(gt + scale * deviation, gt).item()
        for scale in (0.0, 1.0, 10.0)
    ]
    assert losses[0] < losses[1] < losses[2] <= 1


def test_smooth_lddt_gradient_reaches_prediction(
    gt: Coords,
    perturbed: Coords,
) -> None:
    """The loss has a finite, nonzero gradient with respect to the prediction.

    Not parametrized: one backward pass covers the differentiable path.

    Args:
        gt: Ground truth.
        perturbed: Prediction.
    """
    pred = perturbed.clone().requires_grad_()
    (grad,) = torch.autograd.grad(smooth_lddt_loss(pred, gt), pred)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0


def test_pairwise_dist_on_a_3_4_5_triangle(
    triangle: Float[torch.Tensor, "3 3"],
) -> None:
    """Distances between the triangle's corners are 3, 4 and 5.

    The diagonal is ``sqrt(1e-8)``, the stabilising epsilon. Not
    parametrized: the triangle checks every off-diagonal entry.

    Args:
        triangle: Corners ``(0,0,0)``, ``(3,0,0)`` and ``(0,4,0)``.
    """
    expected = torch.tensor([[0.0, 3, 4], [3, 0, 5], [4, 5, 0]])
    torch.testing.assert_close(
        pairwise_dist(triangle),
        expected,
        atol=1e-4,
        rtol=0,
    )


@pytest.mark.parametrize(
    "target_form",
    list(TargetForm),
    ids=[case.name for case in TargetForm],
)
@pytest.mark.parametrize(
    "residue_mask_case",
    list(ResidueMask),
    ids=[case.name for case in ResidueMask],
)
def test_distogram_loss_residue_uniform_logits_give_log_bins(
    dims: LossDims,
    residue_bins: Int[torch.Tensor, "B N_res N_res"],
    residue_target: (
        Int[torch.Tensor, "B N_res N_res"]
        | Float[torch.Tensor, "B N_res N_res n"]
    ),
    residue_mask: Float[torch.Tensor, "B N_res"] | None,
) -> None:
    """Uniform logits give ``log(n_bins)`` per item for any target and mask.

    Args:
        dims: Supplies ``n_bins``.
        residue_bins: Supplies the pair layout for the logits.
        residue_target: Target bins as indices or one-hot.
        residue_mask: Mask hiding the last residue, or ``None``.
    """
    logits = torch.zeros(*residue_bins.shape, dims.n_bins)
    loss = distogram_loss_residue(logits, residue_target, residue_mask)
    expected = torch.full((dims.batch,), math.log(dims.n_bins))
    torch.testing.assert_close(loss, expected)


def test_distogram_loss_residue_mask_hides_wrong_pairs(
    dims: LossDims,
    residue_bins: Int[torch.Tensor, "B N_res N_res"],
) -> None:
    """Confidently wrong logits on masked residues leave the loss near 0.

    Logits are 50 on the true bin everywhere except the last residue's
    pairs, which put 50 on a wrong bin. Unmasked that costs about 50 per
    wrong pair; masking the last residue removes them. Not parametrized:
    one masked residue shows the mask is applied to both pair axes.

    Args:
        dims: Supplies the sizes.
        residue_bins: Target bins.
    """
    wrong = residue_bins.clone()
    wrong[:, -1, :] += 1
    wrong[:, :, -1] += 1
    logits = 50 * F.one_hot(wrong % dims.n_bins, dims.n_bins).float()
    mask = torch.ones(dims.batch, dims.n_res)
    mask[:, -1] = 0
    masked = distogram_loss_residue(logits, residue_bins, mask)
    unmasked = distogram_loss_residue(logits, residue_bins)
    torch.testing.assert_close(
        masked,
        torch.zeros_like(masked),
        atol=1e-6,
        rtol=0,
    )
    assert bool((unmasked > 1).all())


@pytest.mark.parametrize(
    "target_form",
    list(TargetForm),
    ids=[case.name for case in TargetForm],
)
@pytest.mark.parametrize(
    "atom_mask_case",
    list(AtomMask),
    ids=[case.name for case in AtomMask],
)
def test_distogram_loss_atom_uniform_logits_give_log_bins(
    dims: LossDims,
    atom_bins: Int[torch.Tensor, "B N_atom K"],
    atom_target: (
        Int[torch.Tensor, "B N_atom K"] | Float[torch.Tensor, "B N_atom K n"]
    ),
    atom_mask: Bool[torch.Tensor, "B N_atom K"] | None,
) -> None:
    """Uniform logits give ``log(n_bins)`` per item for any target and mask.

    The documented loss is ``-1/(N_atom K) sum``, so with ``K = 5`` slots for
    12 atoms the unmasked case fails today: it divides by ``K²`` and
    returns ``(12/5) log(n_bins)``. ``atom_mask`` marks that case strict
    ``xfail``.

    Args:
        dims: Supplies ``n_bins``.
        atom_bins: Supplies the slot layout for the logits.
        atom_target: Target bins as indices or one-hot.
        atom_mask: All-valid slot mask, or ``None``.
    """
    logits = torch.zeros(*atom_bins.shape, dims.n_bins)
    loss = distogram_loss_atom(logits, atom_target, atom_mask)
    expected = torch.full((dims.batch,), math.log(dims.n_bins))
    torch.testing.assert_close(loss, expected)


def test_distogram_loss_atom_mask_hides_wrong_slots(
    dims: LossDims,
    atom_bins: Int[torch.Tensor, "B N_atom K"],
) -> None:
    """Confidently wrong logits on masked slots leave the loss near 0.

    Not parametrized: one masked slot per atom shows the mask is applied.

    Args:
        dims: Supplies the sizes.
        atom_bins: Target bins.
    """
    wrong = atom_bins.clone()
    wrong[..., -1] += 1
    logits = 50 * F.one_hot(wrong % dims.n_bins, dims.n_bins).float()
    mask = torch.ones(atom_bins.shape, dtype=torch.bool)
    mask[..., -1] = False
    loss = distogram_loss_atom(logits, atom_bins, mask)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-6, rtol=0)


def test_distogram_loss_one_hot_and_index_targets_agree(
    dims: LossDims,
    residue_bins: Int[torch.Tensor, "B N_res N_res"],
    atom_bins: Int[torch.Tensor, "B N_atom K"],
) -> None:
    """Both losses give the same value for one-hot and index targets.

    Uses non-uniform logits, where the two code paths, an einsum and a
    gather, could disagree. Not parametrized: both losses are checked.

    Args:
        dims: Supplies ``n_bins``.
        residue_bins: Residue targets.
        atom_bins: Atom targets.
    """
    res_logits = 3 * sine_pattern((*residue_bins.shape, dims.n_bins), 0.37)
    atom_logits = 3 * sine_pattern((*atom_bins.shape, dims.n_bins), 0.53)
    mask = torch.ones(atom_bins.shape, dtype=torch.bool)
    torch.testing.assert_close(
        distogram_loss_residue(res_logits, residue_bins),
        distogram_loss_residue(
            res_logits,
            F.one_hot(residue_bins, dims.n_bins).float(),
        ),
    )
    torch.testing.assert_close(
        distogram_loss_atom(atom_logits, atom_bins, mask),
        distogram_loss_atom(
            atom_logits,
            F.one_hot(atom_bins, dims.n_bins).float(),
            mask,
        ),
    )


def test_seq_ce_loss_uniform_logits_give_log_vocabulary(
    aa_indices: Int[torch.Tensor, "B N_res"],
) -> None:
    """Uniform logits give a scalar loss of exactly ``log 20``.

    Not parametrized: the value is the same for any valid targets.

    Args:
        aa_indices: Valid residue types.
    """
    logits = torch.zeros(*aa_indices.shape, RESTYPE_NUM_NO_X)
    loss = seq_ce_loss(logits, aa_indices)
    assert loss.shape == ()
    torch.testing.assert_close(loss, torch.tensor(math.log(RESTYPE_NUM_NO_X)))


@pytest.mark.parametrize(
    "ignored",
    [PAD_TOKEN, RESTYPE_NUM_NO_X],
    ids=["padding", "unknown_x"],
)
def test_seq_ce_loss_skips_ignored_positions(
    aa_indices: Int[torch.Tensor, "B N_res"],
    ignored: int,
) -> None:
    """Positions with ``ignored`` as their target add nothing to the loss.

    Valid positions get logits of 50 on their target, so they cost about 0;
    the ignored residue gets uniform logits, which would add ``log 20`` if
    it counted.

    Args:
        aa_indices: Residue types; the last residue is replaced here.
        ignored: Padding or the unknown ``X`` index.
    """
    logits = 50 * F.one_hot(aa_indices, RESTYPE_NUM_NO_X).float()
    logits[:, -1] = 0
    targets = aa_indices.clone()
    targets[:, -1] = ignored
    loss = seq_ce_loss(logits, targets)
    torch.testing.assert_close(loss, torch.zeros(()), atol=1e-6, rtol=0)


def test_seq_ce_loss_with_no_valid_target_is_zero(
    aa_indices: Int[torch.Tensor, "B N_res"],
) -> None:
    """If every target is ignored the loss is exactly 0, not NaN.

    Not parametrized: both ignored tokens take the same path, covered by
    the test above.

    Args:
        aa_indices: Supplies the shape.
    """
    logits = torch.zeros(*aa_indices.shape, RESTYPE_NUM_NO_X)
    targets = torch.full_like(aa_indices, RESTYPE_NUM_NO_X)
    assert seq_ce_loss(logits, targets).item() == 0


def test_featurize_keeps_pdb_x_at_the_ignored_index(
    dims: LossDims,
) -> None:
    """A PDB ``X`` residue reaches the loss as index 20, which CE ignores.

    ``seq_ce_loss`` skips targets of ``RESTYPE_NUM_NO_X`` and above, so the
    featurisation must keep ``X`` there rather than remap it to a
    canonical residue. Not parametrized: one ``X`` in a short chain covers
    the mapping.

    Args:
        dims: Supplies ``n_bins`` for the distogram modules.
    """
    n_res = 5
    aatype = np.array([restype_order[c] for c in "ACXDE"], dtype=np.intp)
    protein = Protein(
        atom_positions=np.zeros((n_res, 37, 3)),
        aatype=aatype,
        atom_mask=np.ones((n_res, 37)),
        residue_index=np.arange(n_res, dtype=np.intp),
        chain_index=np.zeros(n_res, dtype=np.intp),
        b_factors=np.zeros((n_res, 37)),
    )
    n = dims.n_bins
    item = featurize_single_item(
        protein,
        build_distogram_module(TemplateDistogramParams(n_bins=n)).eval(),
        build_distogram_module(ResidueDistogramParams(n_bins=n)).eval(),
        build_distogram_module(AtomDistogramParams(n_bins=n)).eval(),
        TrainConfig(),
        max_seq_len_in_batch=n_res,
    )
    assert item.aa_indices[2].item() == RESTYPE_NUM_NO_X


@pytest.mark.parametrize(
    "malformed_scenario",
    list(MalformedScenario),
    ids=[case.name for case in MalformedScenario],
)
def test_losses_reject_malformed_inputs(
    malformed_scenario: MalformedScenario,
    malformed_inputs: MalformedInputs,
) -> None:
    """Each loss's jaxtyping contract rejects its malformed input.

    Args:
        malformed_scenario: Call with a wrong-shape input.
        malformed_inputs: Tensors the call reads.
    """
    with pytest.raises(TypeCheckError):
        _ = malformed_scenario.value.call(malformed_inputs)

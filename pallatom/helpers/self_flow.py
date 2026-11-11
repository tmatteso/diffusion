"""Dual-timestep noising for Self-Flow training.

Self-Flow (Chefer et al. 2026, arXiv 2603.06507) trains on two noised views
of each protein. The student view noises each residue at one of two noise
levels, chosen per residue by a random mask. The teacher view noises every
residue at the lowest level the student's valid residues received, so it is
never noisier than the student anywhere. Both views use the same noise
sample.

This module holds the tensor logic that does not depend on the rest of the
featurizer: the per-residue noise levels, the zero-centre-of-mass noise
displacement and the pairwise template time. ``helpers.data`` combines them
with the template distogram into a ``NoisedView`` per view.
"""

import dataclasses

import torch
from beartype import beartype
from einops import rearrange, repeat
from helpers.alignment import masked_com
from jaxtyping import Bool, Float, jaxtyped


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class NoisedView:
    """One noised copy of a protein and the conditioning that goes with it.

    Attributes:
        r_noised: Noised atom positions, flat (atom-indexed) layout.
        noised_res_distogram: One-hot template distogram of the noised Cβ
            positions, binned into ``n_templ_bins`` distance categories.
        t_hat: Noise level sigma of each residue.
        t_normalized: Template time weight of each residue pair.
    """

    r_noised: Float[torch.Tensor, "N_atom 3"]
    noised_res_distogram: Float[torch.Tensor, "N_res N_res n_templ_bins"]
    t_hat: Float[torch.Tensor, "N_res"]
    t_normalized: Float[torch.Tensor, "N_res N_res"]


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class NoisedViewBatch:
    """A batch of ``NoisedView``, every field with a leading batch dimension.

    Attributes:
        r_noised: Noised atom positions.
        noised_res_distogram: One-hot template distogram of the noised Cβ
            positions.
        t_hat: Noise level sigma of each residue.
        t_normalized: Template time weight of each residue pair.
    """

    r_noised: Float[torch.Tensor, "B N_atom 3"]
    noised_res_distogram: Float[torch.Tensor, "B N_res N_res n_templ_bins"]
    t_hat: Float[torch.Tensor, "B N_res"]
    t_normalized: Float[torch.Tensor, "B N_res N_res"]

    @classmethod
    def stack(cls, views: list[NoisedView]) -> "NoisedViewBatch":
        """Stack per-protein views into a batch.

        Args:
            views: One view per protein, all padded to the same length.

        Returns:
            The views stacked along a new leading batch dimension.
        """
        return cls(
            r_noised=torch.stack([v.r_noised for v in views]),
            noised_res_distogram=torch.stack(
                [v.noised_res_distogram for v in views],
            ),
            t_hat=torch.stack([v.t_hat for v in views]),
            t_normalized=torch.stack([v.t_normalized for v in views]),
        )

    @classmethod
    def stack_optional(
        cls,
        views: list[NoisedView | None],
    ) -> "NoisedViewBatch | None":
        """Stack per-protein views, or return None if any protein has none.

        Self-Flow either builds a teacher view for every protein in a batch
        or for none, so a missing view means the batch has no teacher.

        Args:
            views: Each protein's view, or None.

        Returns:
            The stacked views, or None.
        """
        present = [view for view in views if view is not None]
        return cls.stack(present) if len(present) == len(views) else None

    def to(
        self,
        device: str | torch.device,
        *,
        non_blocking: bool,
    ) -> "NoisedViewBatch":
        """Return a copy with every tensor moved to ``device``.

        Args:
            device: Target device.
            non_blocking: Whether the copy may overlap with host work.

        Returns:
            The moved batch.
        """
        return NoisedViewBatch(
            r_noised=self.r_noised.to(device, non_blocking=non_blocking),
            noised_res_distogram=self.noised_res_distogram.to(
                device,
                non_blocking=non_blocking,
            ),
            t_hat=self.t_hat.to(device, non_blocking=non_blocking),
            t_normalized=self.t_normalized.to(
                device,
                non_blocking=non_blocking,
            ),
        )

    def pin_memory(self) -> "NoisedViewBatch":
        """Return a copy with every tensor in page-locked memory.

        Returns:
            The pinned batch.
        """
        return NoisedViewBatch(
            r_noised=self.r_noised.pin_memory(),
            noised_res_distogram=self.noised_res_distogram.pin_memory(),
            t_hat=self.t_hat.pin_memory(),
            t_normalized=self.t_normalized.pin_memory(),
        )


@jaxtyped(typechecker=beartype)
@dataclasses.dataclass(frozen=True)
class DualTimestepSigmas:
    """Noise levels of the two Self-Flow views of one protein.

    Attributes:
        student: Noise level of each residue in the student view.
        teacher: The single noise level of every residue in the teacher
            view: the lowest student level over valid residues.
    """

    student: Float[torch.Tensor, "N_res"]
    teacher: Float[torch.Tensor, ""]


@jaxtyped(typechecker=beartype)
def sample_residue_mask(
    n_res: int,
    mask_ratio: float,
    apply_prob: float,
) -> Bool[torch.Tensor, "N_res"]:
    """Choose which residues take the second noise level.

    Each residue is masked independently with probability ``mask_ratio``.
    With probability ``1 - apply_prob`` the whole protein is left unmasked.

    Args:
        n_res: Number of residues, padding included.
        mask_ratio: Per-residue masking probability.
        apply_prob: Probability that the protein is masked at all.

    Returns:
        True for residues that take the second noise level.
    """
    apply = torch.rand(()) < apply_prob
    return (torch.rand(n_res) < mask_ratio) & apply


@jaxtyped(typechecker=beartype)
def dual_timestep_sigmas(
    first_sigma: Float[torch.Tensor, ""],
    second_sigma: Float[torch.Tensor, ""],
    residue_noise_mask: Bool[torch.Tensor, "N_res"],
    residue_mask: Bool[torch.Tensor, "N_res"],
) -> DualTimestepSigmas:
    """Per-residue student noise levels and the teacher's single level.

    The student gives masked residues ``second_sigma`` and all others
    ``first_sigma``. The teacher takes the lowest level any valid residue
    received, so it equals the student view when no valid residue is masked
    or every valid residue is. With no valid residues it falls back to
    ``first_sigma``.

    Args:
        first_sigma: Noise level of unmasked residues.
        second_sigma: Noise level of masked residues.
        residue_noise_mask: True for residues that take ``second_sigma``.
        residue_mask: True for residues that exist (not padding).

    Returns:
        The student's per-residue levels and the teacher's level.
    """
    student = torch.where(residue_noise_mask, second_sigma, first_sigma)
    lowest_valid = torch.where(
        residue_mask,
        student,
        torch.full_like(student, torch.inf),
    ).amin()
    teacher = torch.where(residue_mask.any(), lowest_valid, first_sigma)
    return DualTimestepSigmas(student=student, teacher=teacher)


@jaxtyped(typechecker=beartype)
def centered_noise_displacement(
    scaled_noise: Float[torch.Tensor, "N_atom 3"],
    atom_mask: Bool[torch.Tensor, "N_atom"],
) -> Float[torch.Tensor, "N_atom 3"]:
    """Shift already-scaled noise so its centre of mass over valid atoms is 0.

    Centring after scaling keeps the centre of mass at zero even when atoms
    carry different noise levels. Scaling unit noise that was centred first
    would not.

    Args:
        scaled_noise: Noise already multiplied by each atom's sigma.
        atom_mask: True for atoms that exist.

    Returns:
        The noise with its valid-atom centre of mass removed.
    """
    com = masked_com(
        rearrange(scaled_noise, "n d -> 1 n d"),
        mask=rearrange(atom_mask, "n -> 1 n"),
    )
    return scaled_noise - rearrange(com, "1 1 d -> 1 d")


@jaxtyped(typechecker=beartype)
def pair_template_time(
    residue_time: Float[torch.Tensor, "N_res"],
) -> Float[torch.Tensor, "N_res N_res"]:
    """Template time of each residue pair: the noisier residue's time.

    The template distance of pair ``(i, j)`` is computed from both noised
    residues, so its reliability is limited by the noisier one. Normalized
    time increases with sigma, so this is the time of the larger sigma.

    Args:
        residue_time: Normalized template time of each residue.

    Returns:
        The larger of the two residues' times, for every pair.
    """
    return torch.maximum(
        rearrange(residue_time, "n -> n 1"),
        rearrange(residue_time, "m -> 1 m"),
    )


@jaxtyped(typechecker=beartype)
def per_atom_sigma(
    residue_sigma: Float[torch.Tensor, "N_res"],
    atoms_per_residue: int,
) -> Float[torch.Tensor, "N_atom 1"]:
    """Repeat each residue's sigma over its atoms, ready to scale xyz noise.

    Args:
        residue_sigma: Noise level of each residue.
        atoms_per_residue: Atoms per residue in the flat layout.

    Returns:
        Each atom's sigma, with a trailing axis that broadcasts over xyz.
    """
    return repeat(residue_sigma, "n -> (n a) 1", a=atoms_per_residue)

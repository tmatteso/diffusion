"""Tests for the template embedder.

Covers TemplateEmbedder's output shape and dtype, that each of its four
inputs conditions the output, batch independence, gradient flow to every
differentiable input, and the jaxtyping shape contract on ``forward``.

Inputs are deterministic patterns rather than random draws, and the model
is built under a fixed seed inside its fixture, so every test is
reproducible on its own regardless of run order.
"""

import abc
import dataclasses
import enum
from typing import cast, override

import pytest
import torch
import torch.nn.functional as F
from architecture.template_embedder import TemplateEmbedder
from einops import rearrange, reduce
from helpers.useful_objects import manual_seed
from jaxtyping import Float, TypeCheckError


@dataclasses.dataclass(frozen=True)
class EmbedderDims:
    """Batch, sequence and channel sizes shared by every test in the module.

    Attributes:
        batch: Batch size ``B``.
        n_res: Residues per protein ``N_res``.
        n_bins: Distogram bins, the last axis of ``f_distogram``.
        c_z: Trunk pair embedding width, the last axis of ``z_ij``.
        c: Internal pair width; must be divisible by ``n_heads``.
        d: Output pair width.
        n_blocks: PairformerStack depth.
        n_heads: Attention heads in the PairformerStack.
    """

    batch: int
    n_res: int
    n_bins: int
    c_z: int
    c: int
    d: int
    n_blocks: int
    n_heads: int


@dataclasses.dataclass(frozen=True)
class EmbedderInputs:
    """The four positional arguments of ``TemplateEmbedder.forward``.

    Deliberately not ``@jaxtyped``: malformed-input scenarios build a copy
    with one field reshaped, and the embedder, not this container, must be
    the thing that rejects it.

    Attributes:
        f_distogram: One-hot distogram ``(B, N_res, N_res, n_bins)``.
        f_pseudo_beta_mask: Per-residue template mask ``(B, N_res)``.
        z_ij: Trunk pair embedding ``(B, N_res, N_res, c_z)``.
        t: Diffusion time broadcast over pairs ``(B, N_res, N_res)``.
    """

    f_distogram: Float[torch.Tensor, "B N_res N_res n_bins"]
    f_pseudo_beta_mask: Float[torch.Tensor, "B N_res"]
    z_ij: Float[torch.Tensor, "B N_res N_res c_z"]
    t: Float[torch.Tensor, "B N_res N_res"]


@dataclasses.dataclass(frozen=True)
class InputEdit(abc.ABC):
    """A change to one embedder input, applied to the base inputs.

    Each subclass names one kind of change and holds its settings as plain
    fields, so every scenario reads as data.

    Attributes:
        field: Name of the ``EmbedderInputs`` field to change.
    """

    field: str

    @abc.abstractmethod
    def transform(
        self,
        tensor: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Return the replacement for the field's tensor.

        Args:
            tensor: The field's current value.

        Returns:
            The tensor to store in ``field``.
        """

    def apply(self, inputs: EmbedderInputs) -> EmbedderInputs:
        """Return a copy of ``inputs`` with ``field`` transformed.

        Args:
            inputs: Inputs to copy.

        Returns:
            The copy, every other field shared with ``inputs``.
        """
        old = cast(Float[torch.Tensor, "..."], getattr(inputs, self.field))
        return dataclasses.replace(inputs, **{self.field: self.transform(old)})


@dataclasses.dataclass(frozen=True)
class ScaleEdit(InputEdit):
    """Multiply the field by a constant.

    Attributes:
        factor: Multiplier applied to every element.
    """

    factor: float

    @override
    def transform(
        self,
        tensor: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Scale ``tensor`` by ``factor``.

        Args:
            tensor: The field's current value.

        Returns:
            ``tensor * factor``.
        """
        return tensor * self.factor


@dataclasses.dataclass(frozen=True)
class ShiftBinsEdit(InputEdit):
    """Move every entry one step up the last axis, wrapping the top bin."""

    @override
    def transform(
        self,
        tensor: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Roll ``tensor`` by one along its last axis.

        Args:
            tensor: The field's current value.

        Returns:
            The rolled tensor; a one-hot bin ``k`` becomes bin ``k + 1``.
        """
        return tensor.roll(1, -1)


@dataclasses.dataclass(frozen=True)
class SelectEdit(InputEdit):
    """Keep one index of an axis, removing that axis.

    Attributes:
        dim: Axis to select from.
        index: Position along ``dim`` to keep.
    """

    dim: int
    index: int

    @override
    def transform(
        self,
        tensor: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Select ``index`` along ``dim``.

        Args:
            tensor: The field's current value.

        Returns:
            ``tensor`` with axis ``dim`` removed.
        """
        return tensor.select(self.dim, self.index)


@dataclasses.dataclass(frozen=True)
class DropFirstEdit(InputEdit):
    """Remove the first entry of an axis, shortening it by one.

    Attributes:
        dim: Axis to shorten.
    """

    dim: int

    @override
    def transform(
        self,
        tensor: Float[torch.Tensor, "..."],
    ) -> Float[torch.Tensor, "..."]:
        """Drop index 0 along ``dim``.

        Args:
            tensor: The field's current value.

        Returns:
            ``tensor`` with ``dim`` one shorter.
        """
        return tensor.narrow(self.dim, 1, tensor.shape[self.dim] - 1)


class ConditioningScenario(enum.Enum):
    """Changes to a single input that must change the embedding.

    Members:
        time: ``t`` moves from 0.5 to 1.0 on every pair.
        mask: Every residue's pseudo-beta mask drops from 1 to 0.
        distogram: Every pair's one-hot bin shifts up by one.
        pair_embedding: ``z_ij`` is negated.
    """

    time = ScaleEdit("t", factor=2.0)
    mask = ScaleEdit("f_pseudo_beta_mask", factor=0.0)
    distogram = ShiftBinsEdit("f_distogram")
    pair_embedding = ScaleEdit("z_ij", factor=-1.0)


class MalformedScenario(enum.Enum):
    """Inputs whose shape breaks the ``forward`` jaxtyping contract.

    Members:
        distogram_missing_bins: ``f_distogram`` loses its ``n_bins`` axis.
        mask_too_short: ``f_pseudo_beta_mask`` has one residue fewer than
            the pair tensors.
        time_per_residue: ``t`` is ``(B, N_res)`` instead of per pair.
    """

    distogram_missing_bins = SelectEdit("f_distogram", dim=-1, index=0)
    mask_too_short = DropFirstEdit("f_pseudo_beta_mask", dim=1)
    time_per_residue = SelectEdit("t", dim=1, index=0)


@pytest.fixture
def dims() -> EmbedderDims:
    """Return the small sizes every test builds its model and inputs from.

    Returns:
        Sizes with ``c`` divisible by ``n_heads`` as the attention requires.
    """
    return EmbedderDims(
        batch=2,
        n_res=6,
        n_bins=8,
        c_z=16,
        c=8,
        d=16,
        n_blocks=1,
        n_heads=4,
    )


@pytest.fixture
def embedder(dims: EmbedderDims) -> TemplateEmbedder:
    """Build a TemplateEmbedder in eval mode under a fixed seed.

    Args:
        dims: Model sizes.

    Returns:
        Deterministically initialised embedder.
    """
    _ = manual_seed(0)
    return TemplateEmbedder(
        n_bins=dims.n_bins,
        c_z=dims.c_z,
        c=dims.c,
        d=dims.d,
        n_blocks=dims.n_blocks,
        n_heads=dims.n_heads,
    ).eval()


@pytest.fixture
def base_inputs(dims: EmbedderDims) -> EmbedderInputs:
    """Build deterministic, hand-readable embedder inputs.

    The distogram puts pair ``(i, j)`` in bin ``(i + j) % n_bins``, every
    residue is masked in, ``z_ij`` is a sine ramp over its elements so no
    two channels are equal, and ``t`` is 0.5 everywhere. Each batch item
    is offset differently so the batch-independence test is meaningful.

    Args:
        dims: Sizes to build the tensors at.

    Returns:
        The four forward inputs.
    """
    b, n = dims.batch, dims.n_res
    idx = torch.arange(n)
    bins = (rearrange(idx, "i -> 1 i 1") + rearrange(idx, "j -> 1 1 j")) + (
        rearrange(torch.arange(b), "b -> b 1 1")
    )
    ramp = torch.arange(b * n * n * dims.c_z, dtype=torch.float32)
    return EmbedderInputs(
        f_distogram=F.one_hot(bins % dims.n_bins, dims.n_bins).float(),
        f_pseudo_beta_mask=torch.ones(b, n),
        z_ij=rearrange(
            torch.sin(ramp),
            "(b i j c) -> b i j c",
            b=b,
            i=n,
            j=n,
        ),
        t=torch.full((b, n, n), 0.5),
    )


def embed(
    model: TemplateEmbedder,
    inputs: EmbedderInputs,
) -> Float[torch.Tensor, "B N_res N_res d"]:
    """Run ``model.forward`` on the four fields of ``inputs`` in order.

    Args:
        model: Template embedder to run.
        inputs: Forward inputs.

    Returns:
        Template pair embedding.
    """
    return model(
        inputs.f_distogram,
        inputs.f_pseudo_beta_mask,
        inputs.z_ij,
        inputs.t,
    )


@pytest.fixture
def first_item_inputs(base_inputs: EmbedderInputs) -> EmbedderInputs:
    """Return the first batch item of ``base_inputs`` as a batch of one.

    Args:
        base_inputs: Batched forward inputs.

    Returns:
        Every field sliced to ``[:1]``, keeping the batch axis.
    """
    return EmbedderInputs(
        f_distogram=base_inputs.f_distogram[:1],
        f_pseudo_beta_mask=base_inputs.f_pseudo_beta_mask[:1],
        z_ij=base_inputs.z_ij[:1],
        t=base_inputs.t[:1],
    )


def test_template_embedder_output_shape_and_dtype(
    dims: EmbedderDims,
    base_inputs: EmbedderInputs,
    embedder: TemplateEmbedder,
) -> None:
    """The embedding is float32 ``(B, N_res, N_res, d)`` and finite.

    Not parametrized: the output shape depends only on the sizes in
    ``dims``, which the shape contract already ties to the inputs.

    Args:
        dims: Sizes the output must match.
        base_inputs: Valid forward inputs.
        embedder: Model under test.
    """
    with torch.no_grad():
        out = embed(embedder, base_inputs)
    assert out.shape == (dims.batch, dims.n_res, dims.n_res, dims.d)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "conditioning_scenario",
    list(ConditioningScenario),
    ids=[case.name for case in ConditioningScenario],
)
def test_template_embedder_output_depends_on_each_input(
    base_inputs: EmbedderInputs,
    conditioning_scenario: ConditioningScenario,
    embedder: TemplateEmbedder,
) -> None:
    """Changing any one input, all else fixed, changes the embedding.

    A failure means that input is not wired into the forward pass.

    Args:
        base_inputs: Valid forward inputs.
        conditioning_scenario: Change applied to one input.
        embedder: Model under test.
    """
    with torch.no_grad():
        base = embed(embedder, base_inputs)
        edited = embed(embedder, conditioning_scenario.value.apply(base_inputs))
    assert not torch.allclose(base, edited)


def test_template_embedder_batch_items_are_independent(
    base_inputs: EmbedderInputs,
    first_item_inputs: EmbedderInputs,
    embedder: TemplateEmbedder,
) -> None:
    """Running the first item alone matches its slice of the batched run.

    Not parametrized: one sliced item is enough to expose any cross-batch
    mixing, since every batch item has distinct inputs.

    Args:
        base_inputs: Batched forward inputs.
        first_item_inputs: The first batch item alone.
        embedder: Model under test.
    """
    with torch.no_grad():
        batched = embed(embedder, base_inputs)[:1]
        single = embed(embedder, first_item_inputs)
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "grad_input_name",
    ["f_distogram", "f_pseudo_beta_mask", "z_ij"],
)
def test_template_embedder_gradient_reaches_input(
    base_inputs: EmbedderInputs,
    grad_input_name: str,
    embedder: TemplateEmbedder,
) -> None:
    """The summed embedding has a finite, nonzero gradient for the input.

    ``t`` is left out: in training it is a sampled noise level, never a
    learned quantity.

    Args:
        base_inputs: Valid forward inputs.
        grad_input_name: Field to differentiate with respect to.
        embedder: Model under test.
    """
    leaf = cast(torch.Tensor, getattr(base_inputs, grad_input_name))
    leaf = leaf.clone().requires_grad_()
    inputs = dataclasses.replace(base_inputs, **{grad_input_name: leaf})
    out = reduce(embed(embedder, inputs), "b i j d -> ", "sum")
    torch.autograd.backward([out])
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
    assert torch.count_nonzero(leaf.grad) > 0


@pytest.mark.parametrize(
    "malformed_scenario",
    list(MalformedScenario),
    ids=[case.name for case in MalformedScenario],
)
def test_template_embedder_rejects_malformed_input(
    base_inputs: EmbedderInputs,
    malformed_scenario: MalformedScenario,
    embedder: TemplateEmbedder,
) -> None:
    """A wrongly shaped input fails the jaxtyping contract on ``forward``.

    Args:
        base_inputs: Valid forward inputs.
        malformed_scenario: Edit that breaks one input's shape.
        embedder: Model under test.
    """
    malformed = malformed_scenario.value.apply(base_inputs)
    with pytest.raises(TypeCheckError):
        _ = embed(embedder, malformed)

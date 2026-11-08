"""Tests for the template embedder.

Covers TemplateEmbedder's output shape and dtype, that each of its four
inputs conditions the output, batch independence, gradient flow to every
differentiable input, and the jaxtyping shape contract on ``forward``.

Inputs are deterministic patterns rather than random draws, and the model
is built under a fixed seed inside its fixture, so every test is
reproducible on its own regardless of run order.
"""

import dataclasses
import enum
from collections.abc import Callable
from typing import cast

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
class InputEdit:
    """A change to one embedder input, applied to the base inputs.

    Attributes:
        field: Name of the ``EmbedderInputs`` field to change.
        transform: Maps that field's tensor to its replacement.
    """

    field: str
    transform: Callable[
        [Float[torch.Tensor, "..."]],
        Float[torch.Tensor, "..."],
    ]

    def apply(self, inputs: EmbedderInputs) -> EmbedderInputs:
        """Return a copy of ``inputs`` with ``field`` transformed.

        Args:
            inputs: Inputs to copy.

        Returns:
            The copy, every other field shared with ``inputs``.
        """
        old = cast(Float[torch.Tensor, "..."], getattr(inputs, self.field))
        return dataclasses.replace(inputs, **{self.field: self.transform(old)})


class ConditioningScenario(enum.Enum):
    """Changes to a single input that must change the embedding.

    Members:
        time: ``t`` moves from 0.5 to 1.0 on every pair.
        mask: Every residue's pseudo-beta mask drops from 1 to 0.
        distogram: Every pair's one-hot bin shifts up by one.
        pair_embedding: ``z_ij`` is negated.
    """

    time = InputEdit("t", lambda t: t * 2)
    mask = InputEdit("f_pseudo_beta_mask", lambda m: m * 0)
    distogram = InputEdit("f_distogram", lambda d: d.roll(1, -1))
    pair_embedding = InputEdit("z_ij", lambda z: -z)


class MalformedScenario(enum.Enum):
    """Inputs whose shape breaks the ``forward`` jaxtyping contract.

    Members:
        distogram_missing_bins: ``f_distogram`` loses its ``n_bins`` axis.
        mask_too_short: ``f_pseudo_beta_mask`` has one residue fewer than
            the pair tensors.
        time_per_residue: ``t`` is ``(B, N_res)`` instead of per pair.
    """

    distogram_missing_bins = InputEdit("f_distogram", lambda d: d[..., 0])
    mask_too_short = InputEdit("f_pseudo_beta_mask", lambda m: m[:, 1:])
    time_per_residue = InputEdit("t", lambda t: t[:, 0])


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


@pytest.fixture
def run_embedder(
    embedder: TemplateEmbedder,
) -> Callable[[EmbedderInputs], Float[torch.Tensor, "B N_res N_res d"]]:
    """Return a callable that feeds an ``EmbedderInputs`` to the embedder.

    Args:
        embedder: Model under test.

    Returns:
        Callable running ``forward`` on the four fields in order.
    """

    def run(x: EmbedderInputs) -> Float[torch.Tensor, "B N_res N_res d"]:
        """Run the embedder on ``x``.

        Args:
            x: Forward inputs.

        Returns:
            Template pair embedding.
        """
        return embedder(x.f_distogram, x.f_pseudo_beta_mask, x.z_ij, x.t)

    return run


@pytest.fixture
def scenario_edit(request: pytest.FixtureRequest) -> InputEdit:
    """Return the edit carried by the scenario routed in via indirect param.

    Args:
        request: Pytest request carrying a ``ConditioningScenario`` or
            ``MalformedScenario`` member in ``param``.

    Returns:
        The member's ``InputEdit``.
    """
    scenario = cast(ConditioningScenario | MalformedScenario, request.param)
    return scenario.value


@pytest.fixture
def grad_input_name(request: pytest.FixtureRequest) -> str:
    """Return the name of the input the gradient test differentiates.

    Args:
        request: Pytest request carrying an ``EmbedderInputs`` field name.

    Returns:
        Field name to mark as requiring grad.
    """
    return cast(str, request.param)


def test_template_embedder_output_shape_and_dtype(
    dims: EmbedderDims,
    base_inputs: EmbedderInputs,
    run_embedder: Callable[[EmbedderInputs], Float[torch.Tensor, "..."]],
) -> None:
    """The embedding is float32 ``(B, N_res, N_res, d)`` and finite.

    Not parametrized: the output shape depends only on the sizes in
    ``dims``, which the shape contract already ties to the inputs.

    Args:
        dims: Sizes the output must match.
        base_inputs: Valid forward inputs.
        run_embedder: Runs the model.
    """
    with torch.no_grad():
        out = run_embedder(base_inputs)
    assert out.shape == (dims.batch, dims.n_res, dims.n_res, dims.d)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "scenario_edit",
    list(ConditioningScenario),
    ids=[case.name for case in ConditioningScenario],
    indirect=True,
)
def test_template_embedder_output_depends_on_each_input(
    base_inputs: EmbedderInputs,
    scenario_edit: InputEdit,
    run_embedder: Callable[[EmbedderInputs], Float[torch.Tensor, "..."]],
) -> None:
    """Changing any one input, all else fixed, changes the embedding.

    A failure means that input is not wired into the forward pass.

    Args:
        base_inputs: Valid forward inputs.
        scenario_edit: Change applied to one input.
        run_embedder: Runs the model.
    """
    with torch.no_grad():
        base = run_embedder(base_inputs)
        edited = run_embedder(scenario_edit.apply(base_inputs))
    assert not torch.allclose(base, edited)


def test_template_embedder_batch_items_are_independent(
    base_inputs: EmbedderInputs,
    run_embedder: Callable[[EmbedderInputs], Float[torch.Tensor, "..."]],
) -> None:
    """Running the first item alone matches its slice of the batched run.

    Not parametrized: one sliced item is enough to expose any cross-batch
    mixing, since every batch item has distinct inputs.

    Args:
        base_inputs: Batched forward inputs.
        run_embedder: Runs the model.
    """
    first = EmbedderInputs(
        f_distogram=base_inputs.f_distogram[:1],
        f_pseudo_beta_mask=base_inputs.f_pseudo_beta_mask[:1],
        z_ij=base_inputs.z_ij[:1],
        t=base_inputs.t[:1],
    )
    with torch.no_grad():
        batched = run_embedder(base_inputs)[:1]
        single = run_embedder(first)
    torch.testing.assert_close(batched, single, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    "grad_input_name",
    ["f_distogram", "f_pseudo_beta_mask", "z_ij"],
    indirect=True,
)
def test_template_embedder_gradient_reaches_input(
    base_inputs: EmbedderInputs,
    grad_input_name: str,
    run_embedder: Callable[[EmbedderInputs], Float[torch.Tensor, "..."]],
) -> None:
    """The summed embedding has a finite, nonzero gradient for the input.

    ``t`` is left out: in training it is a sampled noise level, never a
    learned quantity.

    Args:
        base_inputs: Valid forward inputs.
        grad_input_name: Field to differentiate with respect to.
        run_embedder: Runs the model.
    """
    leaf = cast(torch.Tensor, getattr(base_inputs, grad_input_name))
    leaf = leaf.clone().requires_grad_()
    inputs = dataclasses.replace(base_inputs, **{grad_input_name: leaf})
    out = reduce(run_embedder(inputs), "b i j d -> ", "sum")
    torch.autograd.backward([out])
    assert leaf.grad is not None
    assert torch.isfinite(leaf.grad).all()
    assert torch.count_nonzero(leaf.grad) > 0


@pytest.mark.parametrize(
    "scenario_edit",
    list(MalformedScenario),
    ids=[case.name for case in MalformedScenario],
    indirect=True,
)
def test_template_embedder_rejects_malformed_input(
    base_inputs: EmbedderInputs,
    scenario_edit: InputEdit,
    run_embedder: Callable[[EmbedderInputs], Float[torch.Tensor, "..."]],
) -> None:
    """A wrongly shaped input fails the jaxtyping contract on ``forward``.

    Args:
        base_inputs: Valid forward inputs.
        scenario_edit: Edit that breaks one input's shape.
        run_embedder: Runs the model.
    """
    malformed = scenario_edit.apply(base_inputs)
    with pytest.raises(TypeCheckError):
        _ = run_embedder(malformed)

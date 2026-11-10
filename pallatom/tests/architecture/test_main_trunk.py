"""Tests for the main trunk forward pass and its EMA.

Covers TimeFourierEmbedding, RelativePositionEncoding (Algorithm 3) and
the end-to-end MainTrunk forward pass: every output's shape and range, the
mask-token input, and finite nonzero gradients reaching every top-level
submodule from a composite training loss. Also covers EMA initialisation,
its bias-corrected update, buffer handling, DDP unwrapping, checkpoint round
trips, and the zero-copy ``swap``.

The trunk runs on 20 residues with a window of 8, so each atom has fewer
sparse neighbours than there are atoms, as in training. Inputs are
deterministic and the model is built under a fixed seed.
"""

import dataclasses
import enum
from typing import Literal, cast

import pytest
import torch
import torch.nn.functional as F
from architecture.atom_transformers import build_sparse_pairs
from architecture.losses import (
    atom_loss,
    distogram_loss_atom,
    distogram_loss_residue,
    seq_ce_loss,
    smooth_lddt_loss,
)
from architecture.main_trunk import (
    EMA,
    MainTrunk,
    PredictedOutputs,
    RelativePositionEncoding,
    TimeFourierEmbedding,
)
from einops import rearrange, repeat
from helpers.atom_utils import RESTYPE_NUM_NO_X
from helpers.data import FeaturizedBatch
from helpers.useful_objects import manual_seed
from jaxtyping import Float
from tests.patterns import sine_pattern
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from train.train_config import (
    AtomDistogramParams,
    ModelParams,
    NoiseScheduleParams,
    ResidueDistogramParams,
    TemplateDistogramParams,
)

type StateDict = dict[str, Float[torch.Tensor, "..."]]


@dataclasses.dataclass(frozen=True)
class TrunkDims:
    """Sizes the trunk, its inputs and its expected outputs share.

    Attributes:
        batch: Batch size ``B``.
        n_res: Residues ``N_res``.
        atoms_per_res: Atoms per residue.
        n_elements: Width ``E`` of the element one-hot.
        c_res: Residue single width.
        c_pair: Residue pair width.
        c_atom: Atom single width.
        c_atompair: Atom pair width.
        n_bins: Template and residue distogram bins.
        n_atom_bins: Atom distogram bins.
        k_unit: Decoder units, so intermediate outputs per forward pass.
        window_size: Sparse attention window in residues.
    """

    batch: int
    n_res: int
    atoms_per_res: int
    n_elements: int
    c_res: int
    c_pair: int
    c_atom: int
    c_atompair: int
    n_bins: int
    n_atom_bins: int
    k_unit: int
    window_size: int

    @property
    def n_atom(self) -> int:
        """Total atoms ``N_atom = n_res * atoms_per_res``.

        Returns:
            Atom count.
        """
        return self.n_res * self.atoms_per_res


@dataclasses.dataclass(frozen=True)
class SubmoduleGrads:
    """Which top-level submodules got a usable gradient from a loss.

    Attributes:
        submodules: Every top-level submodule name that owns parameters.
        without_usable_grad: Those where no parameter got a finite,
            nonzero gradient, including those left out of the graph.
    """

    submodules: tuple[str, ...]
    without_usable_grad: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class PairEncodings:
    """One pair's relative position encoding before and after an edit.

    Attributes:
        before: Encoding of the pair on the base layout.
        after: Encoding of the same pair on the edited layout.
    """

    before: Float[torch.Tensor, " c_pair"]
    after: Float[torch.Tensor, " c_pair"]


@dataclasses.dataclass(frozen=True)
class RelPosCase:
    """One input change to RelativePositionEncoding and its effect on a pair.

    The base layout has four tokens with ``residue_index = 0..3`` and one
    entity, on one chain (``asym_id = sym_id = 0``) or two chains
    (``asym_id = sym_id = (0, 0, 1, 1)``).

    Attributes:
        chains: 1 or 2, selecting the base layout.
        field: Keyword argument of the encoding to replace.
        values: Replacement values for that argument, one per token.
        i: Row of the pair to compare.
        j: Column of the pair to compare.
        effect: Whether the pair's encoding stays the ``"same"`` or
            ``"differs"`` after the change.
    """

    chains: int
    field: str
    values: tuple[int, ...]
    i: int
    j: int
    effect: Literal["same", "differs"]


class RelPosScenario(enum.Enum):
    """Algorithm 3 features that must, or must not, reach a pair's encoding.

    Members:
        cross_gap: Across chains, a residue gap of 50 changes nothing: such
            pairs always use the overflow bin.
        chain_gap: Within a chain, spreading residues 3 apart changes the
            clipped residue offset.
        entity: Splitting tokens into two entities flips the same-entity
            flag for the pair ``(0, 2)``.
        sym: Across chains, shifting ``sym_id`` changes the clipped chain
            offset.
    """

    cross_gap = RelPosCase(2, "residue_index", (0, 1, 50, 51), 0, 2, "same")
    chain_gap = RelPosCase(1, "residue_index", (0, 3, 6, 9), 0, 1, "differs")
    entity = RelPosCase(1, "entity_id", (0, 0, 1, 1), 0, 2, "differs")
    sym = RelPosCase(2, "sym_id", (0, 0, 2, 2), 0, 2, "differs")


@pytest.fixture
def dims() -> TrunkDims:
    """Return the small sizes the trunk is built and fed at.

    Returns:
        Sizes with a window smaller than the chain, so ``K < N_atom``.
    """
    return TrunkDims(
        batch=2,
        n_res=20,
        atoms_per_res=3,
        n_elements=4,
        c_res=32,
        c_pair=32,
        c_atom=32,
        c_atompair=16,
        n_bins=38,
        n_atom_bins=22,
        k_unit=2,
        window_size=8,
    )


@pytest.fixture
def model(dims: TrunkDims) -> MainTrunk:
    """Build a small MainTrunk in eval mode under a fixed seed.

    Args:
        dims: Model sizes.

    Returns:
        The trunk.
    """
    _ = manual_seed(0)
    return MainTrunk(
        model_params=ModelParams(
            window_size=dims.window_size,
            f_ref_dim=dims.atoms_per_res * (3 + dims.n_elements),
            c_atom=dims.c_atom,
            c_pair=dims.c_pair,
            c_res=dims.c_res,
            c_atompair=dims.c_atompair,
            K_unit=dims.k_unit,
            max_residues=dims.n_res,
            n_amino=RESTYPE_NUM_NO_X,
            n_blocks_atom_transformer_encoder=1,
            n_heads_atom_transformer_encoder=4,
            n_blocks_atom_transformer_decoder=1,
            n_heads_atom_transformer_decoder=4,
            n_pairformer_blocks_template_embedder=1,
            n_paiformer_heads_template_embedder=4,
        ),
        template_distogram_params=TemplateDistogramParams(n_bins=dims.n_bins),
        residue_distogram_params=ResidueDistogramParams(n_bins=dims.n_bins),
        atom_distogram_params=AtomDistogramParams(n_bins=dims.n_atom_bins),
        noise_params=NoiseScheduleParams(sigma_data=16),
    ).eval()


@pytest.fixture
def featurized_batch(
    dims: TrunkDims,
) -> FeaturizedBatch:
    """Build a deterministic FeaturizedBatch at the test sizes.

    Positions and noise levels are sine patterns; residue ``r`` has amino
    acid ``r % 20``, atom ``l`` has element ``l % E``, the template
    distogram puts pair ``(i, j)`` in bin ``(i + j) % n_bins``, and the
    sparse atom distogram target for slot ``k`` of atom ``l`` is bin
    ``(l + k) % n_atom_bins``. ``K`` comes from ``build_sparse_pairs`` at
    the model's window, as the data pipeline computes it.

    Args:
        dims: Batch sizes.

    Returns:
        The batch.
    """
    b, r, n = dims.batch, dims.n_res, dims.n_atom
    tok = torch.repeat_interleave(torch.arange(r), dims.atoms_per_res)
    _, mask = build_sparse_pairs(tok, window_size=dims.window_size)
    k = mask.shape[1]
    pair_bin = rearrange(torch.arange(r), "i -> i 1") + torch.arange(r)
    slot_bin = rearrange(torch.arange(n), "l -> l 1") + torch.arange(k)
    centres = repeat(
        torch.arange(0, n, dims.atoms_per_res),
        "r -> (r a)",
        a=dims.atoms_per_res,
    )
    r_gt = sine_pattern((b, n, 3), 0.61)
    return FeaturizedBatch(
        ref_pos=sine_pattern((b, n, 3), 0.37),
        ref_element=repeat(
            F.one_hot(
                torch.arange(n) % dims.n_elements,
                dims.n_elements,
            ).float(),
            "n e -> b n e",
            b=b,
        ),
        ref_space_uid=torch.zeros(b, n, dtype=torch.long),
        gt_res_distogram_indices=repeat(
            pair_bin % dims.n_bins,
            "i j -> b i j",
            b=b,
        ),
        noised_res_distogram=repeat(
            F.one_hot(pair_bin % dims.n_bins, dims.n_bins).float(),
            "i j c -> b i j c",
            b=b,
        ),
        f_pseudo_beta_mask=torch.ones(b, r, dtype=torch.long),
        f_residue_idx=repeat(torch.arange(r), "r -> b r", b=b),
        r_gt=r_gt,
        r_gt_noised=r_gt + sine_pattern((b, n, 3), 0.83),
        atom5_mask=torch.ones(b, n, dtype=torch.bool),
        aa_indices=repeat(
            torch.arange(r) % RESTYPE_NUM_NO_X,
            "r -> b r",
            b=b,
        ),
        t_hat=torch.tensor([0.5, 1.5]),
        t_normalized=sine_pattern((b, r, r), 0.29),
        tok_idx=repeat(tok, "n -> b n", b=b),
        center_uid=repeat(centres, "n -> b n", b=b),
        gt_atom_distogram_sparse=repeat(
            slot_bin % dims.n_atom_bins,
            "n k -> b n k",
            b=b,
        ),
        gt_atom_distogram_mask_sparse=repeat(mask, "n k -> b n k", b=b),
    )


@pytest.fixture
def outputs(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> PredictedOutputs:
    """Run the trunk on the batch without gradients.

    Args:
        model: Trunk under test.
        featurized_batch: Input batch.

    Returns:
        Every head's output.
    """
    with torch.no_grad():
        return model(featurized_batch)


@pytest.fixture
def submodule_grads(
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> SubmoduleGrads:
    """Backpropagate the training's composite loss and group the gradients.

    The loss sums the Kabsch-aligned atom loss, sequence cross-entropy,
    smooth LDDT, both distogram losses, and the decayed intermediate loss
    over every decoder unit, as the train loop does. Gradients are grouped
    by the first component of each parameter's name.

    Args:
        model: Trunk to train; switched to train mode here.
        featurized_batch: Input batch with ground truth.

    Returns:
        Every parameter-owning submodule, and those with no usable
        gradient.
    """
    x = featurized_batch
    pred: PredictedOutputs = model.train()(x)
    weight = torch.ones(x.t_hat.shape)
    mask = x.atom5_mask
    units = list(
        zip(
            pred.intermediate_denoised_coord_stack,
            pred.intermediate_pred_aa_logit_stack,
            strict=True,
        ),
    )
    intermediate = sum(
        0.99 ** (len(units) - k - 1)
        * (
            atom_loss(
                coords,
                x.r_gt,
                mask.float(),
                aa_indices=x.aa_indices,
                lambda_sigma_weight=weight,
            )
            + seq_ce_loss(logits, x.aa_indices)
        )
        for k, (coords, logits) in enumerate(units)
    )
    total = (
        atom_loss(
            pred.r_denoised,
            x.r_gt,
            mask.float(),
            aa_indices=x.aa_indices,
            lambda_sigma_weight=weight,
        ).mean()
        + seq_ce_loss(pred.seq_logits, x.aa_indices)
        + smooth_lddt_loss(pred.r_denoised, x.r_gt, mask)
        + distogram_loss_residue(
            pred.residue_distogram_logits,
            x.gt_res_distogram_indices,
            x.f_pseudo_beta_mask.float(),
        ).mean()
        + distogram_loss_atom(
            pred.atom_distogram_logits,
            x.gt_atom_distogram_sparse,
            x.gt_atom_distogram_mask_sparse,
        ).mean()
        + cast(torch.Tensor, intermediate / len(units)).mean()
    )
    torch.autograd.backward([total])
    usable: dict[str, bool] = {}
    for name, param in model.named_parameters():
        grad = param.grad
        good = (
            grad is not None
            and bool(torch.isfinite(grad).all())
            and bool(grad.abs().max() > 0)
        )
        prefix = name.split(".")[0]
        usable[prefix] = usable.get(prefix, False) or good
    return SubmoduleGrads(
        submodules=tuple(sorted(usable)),
        without_usable_grad=tuple(
            sorted(p for p, ok in usable.items() if not ok),
        ),
    )


@pytest.fixture
def rel_pos_encodings(
    rel_pos_scenario: RelPosScenario,
    dims: TrunkDims,
) -> PairEncodings:
    """Encode the scenario's base layout and its edited copy.

    Args:
        rel_pos_scenario: Base layout and the change to apply; the
            requesting test parametrizes it.
        dims: Supplies the pair width.

    Returns:
        The pair ``(i, j)``'s encoding before and after the change.
    """
    case = rel_pos_scenario.value
    _ = manual_seed(0)
    encoding = RelativePositionEncoding(dims.c_pair, r_max=4, s_max=2)
    chain = (0, 0, case.chains - 1, case.chains - 1)
    base = {
        "residue_index": (0, 1, 2, 3),
        "asym_id": chain,
        "entity_id": (0, 0, 0, 0),
        "token_index": (0, 0, 0, 0),
        "sym_id": chain,
    }
    edited = base | {case.field: case.values}
    with torch.no_grad():
        before, after = (
            encoding(
                **{
                    key: rearrange(torch.tensor(val), "n -> 1 n")
                    for key, val in layout.items()
                },
            )[0, case.i, case.j]
            for layout in (base, edited)
        )
    return PairEncodings(before=before, after=after)


def test_time_fourier_embedding_is_fixed_and_bounded(
    dims: TrunkDims,
) -> None:
    """The embedding has no parameters, frozen buffers, and outputs in [-1, 1].

    Its frequencies and phases are fixed at init, so they must not train;
    the output is a cosine, so it is bounded. Not parametrized: the
    contract does not depend on the noise levels fed in.

    Args:
        dims: Supplies the embedding width and a sequence length.
    """
    embedding = TimeFourierEmbedding(dims.c_res)
    out = embedding(3 * sine_pattern((dims.n_res,), 0.7))
    assert not list(embedding.parameters())
    assert not embedding.freqs.requires_grad
    assert not embedding.phases.requires_grad
    assert out.shape == (dims.n_res, dims.c_res)
    assert bool(((out >= -1) & (out <= 1)).all())


def test_relative_position_encoding_shape_and_finite(
    dims: TrunkDims,
    featurized_batch: FeaturizedBatch,
) -> None:
    """The encoding of a single chain is a finite ``(B, N, N, c_pair)``.

    Not parametrized: the scenario test below covers the layouts.

    Args:
        dims: Expected sizes.
        featurized_batch: Supplies residue indices.
    """
    _ = manual_seed(0)
    encoding = RelativePositionEncoding(dims.c_pair)
    res_idx = featurized_batch.f_residue_idx
    zeros = torch.zeros_like(res_idx)
    with torch.no_grad():
        out = encoding(
            residue_index=res_idx,
            asym_id=zeros,
            entity_id=zeros,
            token_index=res_idx,
            sym_id=zeros,
        )
    assert out.shape == (dims.batch, dims.n_res, dims.n_res, dims.c_pair)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "rel_pos_scenario",
    list(RelPosScenario),
    ids=[case.name for case in RelPosScenario],
)
def test_relative_position_encoding_reads_the_right_features(
    rel_pos_scenario: RelPosScenario,
    rel_pos_encodings: PairEncodings,
) -> None:
    """A feature change alters a pair's encoding exactly when it should.

    Args:
        rel_pos_scenario: Change and expected effect.
        rel_pos_encodings: The pair's encoding before and after.
    """
    x = rel_pos_encodings
    unchanged = torch.allclose(x.before, x.after)
    expect_unchanged = {"same": True, "differs": False}
    assert unchanged == expect_unchanged[rel_pos_scenario.value.effect]


def test_main_trunk_outputs_have_their_shapes_and_are_finite(
    dims: TrunkDims,
    featurized_batch: FeaturizedBatch,
    outputs: PredictedOutputs,
) -> None:
    """Every head returns a finite tensor of its documented shape.

    One forward pass checks all heads at once, since the trunk computes
    them together. The atom distogram's ``K`` axis must match the sparse
    targets it is scored against, and there is one intermediate output per
    decoder unit. Not parametrized: shapes depend only on ``dims``.

    Args:
        dims: Expected sizes.
        featurized_batch: Supplies the sparse target layout.
        outputs: Every head's output.
    """
    b, r, n = dims.batch, dims.n_res, dims.n_atom
    k = featurized_batch.gt_atom_distogram_sparse.shape[2]
    tensors = {
        "r_denoised": (outputs.r_denoised, (b, n, 3)),
        "seq_logits": (outputs.seq_logits, (b, r, RESTYPE_NUM_NO_X)),
        "residue_distogram_logits": (
            outputs.residue_distogram_logits,
            (b, r, r, dims.n_bins),
        ),
        "atom_distogram_logits": (
            outputs.atom_distogram_logits,
            (b, n, k, dims.n_atom_bins),
        ),
    }
    for unit, coords in enumerate(outputs.intermediate_denoised_coord_stack):
        tensors[f"coords_{unit}"] = (coords, (b, n, 3))
    for unit, logits in enumerate(outputs.intermediate_pred_aa_logit_stack):
        tensors[f"aa_logits_{unit}"] = (logits, (b, r, RESTYPE_NUM_NO_X))
    wrong = [
        name
        for name, (tensor, shape) in tensors.items()
        if tuple(tensor.shape) != shape or not torch.isfinite(tensor).all()
    ]
    assert len(outputs.intermediate_denoised_coord_stack) == dims.k_unit
    assert len(outputs.intermediate_pred_aa_logit_stack) == dims.k_unit
    assert not wrong


def test_main_trunk_accepts_mask_token_amino_acids(
    dims: TrunkDims,
    model: MainTrunk,
    featurized_batch: FeaturizedBatch,
) -> None:
    """A batch whose every residue is the mask token ``X`` runs cleanly.

    ``X`` has index ``RESTYPE_NUM_NO_X``, one past the predicted vocabulary,
    so an embedding sized to the vocabulary would raise ``IndexError``. Not
    parametrized: all-``X`` is the extreme of any partial masking.

    Args:
        dims: Expected sizes.
        model: Trunk under test.
        featurized_batch: Batch whose amino acids are replaced here.
    """
    masked = dataclasses.replace(
        featurized_batch,
        aa_indices=torch.full_like(
            featurized_batch.aa_indices,
            RESTYPE_NUM_NO_X,
        ),
    )
    with torch.no_grad():
        out = model(masked)
    assert out.r_denoised.shape == (dims.batch, dims.n_atom, 3)
    assert torch.isfinite(out.r_denoised).all()


def test_composite_loss_reaches_every_submodule(
    submodule_grads: SubmoduleGrads,
) -> None:
    """Every top-level submodule gets a finite, nonzero gradient.

    A submodule without one would never train, and under DDP would crash
    the unused-parameter check. Not parametrized: the composite loss is the
    single objective the trunk is trained on.

    Args:
        submodule_grads: Per-submodule gradient summary.
    """
    assert submodule_grads.submodules
    assert not submodule_grads.without_usable_grad


def test_ema_init_shadow_is_an_independent_snapshot(model: MainTrunk) -> None:
    """The shadow starts equal to the model's state but shares no storage.

    Not parametrized: it is a single invariant (the shadow is a snapshot,
    not a view); the decay arithmetic is tested below.

    Args:
        model: Trunk to snapshot; perturbed here after the EMA is built.
    """
    ema = EMA(model, decay=0.999)
    state = cast(StateDict, model.state_dict())
    matches = all(torch.equal(ema.shadow[k], v) for k, v in state.items())
    snapshot = {k: v.clone() for k, v in ema.shadow.items()}
    with torch.no_grad():
        for param in model.parameters():
            _ = param.add_(1.0)
    assert matches
    assert all(torch.equal(ema.shadow[k], v) for k, v in snapshot.items())


@pytest.mark.parametrize(
    ("decay", "step", "effective"),
    [
        pytest.param(0.999, 0, 1.0 / 10.0, id="step0_ramp_dominates"),
        pytest.param(0.999, 89, 90.0 / 99.0, id="mid_ramp_below_target"),
        pytest.param(0.9, 10_000, 0.9, id="late_step_target_decay_dominates"),
    ],
)
def test_ema_update_blends_by_bias_corrected_decay(
    model: MainTrunk,
    decay: float,
    step: int,
    effective: float,
) -> None:
    """``update`` blends by ``min(decay, (step + 1) / (step + 10))``.

    Covers the ramp's dominant regime at step 0, a mid-ramp step still
    below the target decay, and a late step where the target has taken
    over.

    Args:
        model: Trunk whose parameters are shifted by 1 here.
        decay: Target decay.
        step: Optimizer step passed to ``update``.
        effective: Hand-computed effective decay.
    """
    ema = EMA(model, decay=decay)
    name, param = next(model.named_parameters())
    before = ema.shadow[name].clone()
    with torch.no_grad():
        _ = param.add_(1.0)
    ema.update(model, step=step)
    expected = effective * before + (1 - effective) * param.detach()
    torch.testing.assert_close(ema.shadow[name], expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("count", [7, -3], ids=["positive", "negative"])
def test_ema_copies_integer_buffers_instead_of_blending(
    model: MainTrunk,
    count: int,
) -> None:
    """A non-float buffer's new value is copied into the shadow as is.

    MainTrunk has no integer buffers today, so this test registers a
    synthetic int64 counter to reach the copy branch. Blending 0 and
    ``count`` at decay 0.1 would give a different, non-integer value.

    Args:
        model: Trunk to which an integer buffer is added here.
        count: Value the counter is set to before the update.
    """
    model.register_buffer("step_count", torch.tensor(0))
    ema = EMA(model, decay=0.999)
    _ = model.get_buffer("step_count").fill_(count)
    ema.update(model, step=0)
    assert ema.shadow["step_count"].item() == count


def test_ema_state_dict_round_trips(model: MainTrunk) -> None:
    """``load_state_dict`` restores a saved shadow exactly.

    Not parametrized: one round trip of a shadow that differs from the
    model covers serialization.

    Args:
        model: Trunk whose parameters are shifted here.
    """
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        for param in model.parameters():
            _ = param.add_(1.0)
    ema.update(model, step=500)
    saved = {k: v.clone() for k, v in ema.state_dict().items()}
    restored = EMA(model, decay=0.999)
    restored.load_state_dict(saved)
    assert all(torch.equal(restored.shadow[k], v) for k, v in saved.items())


@pytest.mark.usefixtures("single_process_gloo_group")
def test_ema_unwraps_ddp_for_init_update_and_swap(model: MainTrunk) -> None:
    """Given a DDP wrapper, EMA keys and swaps against the inner trunk.

    ``DDP.state_dict()`` prefixes every key with ``module.``; the shadow
    must instead match the plain model's keys, which is what
    ``save_checkpoint`` writes, through init, update and swap. Not
    parametrized: one wrapped model exercises all three paths.

    Args:
        model: Trunk to wrap in a one-rank DDP group.
    """
    plain_keys = set(model.state_dict())
    ddp_model = DDP(model)
    ema = EMA(ddp_model, decay=0.999)
    keys_at_init = set(ema.shadow)
    name, param = next(model.named_parameters())
    shadow_before = ema.shadow[name].clone()
    with torch.no_grad():
        _ = param.add_(1.0)
    ema.update(ddp_model, step=0)
    keys_after_update = set(ema.shadow)
    shadow_after_update = ema.shadow[name].clone()
    ema.swap(ddp_model)
    assert keys_at_init == keys_after_update == plain_keys
    assert not torch.equal(shadow_after_update, shadow_before)
    assert torch.equal(
        dict(model.named_parameters())[name],
        shadow_after_update,
    )


def test_ema_swap_exchanges_parameters_and_buffers_with_shadow(
    model: MainTrunk,
) -> None:
    """After one ``swap``, model and shadow hold each other's prior values.

    Both a parameter and a buffer (``time_fourier.freqs``) are perturbed
    first so the two sides differ. Not parametrized: one parameter and one
    buffer cover both lists ``swap`` walks.

    Args:
        model: Trunk perturbed here after the EMA is built.
    """
    ema = EMA(model, decay=0.999)
    name, param = next(model.named_parameters())
    buffer_name = "time_fourier.freqs"
    shadow_param = ema.shadow[name].clone()
    shadow_buffer = ema.shadow[buffer_name].clone()
    with torch.no_grad():
        _ = param.add_(1.0)
        _ = model.time_fourier.freqs.add_(1.0)
    live_param, live_buffer = param.clone(), model.time_fourier.freqs.clone()
    ema.swap(model)
    assert torch.equal(param, shadow_param)
    assert torch.equal(model.time_fourier.freqs, shadow_buffer)
    assert torch.equal(ema.shadow[name], live_param)
    assert torch.equal(ema.shadow[buffer_name], live_buffer)


def test_ema_swap_twice_restores_both_sides(model: MainTrunk) -> None:
    """``swap`` is its own inverse: two calls leave model and shadow as is.

    Not parametrized: the round trip is the same for any divergence.

    Args:
        model: Trunk perturbed here after the EMA is built.
    """
    ema = EMA(model, decay=0.999)
    with torch.no_grad():
        for param in model.parameters():
            _ = param.add_(1.0)
    model_before = {
        k: v.clone() for k, v in cast(StateDict, model.state_dict()).items()
    }
    shadow_before = {k: v.clone() for k, v in ema.shadow.items()}
    ema.swap(model)
    ema.swap(model)
    model_after = cast(StateDict, model.state_dict())
    assert all(torch.equal(model_after[k], v) for k, v in model_before.items())
    assert all(torch.equal(ema.shadow[k], v) for k, v in shadow_before.items())


def test_ema_swap_reuses_storage_and_keeps_optimizer_state_valid(
    model: MainTrunk,
) -> None:
    """``swap`` moves storage pointers and keeps every Parameter object.

    Adam keys its state by Parameter object, so after a swap the model's
    current parameters must still be the keys of the optimizer's state,
    and the parameter must now own the shadow's former storage, proving no
    copy was made. Not parametrized: one optimizer step populates state for
    every parameter.

    Args:
        model: Trunk to optimize one step and then swap.
    """
    optimizer = Adam(model.parameters(), lr=1e-3)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    _ = optimizer.step()  # pyright: ignore[reportUnknownMemberType]
    ema = EMA(model, decay=0.999)
    name, param = next(model.named_parameters())
    shadow_ptr = ema.shadow[name].data_ptr()
    ema.swap(model)
    assert param.data_ptr() == shadow_ptr
    assert all(p in optimizer.state for p in model.parameters())
    assert dict(model.named_parameters())[name] is param

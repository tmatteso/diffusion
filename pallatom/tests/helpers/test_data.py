"""Tests for data types, featurization utilities, and dataset / data loading.

Covers the jaxtyping contracts of FeaturizedItem and FeaturizedBatch; the
Distogram's bin assignment, masking, symmetry and shape; the closed form of
sinusoidal_encoding and ref_pos_for_residue's frame; featurize_single_item
and featurize_batch outputs and index maps; FeaturizeCollate after a pickle
round trip; ProteinDataset selection, padding, truncation and pickling;
ProteinShardDataset metadata, caching and parsing; the ShardDataLoader
lifecycle; make_bucketed_data_loaders end to end; and DatasetSplitsManifest
parsing.

Every synthetic protein has deterministic backbone coordinates, so no test
depends on a random generator except through ``featurize``'s noise, which
is seeded in its fixtures.
"""

import dataclasses
import enum
import json
import math
import pathlib
import pickle
from collections.abc import Callable, Mapping
from concurrent.futures import Executor
from typing import Protocol, cast

import numpy as np
import pytest
import torch
from einops import rearrange, reduce, repeat
from helpers.atom_utils import ATOMS_PER_RES, Protein, restype_order
from helpers.data import (
    DatasetSplitsManifest,
    Distogram,
    FeaturizeCollate,
    FeaturizedBatch,
    FeaturizedItem,
    ProteinDataset,
    ProteinShardDataset,
    ShardBudgetParameters,
    ShardDataLoader,
    ShardMetadata,
    build_distogram_module,
    featurize_batch,
    featurize_single_item,
    make_bucketed_data_loaders,
    ref_pos_for_residue,
    sinusoidal_encoding,
)
from helpers.useful_objects import manual_seed
from jaxtyping import TypeCheckError
from tests.conftest import PatternFactory
from train.train_config import LoaderConfig as EvalLoaderConfig
from train.train_config import TrainArgs, TrainConfig, TrainLoaderConfig

type Entry = dict[str, object]
type EntryFactory = Callable[[str, int], Entry]
type JsonlWriter = Callable[[pathlib.Path, list[Entry]], pathlib.Path]
type ShardFactory = Callable[[list[str], list[int]], ProteinShardDataset]


class ArgsFactory(Protocol):
    """Builds TrainArgs over a fresh dataset of ``n`` proteins."""

    def __call__(self, n: int, *, debug_run: bool) -> TrainArgs:
        """Write the dataset and return arguments pointing at it.

        Args:
            n: Number of proteins.
            debug_run: Value of ``TrainArgs.debug_run``.

        Returns:
            Arguments pointing at the new dataset.
        """
        ...  # pylint: disable=unnecessary-ellipsis


@dataclasses.dataclass(frozen=True)
class Distograms:
    """The three distogram modules featurization takes.

    Attributes:
        template: Self-conditioning template distogram.
        residue: Residue-level ground-truth distogram.
        atom: Sparse atom-level distogram.
    """

    template: Distogram
    residue: Distogram
    atom: Distogram


@dataclasses.dataclass(frozen=True)
class FieldBreak:
    """A featurized field replaced by a tensor with one axis wrong.

    Attributes:
        field: Name of the dataclass field to replace.
        malform: Maps the valid tensor to the malformed one.
    """

    field: str
    malform: Callable[[torch.Tensor], torch.Tensor]


class ItemBreak(enum.Enum):
    """FeaturizedItem fields whose shape contract a replacement breaks.

    Members:
        coords_flat: ``r_gt`` loses its xyz axis.
        coords_four: ``r_gt`` gains a fourth coordinate.
        element_width: ``ref_element`` has 3 element classes, not 4.
        atom_count_drift: ``atom5_mask`` has one atom more than ``r_gt``.
        residue_count_drift: ``aa_indices`` has one residue fewer.
    """

    coords_flat = FieldBreak("r_gt", lambda t: t[:, 0])
    coords_four = FieldBreak("r_gt", lambda t: torch.cat([t, t[:, :1]], -1))
    element_width = FieldBreak("ref_element", lambda t: t[:, :3])
    atom_count_drift = FieldBreak("atom5_mask", lambda t: torch.cat([t, t[:1]]))
    residue_count_drift = FieldBreak("aa_indices", lambda t: t[1:])


class BatchBreak(enum.Enum):
    """FeaturizedBatch fields whose shape contract a replacement breaks.

    Members:
        unbatched_ref_pos: ``ref_pos`` loses its batch axis.
        coords_four: ``r_gt`` gains a fourth coordinate.
        batch_drift: ``t_hat`` has one batch item fewer.
    """

    unbatched_ref_pos = FieldBreak("ref_pos", lambda t: t[0])
    coords_four = FieldBreak("r_gt", lambda t: torch.cat([t, t[..., :1]], -1))
    batch_drift = FieldBreak("t_hat", lambda t: t[1:])


@dataclasses.dataclass(frozen=True)
class BinCase:
    """A pair distance and where a 16-bin, 2-22 angstrom Distogram puts it.

    Bin edges are ``2 + 1.25 k``; a distance lands in the first bin whose
    upper edge it does not exceed.

    Attributes:
        distance: Distance between the two atoms, in angstroms.
        expected_bin: Index of the one-hot bin.
        overflow: Whether the Distogram has an overflow bin.
        valid: Whether the pair mask keeps the pair.
    """

    distance: float
    expected_bin: int
    overflow: bool
    valid: bool


class BinScenario(enum.Enum):
    """Distances across the bin range, with and without an overflow bin.

    Members:
        coincident: 0 angstroms falls below the first edge, into bin 0.
        interior: 9 angstroms: ``(9 - 2) / 1.25 = 5.6``, so bin 5.
        top_edge: 22 angstroms is the last edge, so bin 15, still valid.
        far_overflow: 1000 angstroms goes to overflow bin 16 and stays valid.
        far_clipped: Without overflow, 1000 angstroms is clipped to bin 15
            and masked out.
    """

    coincident = BinCase(0.0, 0, overflow=False, valid=True)
    interior = BinCase(9.0, 5, overflow=False, valid=True)
    top_edge = BinCase(22.0, 15, overflow=False, valid=True)
    far_overflow = BinCase(1000.0, 16, overflow=True, valid=True)
    far_clipped = BinCase(1000.0, 15, overflow=False, valid=False)


@pytest.fixture
def tcfg() -> TrainConfig:
    """Return the default training configuration.

    Returns:
        ``TrainConfig()``.
    """
    return TrainConfig()


@pytest.fixture
def distograms(tcfg: TrainConfig) -> Distograms:
    """Build the template, residue and atom distograms from ``tcfg``.

    Args:
        tcfg: Configuration holding the three distogram parameter sets.

    Returns:
        The three distograms, in eval mode.
    """
    return Distograms(
        template=build_distogram_module(tcfg.distogram_template).eval(),
        residue=build_distogram_module(tcfg.distogram_residue).eval(),
        atom=build_distogram_module(tcfg.distogram_atom).eval(),
    )


@pytest.fixture
def protein() -> Protein:
    """Return a 12-residue protein with every atom present.

    Atom ``a`` of residue ``i`` sits at ``(3.8 i + 0.1 a, sin(i + a), 0)``,
    and the sequence is ``ACDEFGHIKLMN``.

    Returns:
        The protein.
    """
    n = 12
    i = rearrange(np.arange(n, dtype=np.float64), "i -> i 1")
    a = rearrange(np.arange(37, dtype=np.float64), "a -> 1 a")
    positions = np.stack(
        [3.8 * i + 0.1 * a, np.sin(i + a), np.zeros((n, 37))],
        axis=-1,
    )
    return Protein(
        atom_positions=positions,
        aatype=np.array([restype_order[c] for c in "ACDEFGHIKLMN"]),
        atom_mask=np.ones((n, 37)),
        residue_index=np.arange(n),
        chain_index=np.zeros(n, dtype=np.intp),
        b_factors=np.zeros((n, 37)),
    )


@pytest.fixture
def featurized_item(
    protein: Protein,
    tcfg: TrainConfig,
    distograms: Distograms,
) -> FeaturizedItem:
    """Featurize ``protein`` alone under a fixed seed.

    Seed 1 keeps the sampled noise level inside the configured range, so
    ``t_normalized`` stays in ``[0, 1]``.

    Args:
        protein: Protein to featurize.
        tcfg: Configuration.
        distograms: Template, residue and atom distograms.

    Returns:
        The featurized item.
    """
    _ = manual_seed(1)
    return featurize_single_item(
        protein,
        distograms.template,
        distograms.residue,
        distograms.atom,
        tcfg,
        max_seq_len_in_batch=len(protein.aatype),
    )


@pytest.fixture
def featurized_batch(
    protein: Protein,
    tcfg: TrainConfig,
    distograms: Distograms,
) -> FeaturizedBatch:
    """Featurize two copies of ``protein`` as a batch under a fixed seed.

    Args:
        protein: Protein to featurize twice.
        tcfg: Configuration.
        distograms: Template, residue and atom distograms.

    Returns:
        The featurized batch.
    """
    _ = manual_seed(1)
    return featurize_batch(
        [protein, protein],
        tcfg,
        distograms.template,
        distograms.residue,
        distograms.atom,
    )


@pytest.fixture
def break_case(request: pytest.FixtureRequest) -> FieldBreak:
    """Return the ``FieldBreak`` of the enum member routed in via param.

    Args:
        request: Pytest request carrying an ``ItemBreak`` or ``BatchBreak``.

    Returns:
        The field and reshape to apply.
    """
    return cast(ItemBreak | BatchBreak, request.param).value


@pytest.fixture
def bin_scenario(request: pytest.FixtureRequest) -> BinScenario:
    """Return the ``BinScenario`` member routed in via indirect param.

    Args:
        request: Pytest request carrying the scenario in ``param``.

    Returns:
        The scenario under test.
    """
    return cast(BinScenario, request.param)


@pytest.fixture(scope="module")
def make_entry() -> EntryFactory:
    """Return a factory for JSONL protein entries with fixed coordinates.

    ``make(name, n)`` gives an all-alanine entry whose residue ``i`` has its
    CA at ``(3.8 i, 0, 0)`` and N, C and O at fixed offsets from it.

    Module-scoped so the module-scoped loader lifecycle can use it; it
    holds no state.

    Returns:
        The factory.
    """
    offsets = {
        "N": (-1.2, 0.6, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (1.2, 0.6, 0.0),
        "O": (1.4, 1.8, 0.0),
    }

    def make(name: str, n: int) -> Entry:
        """Build one entry.

        Args:
            name: Entry name.
            n: Residue count.

        Returns:
            Entry with ``name``, ``seq`` and ``coords``.
        """
        coords = {
            atom: [[3.8 * i + dx, dy, dz] for i in range(n)]
            for atom, (dx, dy, dz) in offsets.items()
        }
        return {"name": name, "seq": "A" * n, "coords": coords}

    return make


@pytest.fixture(scope="module")
def write_jsonl() -> JsonlWriter:
    """Return a function that writes entries as JSON lines.

    Module-scoped for the same reason as ``make_entry``.

    Returns:
        ``write(path, entries)``, which returns ``path``.
    """

    def write(path: pathlib.Path, entries: list[Entry]) -> pathlib.Path:
        """Write one JSON object per line.

        Args:
            path: Destination file.
            entries: Objects to serialise.

        Returns:
            The path written.
        """
        _ = path.write_text("".join(json.dumps(e) + "\n" for e in entries))
        return path

    return write


@pytest.fixture
def dataset_jsonl(
    tmp_path: pathlib.Path,
    make_entry: EntryFactory,
    write_jsonl: JsonlWriter,
) -> pathlib.Path:
    """Write five 6-residue entries named ``1aa.A`` to ``5ee.A``.

    Args:
        tmp_path: Per-test directory.
        make_entry: Entry factory.
        write_jsonl: JSONL writer.

    Returns:
        Path of the JSONL file.
    """
    names = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A"]
    entries = [make_entry(name, 6) for name in names]
    return write_jsonl(tmp_path / "proteins.jsonl", entries)


@pytest.fixture
def train_dataset(dataset_jsonl: pathlib.Path) -> ProteinDataset:
    """Load the first three entries, padded to 8 residues.

    Args:
        dataset_jsonl: Five-entry JSONL.

    Returns:
        The dataset.
    """
    names = ["1aa.A", "2bb.A", "3cc.A"]
    return ProteinDataset(dataset_jsonl, names, max_seq_length=8)


@pytest.fixture
def shard_budget(tmp_path: pathlib.Path) -> ShardBudgetParameters:
    """Return single-rank, single-worker shard parameters under ``tmp_path``.

    Args:
        tmp_path: Per-test directory for shards and logs.

    Returns:
        The parameters.
    """
    return ShardBudgetParameters(
        shard_dir=tmp_path / "shards",
        structlog_path=tmp_path / "train.jsonl",
        token_budget=512,
        max_seq_len=128,
        seed=0,
        n_threads=1,
        world_size=1,
        rank=0,
        n_proteins_in_shard=100,
        noise_magnitude=0,
        num_workers=1,
    )


@pytest.fixture
def make_shard_dataset(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
    make_entry: EntryFactory,
    write_jsonl: JsonlWriter,
) -> ShardFactory:
    """Return a factory for ProteinShardDatasets over fresh JSONL files.

    ``make(names, lengths)`` writes one entry per name, then builds a
    dataset sharding them into a directory unique to that call.

    Args:
        tmp_path: Per-test directory.
        shard_budget: Base shard parameters.
        make_entry: Entry factory.
        write_jsonl: JSONL writer.

    Returns:
        The factory.
    """
    calls: list[int] = []

    def make(names: list[str], lengths: list[int]) -> ProteinShardDataset:
        """Write the entries and shard them.

        Args:
            names: Entry names.
            lengths: Residue count per entry.

        Returns:
            The dataset.
        """
        calls.append(len(calls))
        entries = [
            make_entry(n, k) for n, k in zip(names, lengths, strict=True)
        ]
        jsonl = write_jsonl(tmp_path / f"p{len(calls)}.jsonl", entries)
        budget = dataclasses.replace(
            shard_budget,
            shard_dir=tmp_path / f"shards{len(calls)}",
        )
        return ProteinShardDataset(
            budget_parameters=budget,
            names=names,
            dataset_jsonl=jsonl,
        )

    return make


@pytest.fixture(scope="module")
def bucketed_cfg() -> TrainConfig:
    """Return a small bucketed-loader configuration with a 512-token budget.

    Module-scoped because the module-scoped loader lifecycle needs it; the
    config is a frozen pydantic model, so sharing it is safe.

    Returns:
        The configuration.
    """
    return TrainConfig(
        train_loader=TrainLoaderConfig(
            batch_size=2,
            max_seq_length=8,
            token_budget=512,
            num_workers=1,
            epoch_prefetch_depth=1,
            n_shards=1,
        ),
        test_loader=EvalLoaderConfig(batch_size=2, max_seq_length=8),
    )


@pytest.fixture
def make_train_args(
    tmp_path: pathlib.Path,
    make_entry: EntryFactory,
    write_jsonl: JsonlWriter,
) -> ArgsFactory:
    """Return a factory for TrainArgs over a dataset of ``n`` proteins.

    ``make(n, debug_run)`` writes ``n`` 6-residue entries, puts them all in
    the train split (and the first in validation and test), and shares one
    shard directory across calls, as successive runs on one machine do.

    Args:
        tmp_path: Per-test directory.
        make_entry: Entry factory.
        write_jsonl: JSONL writer.

    Returns:
        The factory.
    """

    def make(n: int, *, debug_run: bool) -> TrainArgs:
        """Write a dataset of ``n`` proteins and point TrainArgs at it.

        Args:
            n: Number of proteins.
            debug_run: Value of ``TrainArgs.debug_run``.

        Returns:
            The arguments.
        """
        names = [f"{i:04d}p.A" for i in range(n)]
        jsonl = tmp_path / f"proteins_{n}.jsonl"
        _ = write_jsonl(jsonl, [make_entry(name, 6) for name in names])
        splits = tmp_path / f"splits_{n}.json"
        _ = splits.write_text(
            json.dumps(
                {"train": names, "validation": names[:1], "test": names[:1]},
            ),
        )
        return TrainArgs(
            dataset_jsonl=jsonl,
            shard_dir=tmp_path / "shards",
            keys_for_splits_json=splits,
            config=tmp_path / "config.json",
            structlog_jsonl=tmp_path / "train.jsonl",
            ddp=False,
            debug_run=debug_run,
        )

    return make


@pytest.fixture
def debug_train_size(
    request: pytest.FixtureRequest,
    bucketed_cfg: TrainConfig,
    make_train_args: ArgsFactory,
) -> int:
    """Build a debug-run loader over 300 proteins and return its train size.

    With a True param, a full-dataset loader is built first in the same
    shard directory, leaving a 300-protein shard cache behind.

    Args:
        request: Pytest request carrying whether to build the full loader.
        bucketed_cfg: Loader configuration.
        make_train_args: TrainArgs factory.

    Returns:
        Number of proteins in the debug-run train dataset.
    """
    if cast(bool, request.param):
        _ = make_bucketed_data_loaders(
            cfg=bucketed_cfg,
            extra_train_args=make_train_args(300, debug_run=False),
        )
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=make_train_args(300, debug_run=True),
    )
    return len(cast(ProteinDataset, train_loader.dataset))


@pytest.fixture
def pickled_dataset(
    request: pytest.FixtureRequest,
    train_dataset: ProteinDataset,
) -> ProteinDataset:
    """Pickle round-trip ``train_dataset``, optionally after reading from it.

    Args:
        request: Pytest request carrying whether to read a sample first,
            which opens the dataset's file handle.
        train_dataset: Dataset to pickle.

    Returns:
        The restored copy.
    """
    if cast(bool, request.param):
        _ = train_dataset[0]
    raw = pickle.dumps(train_dataset)
    return cast(ProteinDataset, pickle.loads(raw))  # noqa: S301


@dataclasses.dataclass(frozen=True)
class ShardLoaderLifecycle:
    """What one ShardDataLoader did over its life, recorded by a fixture.

    Attributes:
        len_after_construction: ``len(loader)`` right after construction.
        epoch_before_iter: ``loader.epoch`` right after construction.
        epoch_after_iter: ``loader.epoch`` after one full iteration.
        batches: Number of batches the iteration yielded.
        process_executor_closed: True if ``__del__`` shut the process
            executor down, so it refuses new work.
        queue_watcher_closed: True if ``__del__`` shut the queue watcher
            down, so it refuses new work.
    """

    len_after_construction: int
    epoch_before_iter: int
    epoch_after_iter: int
    batches: int
    process_executor_closed: bool
    queue_watcher_closed: bool


def refuses_new_work(executor: Executor) -> bool:
    """Return True if an executor has been shut down.

    A plain function because the module-scoped lifecycle fixture calls it
    after the loader is torn down, where no fixture can be requested.

    Args:
        executor: Executor to probe with a no-op task.

    Returns:
        True if submitting raises the RuntimeError of a closed executor.
    """
    try:
        _ = executor.submit(int)
    except RuntimeError:
        return True
    return False


@pytest.fixture(scope="module")
def shard_loader_lifecycle(
    tmp_path_factory: pytest.TempPathFactory,
    bucketed_cfg: TrainConfig,
    make_entry: EntryFactory,
    write_jsonl: JsonlWriter,
) -> ShardLoaderLifecycle:
    """Take one ShardDataLoader through its whole life and record the result.

    Constructing a ShardDataLoader spawns a worker that re-imports torch,
    which takes seconds, so the tests share this module-scoped run. The
    result is a frozen snapshot of each stage rather than the live loader,
    so the tests stay order-independent even though iterating and shutting
    down both change the loader. ``__del__`` is called directly because the
    watcher thread can still hold a reference to the loader after
    iteration, which would make garbage collection unpredictable.

    Args:
        tmp_path_factory: Factory for the run's temporary directory.
        bucketed_cfg: Loader configuration.
        make_entry: Entry factory.
        write_jsonl: JSONL writer.

    Returns:
        ShardLoaderLifecycle recorded across construction, one iteration,
        and shutdown.
    """
    work = tmp_path_factory.mktemp("shard_loader")
    budget = ShardBudgetParameters(
        shard_dir=work / "shards",
        structlog_path=work / "train.jsonl",
        token_budget=512,
        max_seq_len=128,
        seed=0,
        n_threads=1,
        world_size=1,
        rank=0,
        n_proteins_in_shard=100,
        noise_magnitude=0,
        num_workers=1,
    )
    entries = [make_entry("p1", 10), make_entry("p2", 20)]
    jsonl = write_jsonl(work / "p.jsonl", entries)
    loader = ShardDataLoader(
        dataset=ProteinShardDataset(
            budget_parameters=budget,
            names=["p1", "p2"],
            dataset_jsonl=jsonl,
        ),
        budget=budget,
        tcfg=bucketed_cfg,
        distogram_template=build_distogram_module(
            bucketed_cfg.distogram_template,
        ).eval(),
        distogram_residue=build_distogram_module(
            bucketed_cfg.distogram_residue,
        ).eval(),
        distogram_atom=build_distogram_module(
            bucketed_cfg.distogram_atom,
        ).eval(),
    )
    len_after_construction = len(loader)
    epoch_before_iter = loader.epoch
    batches = len(list(loader))
    epoch_after_iter = loader.epoch
    loader.__del__()  # pylint: disable=unnecessary-dunder-call
    return ShardLoaderLifecycle(
        len_after_construction=len_after_construction,
        epoch_before_iter=epoch_before_iter,
        epoch_after_iter=epoch_after_iter,
        batches=batches,
        process_executor_closed=refuses_new_work(loader.process_executor),
        queue_watcher_closed=refuses_new_work(loader.queue_watcher),
    )


@pytest.mark.parametrize(
    "break_case",
    list(ItemBreak),
    ids=[case.name for case in ItemBreak],
    indirect=True,
)
def test_featurized_item_rejects_a_malformed_field(
    featurized_item: FeaturizedItem,
    break_case: FieldBreak,
) -> None:
    """Rebuilding a valid item with one malformed field fails the contract.

    Every other field stays valid, so the error can only come from the
    replaced one.

    Args:
        featurized_item: Valid item from ``featurize_single_item``.
        break_case: Field and reshape to apply.
    """
    old = cast(torch.Tensor, getattr(featurized_item, break_case.field))
    with pytest.raises(TypeCheckError):
        _ = dataclasses.replace(
            featurized_item,
            **{break_case.field: break_case.malform(old)},
        )


@pytest.mark.parametrize(
    "break_case",
    list(BatchBreak),
    ids=[case.name for case in BatchBreak],
    indirect=True,
)
def test_featurized_batch_rejects_a_malformed_field(
    featurized_batch: FeaturizedBatch,
    break_case: FieldBreak,
) -> None:
    """Rebuilding a valid batch with one malformed field fails the contract.

    Args:
        featurized_batch: Valid batch from ``featurize_batch``.
        break_case: Field and reshape to apply.
    """
    old = cast(torch.Tensor, getattr(featurized_batch, break_case.field))
    with pytest.raises(TypeCheckError):
        _ = dataclasses.replace(
            featurized_batch,
            **{break_case.field: break_case.malform(old)},
        )


@pytest.mark.parametrize(
    "bin_scenario",
    list(BinScenario),
    ids=[case.name for case in BinScenario],
    indirect=True,
)
def test_distogram_assigns_each_distance_its_bin(
    bin_scenario: BinScenario,
) -> None:
    """Two atoms ``distance`` apart get the expected one-hot bin and mask.

    Args:
        bin_scenario: Distance, overflow setting and expected outcome.
    """
    case = bin_scenario.value
    disto = Distogram(n_bins=16, overflow_bin=case.overflow).eval()
    coords = torch.tensor([[0.0, 0, 0], [case.distance, 0, 0]])
    with torch.no_grad():
        onehot, mask = disto(coords)
    expected = torch.zeros(16 + case.overflow)
    expected[case.expected_bin] = 1
    assert torch.equal(onehot[0, 1], expected)
    assert bool(mask[0, 1]) == case.valid


@pytest.mark.parametrize("overflow", [False, True], ids=["clipped", "overflow"])
def test_distogram_is_a_symmetric_one_hot_over_its_bins(
    overflow: bool,  # noqa: FBT001
    make_pattern: PatternFactory,
) -> None:
    """Every pair gets exactly one bin, and ``(i, j)`` matches ``(j, i)``.

    The overflow bin adds one class to the last axis.

    Args:
        overflow: Whether the Distogram has an overflow bin.
        make_pattern: Builds 12 coordinates spread over about 20 angstroms.
    """
    disto = Distogram(n_bins=16, overflow_bin=overflow).eval()
    coords = 20 * make_pattern((12, 3), 0.37)
    with torch.no_grad():
        onehot, mask = disto(coords)
    assert onehot.shape == (12, 12, 16 + overflow)
    assert torch.equal(
        reduce(onehot, "i j b -> i j", "sum"),
        torch.ones(12, 12),
    )
    assert torch.equal(onehot, rearrange(onehot, "i j b -> j i b"))
    assert torch.equal(mask, rearrange(mask, "i j -> j i"))


def test_distogram_masks_every_pair_of_an_invalid_atom(
    make_pattern: PatternFactory,
) -> None:
    """An atom marked invalid drops its whole row and column of pairs.

    Every coordinate is within 2 angstroms of every other, so no distance
    cutoff applies and the mask alone decides. Not parametrized: which
    atom is invalid does not matter.

    Args:
        make_pattern: Builds 12 nearby coordinates.
    """
    disto = Distogram(n_bins=16, overflow_bin=False).eval()
    valid = torch.ones(12, dtype=torch.bool)
    valid[0] = False
    with torch.no_grad():
        _, mask = disto(make_pattern((12, 3), 0.37), valid)
    expected = rearrange(valid, "i -> i 1") & rearrange(valid, "j -> 1 j")
    assert torch.equal(mask, expected)


@pytest.mark.parametrize(
    ("positions", "dim", "expected"),
    [
        pytest.param((0.0,), 4, (0.0, 0.0, 1.0, 1.0), id="origin"),
        pytest.param(
            (1.0,),
            4,
            (math.sin(1), math.sin(1e-4), math.cos(1), math.cos(1e-4)),
            id="position_one",
        ),
        pytest.param(
            (2.0,),
            4,
            (math.sin(2), math.sin(2e-4), math.cos(2), math.cos(2e-4)),
            id="two_freqs",
        ),
    ],
)
def test_sinusoidal_encoding_matches_its_closed_form(
    positions: tuple[float, ...],
    dim: int,
    expected: tuple[float, ...],
) -> None:
    """Position ``p`` maps to ``sin(p f)`` then ``cos(p f)`` per frequency.

    Frequencies are ``10000 ** (-k / (dim/2 - 1))``, so ``dim = 4`` gives
    ``1`` and ``1e-4``.

    Args:
        positions: One residue position.
        dim: Embedding width.
        expected: The ``dim`` encoded values.
    """
    out = sinusoidal_encoding(torch.tensor([positions]), dim=dim)
    torch.testing.assert_close(out, torch.tensor([[expected]]))


@pytest.mark.parametrize(
    ("resname", "has_cb"),
    [
        pytest.param("ALA", True, id="alanine"),
        pytest.param("GLY", False, id="glycine_has_no_cb"),
        pytest.param("TRP", True, id="tryptophan"),
    ],
)
def test_ref_pos_for_residue_puts_ca_at_the_origin(
    resname: str,
    has_cb: bool,  # noqa: FBT001
) -> None:
    """CA is at the origin, N and C are off it, and a missing CB is zero.

    Glycine is the only canonical residue without a CB, so its CB row
    falls back to ``(0, 0, 0)``.

    Args:
        resname: Three-letter residue name.
        has_cb: Whether the residue has a CB atom.
    """
    pos = ref_pos_for_residue(resname)
    assert pos.shape == (ATOMS_PER_RES, 3)
    assert torch.equal(pos[1], torch.zeros(3))
    assert bool((pos[[0, 2]].abs().sum(-1) > 1).all())
    assert bool(pos[4].abs().sum() > 0) == has_cb


def test_featurize_single_item_shapes_ranges_and_index_maps(
    featurized_item: FeaturizedItem,
    protein: Protein,
    tcfg: TrainConfig,
) -> None:
    """The item has every field at its shape, finite, with exact index maps.

    Atom ``l`` belongs to residue ``l // 5``, its centre atom is that
    residue's CA at ``5 r + 1``, and the noise level ``t_normalized`` lies
    in ``[0, 1]``. Not parametrized: one protein covers the layout.

    Args:
        featurized_item: The featurized protein.
        protein: The 12-residue input.
        tcfg: Supplies the template distogram width.
    """
    x = featurized_item
    n_res = len(protein.aatype)
    n_atom = n_res * ATOMS_PER_RES
    residue = torch.arange(n_res).repeat_interleave(ATOMS_PER_RES)
    shapes = {
        "ref_pos": (n_atom, 3),
        "ref_element": (n_atom, 4),
        "ref_space_uid": (n_atom,),
        "gt_res_distogram_indices": (n_res, n_res),
        "noised_res_distogram": (n_res, n_res, tcfg.distogram_template.n_bins),
        "f_pseudo_beta_mask": (n_res,),
        "tok_idx": (n_atom,),
        "center_uid": (n_atom,),
    }
    fields: Mapping[str, torch.Tensor] = {
        f.name: cast(torch.Tensor, getattr(x, f.name))
        for f in dataclasses.fields(x)
    }
    assert {k: tuple(fields[k].shape) for k in shapes} == shapes
    assert all(torch.isfinite(t.float()).all() for t in fields.values())
    assert bool(x.t_hat > 0)
    assert bool(((x.t_normalized >= 0) & (x.t_normalized <= 1)).all())
    assert torch.equal(x.ref_space_uid, residue)
    assert torch.equal(x.tok_idx, residue)
    assert torch.equal(x.center_uid, ATOMS_PER_RES * residue + 1)


def test_featurize_batch_stacks_items_along_a_batch_axis(
    featurized_batch: FeaturizedBatch,
    featurized_item: FeaturizedItem,
) -> None:
    """Each batch field is ``(2, *item_shape)``, with each item's index maps.

    Not parametrized: the item test above covers the per-field contents.

    Args:
        featurized_batch: Two copies of the protein, featurized together.
        featurized_item: The protein featurized alone.
    """
    batch, item = featurized_batch, featurized_item
    pairs = [
        (
            cast(torch.Tensor, getattr(batch, f.name)).shape,
            cast(torch.Tensor, getattr(item, f.name)).shape,
        )
        for f in dataclasses.fields(item)
    ]
    assert all(b == (2, *i) for b, i in pairs)
    assert torch.equal(batch.tok_idx, repeat(item.tok_idx, "n -> 2 n"))
    assert torch.equal(batch.center_uid, repeat(item.center_uid, "n -> 2 n"))


def test_featurize_collate_works_after_a_pickle_round_trip(
    protein: Protein,
    tcfg: TrainConfig,
    distograms: Distograms,
    featurized_batch: FeaturizedBatch,
) -> None:
    """A collate restored from pickle, as DataLoader workers get it, works.

    It must produce a FeaturizedBatch shaped like ``featurize_batch``'s.
    Not parametrized: pickling is the one transport DataLoader uses.

    Args:
        protein: Protein to collate twice.
        tcfg: Configuration.
        distograms: Template, residue and atom distograms.
        featurized_batch: Reference batch of the same two proteins.
    """
    collate = FeaturizeCollate(
        tcfg=tcfg,
        distogram_template=distograms.template,
        distogram_residue=distograms.residue,
        distogram_atom=distograms.atom,
    )
    raw = pickle.dumps(collate)
    restored = cast(FeaturizeCollate, pickle.loads(raw))  # noqa: S301
    out = restored([protein, protein])
    assert isinstance(out, FeaturizedBatch)
    assert out.r_gt.shape == featurized_batch.r_gt.shape


@pytest.mark.parametrize(
    ("names", "expected"),
    [
        pytest.param(["1aa.A", "2bb.A", "3cc.A"], 3, id="three_listed"),
        pytest.param(["1aa.A"], 1, id="one_listed"),
        pytest.param([], 0, id="none_listed"),
    ],
)
def test_protein_dataset_holds_only_listed_names(
    dataset_jsonl: pathlib.Path,
    names: list[str],
    expected: int,
) -> None:
    """The dataset's length is the number of listed names, and each loads.

    Args:
        dataset_jsonl: Five-entry JSONL.
        names: Names to load.
        expected: Expected length.
    """
    dataset = ProteinDataset(dataset_jsonl, names, max_seq_length=8)
    assert len(dataset) == expected
    assert all(isinstance(dataset[i], Protein) for i in range(expected))


def test_protein_dataset_pads_samples_to_max_length(
    train_dataset: ProteinDataset,
) -> None:
    """A 6-residue entry comes back padded to 8, with float64 coordinates.

    The two padded residues have every atom masked out. Not parametrized:
    truncation is the next test.

    Args:
        train_dataset: Dataset of 6-residue entries padded to 8.
    """
    sample = train_dataset[0]
    assert sample.atom_positions.shape == (8, 37, 3)
    assert sample.atom_mask.shape == (8, 37)
    assert sample.residue_index.shape == (8,)
    assert sample.aatype.shape == (8,)
    assert sample.atom_positions.dtype == np.float64
    assert sample.atom_mask.dtype == np.float64
    assert np.issubdtype(sample.aatype.dtype, np.integer)
    assert not sample.atom_mask[6:].any()


def test_protein_dataset_truncates_long_entries(
    tmp_path: pathlib.Path,
    make_entry: EntryFactory,
    write_jsonl: JsonlWriter,
) -> None:
    """A 100-residue entry comes back cut to ``max_seq_length = 8``.

    Not parametrized: padding is the test above.

    Args:
        tmp_path: Per-test directory.
        make_entry: Entry factory.
        write_jsonl: JSONL writer.
    """
    path = write_jsonl(tmp_path / "long.jsonl", [make_entry("long.A", 100)])
    sample = ProteinDataset(path, ["long.A"], max_seq_length=8)[0]
    assert sample.aatype.shape == (8,)
    assert sample.atom_positions.shape == (8, 37, 3)


@pytest.mark.parametrize(
    "pickled_dataset",
    [False, True],
    ids=["unopened", "opened"],
    indirect=True,
)
def test_protein_dataset_survives_pickling(
    train_dataset: ProteinDataset,
    pickled_dataset: ProteinDataset,
) -> None:
    """The dataset pickles whether or not its file handle is open yet.

    DataLoader workers receive it by pickle, sometimes after the parent
    has already read from it.

    Args:
        train_dataset: The original dataset.
        pickled_dataset: Its pickle round trip.
    """
    restored = pickled_dataset
    assert len(restored) == len(train_dataset)
    np.testing.assert_array_equal(
        restored[0].atom_positions,
        train_dataset[0].atom_positions,
    )


def test_shard_metadata_records_the_build_parameters(
    make_shard_dataset: ShardFactory,
) -> None:
    """The metadata sidecar records the token budget and shard size used.

    Not parametrized: the fields are copied from the budget as is.

    Args:
        make_shard_dataset: Shard dataset factory.
    """
    dataset = make_shard_dataset(["p1", "p2"], [10, 20])
    meta = ShardMetadata.model_validate_json(
        dataset.shard_metadata_path.read_text(),
    )
    assert meta.token_budget == dataset.token_budget
    assert meta.shard_size == dataset.n_proteins_in_shard
    assert meta.n_shards >= 1


@pytest.mark.parametrize(
    ("names_a", "names_b"),
    [
        pytest.param(["p1", "p2"], ["p1", "p2"], id="same_names"),
        pytest.param(["p2", "p1"], ["p1", "p2"], id="same_names_reordered"),
        pytest.param(["p1"], ["p1", "p2"], id="different_names"),
    ],
)
def test_shard_metadata_names_hash_identifies_the_name_set(
    make_shard_dataset: ShardFactory,
    names_a: list[str],
    names_b: list[str],
) -> None:
    """Two builds share a names hash exactly when they shard the same names.

    The hash is over the sorted names, so order does not matter.

    Args:
        make_shard_dataset: Shard dataset factory.
        names_a: Names of the first build.
        names_b: Names of the second build.
    """
    hashes = [
        ShardMetadata.model_validate_json(
            make_shard_dataset(
                names,
                [10] * len(names),
            ).shard_metadata_path.read_text(),
        ).names_hash
        for names in (names_a, names_b)
    ]
    assert (hashes[0] == hashes[1]) == (sorted(names_a) == sorted(names_b))


def test_shard_dataset_reuses_existing_metadata(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
    make_entry: EntryFactory,
    write_jsonl: JsonlWriter,
) -> None:
    """A second dataset in the same shard directory leaves the shards alone.

    The metadata file is not rewritten, so its mtime is unchanged. Not
    parametrized: the cache check reads one file.

    Args:
        tmp_path: Per-test directory.
        shard_budget: Shard parameters, shared by both builds.
        make_entry: Entry factory.
        write_jsonl: JSONL writer.
    """
    jsonl = write_jsonl(tmp_path / "p.jsonl", [make_entry("p1", 10)])
    first = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=jsonl,
    )
    before = first.shard_metadata_path.stat().st_mtime_ns
    _ = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=jsonl,
    )
    assert first.shard_metadata_path.stat().st_mtime_ns == before


@pytest.mark.parametrize("encoded", [False, True], ids=["dict", "bytes"])
def test_parse_protein_centres_ca_and_maps_unknown_residues_to_x(
    make_shard_dataset: ShardFactory,
    make_entry: EntryFactory,
    encoded: bool,  # noqa: FBT001
) -> None:
    """Parsing gives a CA centroid at the origin and ``X`` for unknown codes.

    The JSON may arrive as bytes or as a decoded dict, as webdataset hands
    over either; both must parse the same way.

    Args:
        make_shard_dataset: Shard dataset factory.
        make_entry: Entry factory.
        encoded: Whether to pass the entry as JSON bytes.
    """
    dataset = make_shard_dataset(["p1"], [10])
    entry = make_entry("p1", 10) | {"seq": "ZZZZZAAAAA"}
    raw: object = json.dumps(entry).encode() if encoded else entry
    parsed = dataset.parse_protein({"json": raw})
    ca = torch.tensor(parsed.atom_positions[:, 1])
    ca_mask = rearrange(torch.tensor(parsed.atom_mask[:, 1]), "n -> n 1")
    centroid = reduce(ca * ca_mask, "n d -> d", "sum") / ca_mask.sum()
    aatype = torch.tensor(parsed.aatype)
    torch.testing.assert_close(centroid, torch.zeros(3, dtype=torch.float64))
    assert torch.equal(aatype[:5], torch.full((5,), restype_order["X"]))
    assert torch.equal(aatype[5:], torch.full((5,), restype_order["A"]))


def test_shard_data_loader_counts_and_advances_epochs(
    shard_loader_lifecycle: ShardLoaderLifecycle,
) -> None:
    """``len`` is known at construction and one pass advances the epoch.

    The pass yields exactly ``len`` batches. Not parametrized: the shared
    module-scoped run covers one lifecycle.

    Args:
        shard_loader_lifecycle: Snapshot of the run.
    """
    x = shard_loader_lifecycle
    assert x.len_after_construction > 0
    assert x.batches == x.len_after_construction
    assert (x.epoch_before_iter, x.epoch_after_iter) == (0, 1)


def test_shard_data_loader_del_shuts_down_executors(
    shard_loader_lifecycle: ShardLoaderLifecycle,
) -> None:
    """``__del__`` shuts down both executors, so neither accepts new work.

    Not parametrized: both executors are checked.

    Args:
        shard_loader_lifecycle: Snapshot of the run.
    """
    assert shard_loader_lifecycle.process_executor_closed
    assert shard_loader_lifecycle.queue_watcher_closed


def test_bucketed_train_loader_yields_featurized_batches(
    bucketed_cfg: TrainConfig,
    make_train_args: ArgsFactory,
) -> None:
    """The bucketed training loader yields FeaturizedBatch, not raw proteins.

    Not parametrized: one batch shows the collate is wired in.

    Args:
        bucketed_cfg: Loader configuration.
        make_train_args: TrainArgs factory.
    """
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=make_train_args(5, debug_run=False),
    )
    assert isinstance(next(iter(train_loader)), FeaturizedBatch)


@pytest.mark.parametrize(
    ("debug_train_size", "expected"),
    [
        pytest.param(False, 252, id="fresh_cache"),
        pytest.param(True, 252, id="after_full_dataset_cache"),
    ],
    indirect=["debug_train_size"],
)
def test_bucketed_debug_run_caps_train_split_at_252(
    debug_train_size: int,
    expected: int,
) -> None:
    """``debug_run`` trains on the first 252 proteins of a 300-protein split.

    With a shard cache already built for all 300 proteins in the same
    directory, the debug run must still cap at 252 rather than reuse the
    full cache.

    Args:
        debug_train_size: Train-set size of the debug-run loader.
        expected: The cap ``make_bucketed_data_loaders`` applies.
    """
    assert debug_train_size == expected


@pytest.mark.parametrize(
    ("extra", "cath_nodes"),
    [
        pytest.param({}, {}, id="cath_nodes_default_empty"),
        pytest.param(
            {"cath_nodes": {"p1": ["1.20.5"]}},
            {"p1": ["1.20.5"]},
            id="cath_nodes_populated",
        ),
        pytest.param({"unknown_field": 42}, {}, id="unknown_field_dropped"),
    ],
)
def test_dataset_splits_manifest_parses_splits_and_cath_nodes(
    extra: dict[str, object],
    cath_nodes: dict[str, list[str]],
) -> None:
    """The manifest reads the three splits, ``cath_nodes``, and ignores extras.

    Args:
        extra: Fields added to the three splits.
        cath_nodes: Expected ``cath_nodes``.
    """
    data = {"train": ["p1"], "validation": ["p2"], "test": ["p3"]} | extra
    manifest = DatasetSplitsManifest.model_validate(data)
    assert (manifest.train, manifest.validation, manifest.test) == (
        ["p1"],
        ["p2"],
        ["p3"],
    )
    assert manifest.cath_nodes == cath_nodes

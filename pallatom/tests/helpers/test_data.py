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

import abc
import dataclasses
import enum
import gc
import json
import multiprocessing
import multiprocessing.queues
import pathlib
from collections.abc import Mapping
from concurrent.futures import Executor
from math import cos, sin
from typing import cast, override

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
    normalized_time_from_sigma,
    null_template_distogram,
    ref_pos_for_residue,
    sinusoidal_encoding,
)
from helpers.useful_objects import manual_seed
from jaxtyping import Bool, Float, Int, Shaped, TypeCheckError
from tests.patterns import sine_pattern
from train.train_config import (
    LoaderConfig,
    NoiseScheduleParams,
    TrainArgs,
    TrainConfig,
    TrainLoaderConfig,
)


def queue_round_trip[T](obj: T) -> T:
    """Send ``obj`` through a ``multiprocessing.Queue`` and return the copy.

    The queue pickles ``obj`` on ``put`` and unpickles it on ``get``, the
    same transport a DataLoader uses to hand datasets and collate functions
    to worker processes, so the copy is what a worker would receive.

    Args:
        obj: Object to send.

    Returns:
        The unpickled copy.
    """
    channel: multiprocessing.queues.Queue[T] = multiprocessing.Queue()
    channel.put(obj)
    copy = channel.get(timeout=60)
    channel.close()
    channel.join_thread()
    return copy


def protein_entry(name: str, n: int) -> dict[str, object]:
    """Build an all-alanine JSONL protein entry with fixed coordinates.

    Residue ``i`` has its CA at ``(3.8 i, 0, 0)`` and N, C and O at fixed
    offsets from it.

    Args:
        name: Entry name.
        n: Residue count.

    Returns:
        Entry with ``name``, ``seq`` and ``coords``.
    """
    offsets = {
        "N": (-1.2, 0.6, 0.0),
        "CA": (0.0, 0.0, 0.0),
        "C": (1.2, 0.6, 0.0),
        "O": (1.4, 1.8, 0.0),
    }
    coords = {
        atom: [[3.8 * i + dx, dy, dz] for i in range(n)]
        for atom, (dx, dy, dz) in offsets.items()
    }
    return {"name": name, "seq": "A" * n, "coords": coords}


def write_jsonl(
    path: pathlib.Path,
    entries: list[dict[str, object]],
) -> pathlib.Path:
    """Write one JSON object per line.

    Args:
        path: Destination file.
        entries: Objects to serialise.

    Returns:
        The path written.
    """
    _ = path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return path


def build_shard_dataset(
    work_dir: pathlib.Path,
    budget: ShardBudgetParameters,
    names: list[str],
    lengths: list[int],
) -> ProteinShardDataset:
    """Write one entry per name under ``work_dir`` and shard them there.

    Args:
        work_dir: Directory for the JSONL file and the shards; give each
            build its own.
        budget: Base shard parameters; the shard directory is replaced.
        names: Entry names.
        lengths: Residue count per entry.

    Returns:
        The dataset.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    entries = [protein_entry(n, k) for n, k in zip(names, lengths, strict=True)]
    jsonl = write_jsonl(work_dir / "p.jsonl", entries)
    return ProteinShardDataset(
        budget_parameters=dataclasses.replace(
            budget,
            shard_dir=work_dir / "shards",
        ),
        names=names,
        dataset_jsonl=jsonl,
    )


def write_train_args(
    work_dir: pathlib.Path,
    n: int,
    *,
    debug_run: bool,
) -> TrainArgs:
    """Write a dataset of ``n`` proteins and point TrainArgs at it.

    Every protein goes in the train split, and the first also in
    validation and test. All calls share ``work_dir / "shards"``, as
    successive runs on one machine do.

    Args:
        work_dir: Directory for the dataset, splits and shards.
        n: Number of 6-residue proteins.
        debug_run: Value of ``TrainArgs.debug_run``.

    Returns:
        The arguments.
    """
    names = [f"{i:04d}p.A" for i in range(n)]
    jsonl = work_dir / f"proteins_{n}.jsonl"
    _ = write_jsonl(jsonl, [protein_entry(name, 6) for name in names])
    splits = work_dir / f"splits_{n}.json"
    _ = splits.write_text(
        json.dumps(
            {"train": names, "validation": names[:1], "test": names[:1]},
        ),
    )
    return TrainArgs(
        dataset_jsonl=jsonl,
        shard_dir=work_dir / "shards",
        keys_for_splits_json=splits,
        config=work_dir / "config.json",
        structlog_jsonl=work_dir / "train.jsonl",
        ddp=False,
        debug_run=debug_run,
    )


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
class FieldBreak(abc.ABC):
    """A featurized field replaced by a tensor with one axis wrong.

    Attributes:
        field: Name of the dataclass field to replace.
    """

    field: str

    @abc.abstractmethod
    def malform(
        self,
        tensor: Shaped[torch.Tensor, "..."],
    ) -> Shaped[torch.Tensor, "..."]:
        """Map the field's valid tensor to the malformed one.

        Args:
            tensor: The field's valid value.

        Returns:
            The malformed replacement.
        """


@dataclasses.dataclass(frozen=True)
class SelectBreak(FieldBreak):
    """Keep one index of an axis, removing that axis.

    Attributes:
        dim: Axis to select from.
        index: Position along ``dim`` to keep.
    """

    dim: int
    index: int

    @override
    def malform(
        self,
        tensor: Shaped[torch.Tensor, "..."],
    ) -> Shaped[torch.Tensor, "..."]:
        """Select ``index`` along ``dim``.

        Args:
            tensor: The field's valid value.

        Returns:
            ``tensor`` with axis ``dim`` removed.
        """
        return tensor.select(self.dim, self.index)


@dataclasses.dataclass(frozen=True)
class DropFirstBreak(FieldBreak):
    """Remove the first entry of an axis, shortening it by one.

    Attributes:
        dim: Axis to shorten.
    """

    dim: int

    @override
    def malform(
        self,
        tensor: Shaped[torch.Tensor, "..."],
    ) -> Shaped[torch.Tensor, "..."]:
        """Drop index 0 along ``dim``.

        Args:
            tensor: The field's valid value.

        Returns:
            ``tensor`` with ``dim`` one shorter.
        """
        return tensor.narrow(self.dim, 1, tensor.shape[self.dim] - 1)


@dataclasses.dataclass(frozen=True)
class KeepFirstBreak(FieldBreak):
    """Keep only the first entries of an axis.

    Attributes:
        dim: Axis to cut.
        count: Number of entries to keep.
    """

    dim: int
    count: int

    @override
    def malform(
        self,
        tensor: Shaped[torch.Tensor, "..."],
    ) -> Shaped[torch.Tensor, "..."]:
        """Keep the first ``count`` entries along ``dim``.

        Args:
            tensor: The field's valid value.

        Returns:
            ``tensor`` with ``dim`` cut to ``count``.
        """
        return tensor.narrow(self.dim, 0, self.count)


@dataclasses.dataclass(frozen=True)
class RepeatFirstBreak(FieldBreak):
    """Append a copy of an axis's first entry, lengthening it by one.

    Attributes:
        dim: Axis to lengthen.
    """

    dim: int

    @override
    def malform(
        self,
        tensor: Shaped[torch.Tensor, "..."],
    ) -> Shaped[torch.Tensor, "..."]:
        """Append index 0 along ``dim`` to the end.

        Args:
            tensor: The field's valid value.

        Returns:
            ``tensor`` with ``dim`` one longer.
        """
        return torch.cat([tensor, tensor.narrow(self.dim, 0, 1)], dim=self.dim)


class ItemBreak(enum.Enum):
    """FeaturizedItem fields whose shape contract a replacement breaks.

    Members:
        coords_flat: ``r_gt`` loses its xyz axis.
        coords_four: ``r_gt`` gains a fourth coordinate.
        element_width: ``ref_element`` has 3 element classes, not 4.
        atom_count_drift: ``atom5_mask`` has one atom more than ``r_gt``.
        residue_count_drift: ``aa_indices`` has one residue fewer.
    """

    coords_flat = SelectBreak("r_gt", dim=1, index=0)
    coords_four = RepeatFirstBreak("r_gt", dim=-1)
    element_width = KeepFirstBreak("ref_element", dim=1, count=3)
    atom_count_drift = RepeatFirstBreak("atom5_mask", dim=0)
    residue_count_drift = DropFirstBreak("aa_indices", dim=0)


class BatchBreak(enum.Enum):
    """FeaturizedBatch fields whose shape contract a replacement breaks.

    Members:
        unbatched_ref_pos: ``ref_pos`` loses its batch axis.
        coords_four: ``r_gt`` gains a fourth coordinate.
        batch_drift: ``t_hat`` has one batch item fewer.
    """

    unbatched_ref_pos = SelectBreak("ref_pos", dim=0, index=0)
    coords_four = RepeatFirstBreak("r_gt", dim=-1)
    batch_drift = DropFirstBreak("t_hat", dim=0)


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


@dataclasses.dataclass(frozen=True)
class OverflowCase:
    """A Distogram's overflow setting and the classes it then has.

    Attributes:
        overflow_bin: Whether the Distogram has an overflow bin.
        n_classes: Length of the one-hot axis: 16 bins, plus one for
            overflow.
    """

    overflow_bin: bool
    n_classes: int


class OverflowScenario(enum.Enum):
    """A 16-bin Distogram with and without an overflow bin.

    Members:
        clipped: No overflow bin; far pairs clip to the last bin.
        overflow: An extra bin for pairs past the last edge.
    """

    clipped = OverflowCase(overflow_bin=False, n_classes=16)
    overflow = OverflowCase(overflow_bin=True, n_classes=17)


@dataclasses.dataclass(frozen=True)
class SinusoidCase:
    """One position, an embedding width, and its closed-form encoding.

    Attributes:
        position: Residue position ``p``.
        dim: Embedding width; ``4`` gives frequencies ``1`` and ``1e-4``.
        expected: ``sin(p f)`` then ``cos(p f)`` per frequency.
    """

    position: float
    dim: int
    expected: tuple[float, ...]


class SinusoidScenario(enum.Enum):
    """Positions whose four-wide encoding is checked against its formula.

    Members:
        origin: Position 0 gives sines of 0 and cosines of 1.
        position_one: Position 1.
        two_freqs: Position 2, so both frequencies move.
    """

    origin = SinusoidCase(0.0, 4, (0.0, 0.0, 1.0, 1.0))
    position_one = SinusoidCase(1.0, 4, (sin(1), sin(1e-4), cos(1), cos(1e-4)))
    two_freqs = SinusoidCase(2.0, 4, (sin(2), sin(2e-4), cos(2), cos(2e-4)))


@dataclasses.dataclass(frozen=True)
class CbCase:
    """A residue and whether its reference frame has a CB atom.

    Attributes:
        resname: Three-letter residue name.
        has_cb: Whether the residue has a CB, so its CB row is nonzero.
    """

    resname: str
    has_cb: bool


class CbScenario(enum.Enum):
    """Residues with and without a CB.

    Members:
        alanine: Smallest residue with a CB.
        glycine_has_no_cb: The only canonical residue without one.
        tryptophan: Largest residue.
    """

    alanine = CbCase("ALA", has_cb=True)
    glycine_has_no_cb = CbCase("GLY", has_cb=False)
    tryptophan = CbCase("TRP", has_cb=True)


@dataclasses.dataclass(frozen=True)
class JsonFormCase:
    """How webdataset hands a protein's JSON to ``parse_protein``.

    Attributes:
        as_bytes: Encoded JSON bytes when true, a decoded dict when false.
    """

    as_bytes: bool


class JsonForm(enum.Enum):
    """The two forms ``parse_protein`` must accept.

    Members:
        dict: Already decoded.
        bytes: Raw JSON bytes.
    """

    dict = JsonFormCase(as_bytes=False)
    bytes = JsonFormCase(as_bytes=True)


@dataclasses.dataclass(frozen=True)
class NamesHashCase:
    """Two shard builds and whether their names hashes must match.

    Attributes:
        names_a: Names of the first build.
        names_b: Names of the second build.
        same_hash: Whether the two builds share a names hash; true exactly
            when they shard the same set of names.
    """

    names_a: tuple[str, ...]
    names_b: tuple[str, ...]
    same_hash: bool


class NamesHashScenario(enum.Enum):
    """Pairs of name lists for the shard names hash.

    Members:
        same_names: Identical lists.
        reordered: The same names in another order, so the hash, taken
            over the sorted names, still matches.
        different_names: The first build lacks ``p2``.
    """

    same_names = NamesHashCase(("p1", "p2"), ("p1", "p2"), same_hash=True)
    reordered = NamesHashCase(("p2", "p1"), ("p1", "p2"), same_hash=True)
    different_names = NamesHashCase(("p1",), ("p1", "p2"), same_hash=False)


@dataclasses.dataclass(frozen=True)
class NamesHashes:
    """The names hashes of two shard builds.

    Attributes:
        first: Hash of the first build.
        second: Hash of the second build.
    """

    first: str
    second: str


@dataclasses.dataclass(frozen=True)
class ManifestCase:
    """A splits manifest's optional fields and the ``cath_nodes`` it parses.

    Attributes:
        cath_nodes: ``cath_nodes`` written to the manifest, omitted when
            empty; also the value the parsed manifest must hold.
        extra: Any further fields written to the manifest.
    """

    cath_nodes: dict[str, list[str]]
    extra: dict[str, object]


class ManifestScenario(enum.Enum):
    """Optional manifest fields beyond the three splits.

    Members:
        cath_nodes_default_empty: No ``cath_nodes``, so it defaults empty.
        cath_nodes_populated: ``cath_nodes`` given for one protein.
        unknown_field_dropped: An unknown field is ignored.
    """

    cath_nodes_default_empty = ManifestCase({}, {})
    cath_nodes_populated = ManifestCase({"p1": ["1.20.5"]}, {})
    unknown_field_dropped = ManifestCase({}, {"unknown_field": 42})


@dataclasses.dataclass(frozen=True)
class PickleCase:
    """Whether the dataset is read from before it is pickled.

    Attributes:
        read_first: Read one sample first, which opens the file handle.
    """

    read_first: bool


class PickleState(enum.Enum):
    """The dataset's file-handle state when it is pickled.

    Members:
        unopened: Pickled straight after construction.
        opened: Pickled after reading one sample.
    """

    unopened = PickleCase(read_first=False)
    opened = PickleCase(read_first=True)


@dataclasses.dataclass(frozen=True)
class DebugCacheCase:
    """Shard-cache state before a debug run and the train size it must get.

    Attributes:
        prebuilt: Build a full-dataset loader first in the same shard
            directory, leaving a 300-protein shard cache behind.
        expected: Train-set size the debug run must cap at.
    """

    prebuilt: bool
    expected: int


class DebugCacheScenario(enum.Enum):
    """Shard-cache states a debug run must ignore.

    Members:
        fresh_cache: No shard cache exists yet.
        after_full_dataset_cache: A 300-protein cache already exists.
    """

    fresh_cache = DebugCacheCase(prebuilt=False, expected=252)
    after_full_dataset_cache = DebugCacheCase(prebuilt=True, expected=252)


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
def dataset_jsonl(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write five 6-residue entries named ``1aa.A`` to ``5ee.A``.

    Args:
        tmp_path: Per-test directory.

    Returns:
        Path of the JSONL file.
    """
    names = ["1aa.A", "2bb.A", "3cc.A", "4dd.A", "5ee.A"]
    entries = [protein_entry(name, 6) for name in names]
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
def shard_dataset(
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> ProteinShardDataset:
    """Shard two entries, ``p1`` with 10 residues and ``p2`` with 20.

    Args:
        tmp_path: Per-test directory.
        shard_budget: Base shard parameters.

    Returns:
        The dataset.
    """
    return build_shard_dataset(tmp_path, shard_budget, ["p1", "p2"], [10, 20])


@pytest.fixture
def names_hashes(
    names_hash_scenario: NamesHashScenario,
    tmp_path: pathlib.Path,
    shard_budget: ShardBudgetParameters,
) -> NamesHashes:
    """Build the scenario's two shard datasets and read their names hashes.

    Each build gets its own directory, so neither reuses the other's cache.

    Args:
        names_hash_scenario: Names of the two builds; the requesting test
            parametrizes it.
        tmp_path: Per-test directory.
        shard_budget: Base shard parameters.

    Returns:
        Both builds' names hashes.
    """
    case = names_hash_scenario.value
    hashes = [
        ShardMetadata.model_validate_json(
            build_shard_dataset(
                tmp_path / tag,
                shard_budget,
                list(names),
                [10] * len(names),
            ).shard_metadata_path.read_text(),
        ).names_hash
        for tag, names in (("a", case.names_a), ("b", case.names_b))
    ]
    return NamesHashes(first=hashes[0], second=hashes[1])


@pytest.fixture
def long_entry_jsonl(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write one 100-residue entry named ``long.A``.

    Args:
        tmp_path: Per-test directory.

    Returns:
        Path of the JSONL file.
    """
    return write_jsonl(tmp_path / "long.jsonl", [protein_entry("long.A", 100)])


@pytest.fixture
def single_entry_jsonl(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write one 10-residue entry named ``p1``.

    Args:
        tmp_path: Per-test directory.

    Returns:
        Path of the JSONL file.
    """
    return write_jsonl(tmp_path / "p.jsonl", [protein_entry("p1", 10)])


@pytest.fixture
def unknown_residue_json(json_form: JsonForm) -> object:
    """Return entry ``p1`` with five unknown codes, in the scenario's form.

    The sequence is ``ZZZZZAAAAA``: five codes outside the alphabet, then
    five alanines.

    Args:
        json_form: Dict or bytes; the requesting test parametrizes it.

    Returns:
        The entry as a dict or as encoded JSON bytes.
    """
    entry = protein_entry("p1", 10) | {"seq": "ZZZZZAAAAA"}
    if json_form.value.as_bytes:
        return json.dumps(entry).encode()
    return entry


@pytest.fixture
def spread_coords() -> Float[torch.Tensor, "12 3"]:
    """Return 12 coordinates spread over about 20 angstroms.

    Returns:
        A sine pattern of shape ``(12, 3)`` scaled by 20.
    """
    return 20 * sine_pattern((12, 3), 0.37)


@pytest.fixture
def near_coords() -> Float[torch.Tensor, "12 3"]:
    """Return 12 coordinates all within 2 angstroms of each other.

    Returns:
        A sine pattern of shape ``(12, 3)`` in ``[-1, 1]``.
    """
    return sine_pattern((12, 3), 0.37)


@pytest.fixture
def manifest(manifest_scenario: ManifestScenario) -> DatasetSplitsManifest:
    """Parse a manifest of one protein per split plus the scenario's fields.

    Args:
        manifest_scenario: Optional fields to add; the requesting test
            parametrizes it.

    Returns:
        The parsed manifest.
    """
    case = manifest_scenario.value
    data: dict[str, object] = {
        "train": ["p1"],
        "validation": ["p2"],
        "test": ["p3"],
    }
    if case.cath_nodes:
        data["cath_nodes"] = case.cath_nodes
    return DatasetSplitsManifest.model_validate(data | case.extra)


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
        test_loader=LoaderConfig(batch_size=2, max_seq_length=8),
    )


@pytest.fixture
def bucketed_train_args(tmp_path: pathlib.Path) -> TrainArgs:
    """Return TrainArgs over a fresh five-protein dataset.

    Args:
        tmp_path: Per-test directory.

    Returns:
        Non-debug arguments.
    """
    return write_train_args(tmp_path, 5, debug_run=False)


@pytest.fixture
def debug_train_size(
    debug_cache_scenario: DebugCacheScenario,
    bucketed_cfg: TrainConfig,
    tmp_path: pathlib.Path,
) -> int:
    """Build a debug-run loader over 300 proteins and return its train size.

    When the scenario asks for it, a full-dataset loader is built first in
    the same shard directory, leaving a 300-protein shard cache behind.

    Args:
        debug_cache_scenario: Whether to build the full loader first; the
            requesting test parametrizes it.
        bucketed_cfg: Loader configuration.
        tmp_path: Per-test directory shared by both loaders.

    Returns:
        Number of proteins in the debug-run train dataset.
    """
    if debug_cache_scenario.value.prebuilt:
        _ = make_bucketed_data_loaders(
            cfg=bucketed_cfg,
            extra_train_args=write_train_args(tmp_path, 300, debug_run=False),
        )
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=write_train_args(tmp_path, 300, debug_run=True),
    )
    return len(cast(ProteinDataset, train_loader.dataset))


@pytest.fixture
def pickled_dataset(
    pickle_state: PickleState,
    train_dataset: ProteinDataset,
) -> ProteinDataset:
    """Pickle round-trip ``train_dataset``, optionally after reading from it.

    Args:
        pickle_state: Whether to read a sample first, which opens the
            dataset's file handle; the requesting test parametrizes it.
        train_dataset: Dataset to pickle.

    Returns:
        The restored copy.
    """
    if pickle_state.value.read_first:
        _ = train_dataset[0]
    return queue_round_trip(train_dataset)


@dataclasses.dataclass(frozen=True)
class ExpectedItemShapes:
    """Shapes ``featurize_single_item`` must give the per-atom and pair fields.

    Attributes:
        ref_pos: ``(N_atom, 3)``.
        ref_element: ``(N_atom, 4)``.
        ref_space_uid: ``(N_atom,)``.
        gt_res_distogram_indices: ``(N_res, N_res)``.
        noised_res_distogram: ``(N_res, N_res, n_template_bins)``.
        f_pseudo_beta_mask: ``(N_res,)``.
        tok_idx: ``(N_atom,)``.
        center_uid: ``(N_atom,)``.
    """

    ref_pos: tuple[int, ...]
    ref_element: tuple[int, ...]
    ref_space_uid: tuple[int, ...]
    gt_res_distogram_indices: tuple[int, ...]
    noised_res_distogram: tuple[int, ...]
    f_pseudo_beta_mask: tuple[int, ...]
    tok_idx: tuple[int, ...]
    center_uid: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class MetadataMtimes:
    """Shard metadata modification times around a second dataset build.

    Attributes:
        before: ``st_mtime_ns`` after the first build.
        after: ``st_mtime_ns`` after the second build over the same names.
    """

    before: int
    after: int


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
    entries = [protein_entry("p1", 10), protein_entry("p2", 20)]
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
    process_executor = loader.process_executor
    queue_watcher = loader.queue_watcher
    del loader
    _ = gc.collect()
    return ShardLoaderLifecycle(
        len_after_construction=len_after_construction,
        epoch_before_iter=epoch_before_iter,
        epoch_after_iter=epoch_after_iter,
        batches=batches,
        process_executor_closed=refuses_new_work(process_executor),
        queue_watcher_closed=refuses_new_work(queue_watcher),
    )


@pytest.fixture
def bin_distogram(bin_scenario: BinScenario) -> Distogram:
    """Build a 16-bin distogram, with an overflow bin if the case asks.

    Args:
        bin_scenario: Overflow setting; the requesting test parametrizes
            it.

    Returns:
        The module in eval mode.
    """
    return Distogram(n_bins=16, overflow_bin=bin_scenario.value.overflow).eval()


@pytest.fixture
def pair_at_distance(bin_scenario: BinScenario) -> Float[torch.Tensor, "2 3"]:
    """Place two atoms the scenario's distance apart along ``x``.

    Args:
        bin_scenario: Supplies the distance; the requesting test
            parametrizes it.

    Returns:
        Coordinates of the two atoms.
    """
    return torch.tensor([[0.0, 0, 0], [bin_scenario.value.distance, 0, 0]])


@pytest.fixture
def expected_bin_one_hot(
    bin_scenario: BinScenario,
) -> Float[torch.Tensor, " C"]:
    """One-hot vector over the distogram's classes at the expected bin.

    Args:
        bin_scenario: Expected bin and overflow setting; the requesting
            test parametrizes it.

    Returns:
        A one-hot vector of length ``16 + overflow``.
    """
    case = bin_scenario.value
    expected = torch.zeros(16 + case.overflow)
    expected[case.expected_bin] = 1
    return expected


@pytest.fixture
def overflow_distogram(overflow_scenario: OverflowScenario) -> Distogram:
    """Build a 16-bin distogram with the scenario's overflow setting.

    Args:
        overflow_scenario: Overflow setting; the requesting test
            parametrizes it.

    Returns:
        The module in eval mode.
    """
    overflow_bin = overflow_scenario.value.overflow_bin
    return Distogram(n_bins=16, overflow_bin=overflow_bin).eval()


@pytest.fixture
def no_overflow_distogram() -> Distogram:
    """Build a 16-bin distogram without an overflow bin.

    Returns:
        The module in eval mode.
    """
    return Distogram(n_bins=16, overflow_bin=False).eval()


@pytest.fixture
def first_atom_invalid() -> Bool[torch.Tensor, "12"]:
    """Mark atom 0 of 12 invalid and the rest valid.

    Returns:
        A boolean validity vector.
    """
    valid = torch.ones(12, dtype=torch.bool)
    valid[0] = False
    return valid


@pytest.fixture
def expected_pair_mask(
    first_atom_invalid: Bool[torch.Tensor, "12"],
) -> Bool[torch.Tensor, "12 12"]:
    """A pair is valid exactly when both of its atoms are.

    Args:
        first_atom_invalid: Per-atom validity.

    Returns:
        The outer AND of the validity vector with itself.
    """
    valid = first_atom_invalid
    return rearrange(valid, "i -> i 1") & rearrange(valid, "j -> 1 j")


@pytest.fixture
def sinusoid_position(
    sinusoid_scenario: SinusoidScenario,
) -> Float[torch.Tensor, "1 1"]:
    """Wrap the scenario's position as a ``(1, 1)`` index tensor.

    Args:
        sinusoid_scenario: Supplies the position; the requesting test
            parametrizes it.

    Returns:
        The position.
    """
    return torch.tensor([[sinusoid_scenario.value.position]])


@pytest.fixture
def expected_item_shapes(
    protein: Protein,
    tcfg: TrainConfig,
) -> ExpectedItemShapes:
    """Derive the per-atom and pair field shapes from ``protein``'s length.

    Args:
        protein: Supplies the residue count.
        tcfg: Supplies the template bin count.

    Returns:
        The expected shapes.
    """
    n_res = len(protein.aatype)
    n_atom = n_res * ATOMS_PER_RES
    return ExpectedItemShapes(
        ref_pos=(n_atom, 3),
        ref_element=(n_atom, 4),
        ref_space_uid=(n_atom,),
        gt_res_distogram_indices=(n_res, n_res),
        noised_res_distogram=(n_res, n_res, tcfg.distogram_template.n_bins),
        f_pseudo_beta_mask=(n_res,),
        tok_idx=(n_atom,),
        center_uid=(n_atom,),
    )


@pytest.fixture
def residue_of_atom(protein: Protein) -> Int[torch.Tensor, " N_atom"]:
    """Map every atom to the index of its residue.

    Args:
        protein: Supplies the residue count.

    Returns:
        ``r`` repeated ``ATOMS_PER_RES`` times for each residue ``r``.
    """
    n_res = len(protein.aatype)
    return repeat(torch.arange(n_res), "r -> (r a)", a=ATOMS_PER_RES)


@pytest.fixture
def featurize_collate(
    tcfg: TrainConfig,
    distograms: Distograms,
) -> FeaturizeCollate:
    """Build the collate function a DataLoader would hand to its workers.

    Args:
        tcfg: Configuration.
        distograms: Template, residue and atom distograms.

    Returns:
        The collate function.
    """
    return FeaturizeCollate(
        tcfg=tcfg,
        distogram_template=distograms.template,
        distogram_residue=distograms.residue,
        distogram_atom=distograms.atom,
    )


@pytest.fixture
def listed_dataset(
    dataset_jsonl: pathlib.Path,
    names: list[str],
) -> ProteinDataset:
    """Build a dataset over ``names`` from the three-entry JSONL.

    Args:
        dataset_jsonl: JSONL holding three proteins.
        names: Names to include; the requesting test parametrizes it.

    Returns:
        The dataset, padding samples to 8 residues.
    """
    return ProteinDataset(dataset_jsonl, names, max_seq_length=8)


@pytest.fixture
def long_entry_dataset(long_entry_jsonl: pathlib.Path) -> ProteinDataset:
    """Build an 8-residue dataset over an entry longer than 8 residues.

    Args:
        long_entry_jsonl: JSONL holding one long entry.

    Returns:
        The dataset.
    """
    return ProteinDataset(long_entry_jsonl, ["long.A"], max_seq_length=8)


@pytest.fixture
def metadata_mtimes(
    shard_budget: ShardBudgetParameters,
    single_entry_jsonl: pathlib.Path,
) -> MetadataMtimes:
    """Build the same shard dataset twice and record its metadata mtime.

    Args:
        shard_budget: Budget and shard directory.
        single_entry_jsonl: JSONL holding one protein.

    Returns:
        The metadata file's ``st_mtime_ns`` after each build.
    """
    first = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=single_entry_jsonl,
    )
    before = first.shard_metadata_path.stat().st_mtime_ns
    _ = ProteinShardDataset(
        budget_parameters=shard_budget,
        names=["p1"],
        dataset_jsonl=single_entry_jsonl,
    )
    after = first.shard_metadata_path.stat().st_mtime_ns
    return MetadataMtimes(before=before, after=after)


@pytest.mark.parametrize(
    "item_break",
    list(ItemBreak),
    ids=[case.name for case in ItemBreak],
)
def test_featurized_item_rejects_a_malformed_field(
    featurized_item: FeaturizedItem,
    item_break: ItemBreak,
) -> None:
    """Rebuilding a valid item with one malformed field fails the contract.

    Every other field stays valid, so the error can only come from the
    replaced one.

    Args:
        featurized_item: Valid item from ``featurize_single_item``.
        item_break: Field and reshape to apply.
    """
    field = item_break.value
    old = cast(torch.Tensor, getattr(featurized_item, field.field))
    with pytest.raises(TypeCheckError):
        _ = dataclasses.replace(
            featurized_item,
            **{field.field: field.malform(old)},
        )


@pytest.mark.parametrize(
    "batch_break",
    list(BatchBreak),
    ids=[case.name for case in BatchBreak],
)
def test_featurized_batch_rejects_a_malformed_field(
    featurized_batch: FeaturizedBatch,
    batch_break: BatchBreak,
) -> None:
    """Rebuilding a valid batch with one malformed field fails the contract.

    Args:
        featurized_batch: Valid batch from ``featurize_batch``.
        batch_break: Field and reshape to apply.
    """
    field = batch_break.value
    old = cast(torch.Tensor, getattr(featurized_batch, field.field))
    with pytest.raises(TypeCheckError):
        _ = dataclasses.replace(
            featurized_batch,
            **{field.field: field.malform(old)},
        )


@pytest.mark.parametrize(
    "bin_scenario",
    list(BinScenario),
    ids=[case.name for case in BinScenario],
)
def test_distogram_assigns_each_distance_its_bin(
    bin_scenario: BinScenario,
    bin_distogram: Distogram,
    pair_at_distance: Float[torch.Tensor, "2 3"],
    expected_bin_one_hot: Float[torch.Tensor, " C"],
) -> None:
    """Two atoms ``distance`` apart get the expected one-hot bin and mask.

    Args:
        bin_scenario: Distance, overflow setting and expected outcome.
        bin_distogram: Distogram under test.
        pair_at_distance: Two atoms the scenario's distance apart.
        expected_bin_one_hot: One-hot vector at the expected bin.
    """
    with torch.no_grad():
        onehot, mask = bin_distogram(pair_at_distance)
    assert torch.equal(onehot[0, 1], expected_bin_one_hot)
    assert bool(mask[0, 1]) == bin_scenario.value.valid


@pytest.mark.parametrize(
    "overflow_scenario",
    list(OverflowScenario),
    ids=[case.name for case in OverflowScenario],
)
def test_distogram_is_a_symmetric_one_hot_over_its_bins(
    overflow_scenario: OverflowScenario,
    overflow_distogram: Distogram,
    spread_coords: Float[torch.Tensor, "12 3"],
) -> None:
    """Every pair gets exactly one bin, and ``(i, j)`` matches ``(j, i)``.

    The overflow bin adds one class to the last axis.

    Args:
        overflow_scenario: Overflow setting and the class count it gives.
        overflow_distogram: Distogram under test.
        spread_coords: Twelve atoms at varied distances.
    """
    with torch.no_grad():
        onehot, mask = overflow_distogram(spread_coords)
    assert onehot.shape == (12, 12, overflow_scenario.value.n_classes)
    assert torch.equal(
        reduce(onehot, "i j b -> i j", "sum"),
        torch.ones(12, 12),
    )
    assert torch.equal(onehot, rearrange(onehot, "i j b -> j i b"))
    assert torch.equal(mask, rearrange(mask, "i j -> j i"))


def test_distogram_masks_every_pair_of_an_invalid_atom(
    no_overflow_distogram: Distogram,
    near_coords: Float[torch.Tensor, "12 3"],
    first_atom_invalid: Bool[torch.Tensor, "12"],
    expected_pair_mask: Bool[torch.Tensor, "12 12"],
) -> None:
    """An atom marked invalid drops its whole row and column of pairs.

    Every coordinate is within 2 angstroms of every other, so no distance
    cutoff applies and the mask alone decides. Not parametrized: which
    atom is invalid does not matter.

    Args:
        no_overflow_distogram: Distogram under test.
        near_coords: Twelve atoms within range.
        first_atom_invalid: Validity with atom 0 invalid.
        expected_pair_mask: Valid exactly where both atoms are.
    """
    with torch.no_grad():
        _, mask = no_overflow_distogram(near_coords, first_atom_invalid)
    assert torch.equal(mask, expected_pair_mask)


@pytest.mark.parametrize(
    "sinusoid_scenario",
    list(SinusoidScenario),
    ids=[case.name for case in SinusoidScenario],
)
def test_sinusoidal_encoding_matches_its_closed_form(
    sinusoid_scenario: SinusoidScenario,
    sinusoid_position: Float[torch.Tensor, "1 1"],
) -> None:
    """Position ``p`` maps to ``sin(p f)`` then ``cos(p f)`` per frequency.

    Frequencies are ``10000 ** (-k / (dim/2 - 1))``, so ``dim = 4`` gives
    ``1`` and ``1e-4``.

    Args:
        sinusoid_scenario: Position, width and expected encoding.
        sinusoid_position: The scenario's position.
    """
    case = sinusoid_scenario.value
    out = sinusoidal_encoding(sinusoid_position, dim=case.dim)
    torch.testing.assert_close(out, torch.tensor([[case.expected]]))


@pytest.mark.parametrize(
    "cb_scenario",
    list(CbScenario),
    ids=[case.name for case in CbScenario],
)
def test_ref_pos_for_residue_puts_ca_at_the_origin(
    cb_scenario: CbScenario,
) -> None:
    """CA is at the origin, N and C are off it, and a missing CB is zero.

    Glycine is the only canonical residue without a CB, so its CB row
    falls back to ``(0, 0, 0)``.

    Args:
        cb_scenario: Residue name and whether it has a CB.
    """
    case = cb_scenario.value
    pos = ref_pos_for_residue(case.resname)
    assert pos.shape == (ATOMS_PER_RES, 3)
    assert torch.equal(pos[1], torch.zeros(3))
    assert bool((pos[[0, 2]].abs().sum(-1) > 1).all())
    assert bool(pos[4].abs().sum() > 0) == case.has_cb


def test_featurize_single_item_shapes_ranges_and_index_maps(
    featurized_item: FeaturizedItem,
    expected_item_shapes: ExpectedItemShapes,
    residue_of_atom: Int[torch.Tensor, " N_atom"],
) -> None:
    """The item has every field at its shape, finite, with exact index maps.

    Atom ``l`` belongs to residue ``l // 5``, its centre atom is that
    residue's CA at ``5 r + 1``, and the noise level ``t_normalized`` lies
    in ``[0, 1]``. Not parametrized: one protein covers the layout.

    Args:
        featurized_item: Item from ``featurize_single_item``.
        expected_item_shapes: Shapes the fields must have.
        residue_of_atom: Residue index of every atom.
    """
    x = featurized_item
    fields: Mapping[str, torch.Tensor] = {
        f.name: cast(torch.Tensor, getattr(x, f.name))
        for f in dataclasses.fields(x)
    }
    expected = dataclasses.asdict(expected_item_shapes)
    assert {k: tuple(fields[k].shape) for k in expected} == expected
    assert all(torch.isfinite(t.float()).all() for t in fields.values())
    assert bool((x.t_hat > 0).all())
    assert bool(((x.t_normalized >= 0) & (x.t_normalized <= 1)).all())
    assert torch.equal(x.ref_space_uid, residue_of_atom)
    assert torch.equal(x.tok_idx, residue_of_atom)
    assert torch.equal(x.center_uid, ATOMS_PER_RES * residue_of_atom + 1)


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
    featurize_collate: FeaturizeCollate,
    protein: Protein,
    featurized_batch: FeaturizedBatch,
) -> None:
    """A collate restored from pickle, as DataLoader workers get it, works.

    It must produce a FeaturizedBatch shaped like ``featurize_batch``'s.
    Not parametrized: pickling is the one transport DataLoader uses.

    Args:
        featurize_collate: Collate function to send.
        protein: Protein to collate twice.
        featurized_batch: Reference batch of the same two proteins.
    """
    restored = queue_round_trip(featurize_collate)
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
    listed_dataset: ProteinDataset,
    expected: int,
) -> None:
    """The dataset's length is the number of listed names, and each loads.

    Args:
        listed_dataset: Dataset over the case's names.
        expected: Number of samples it must hold.
    """
    assert len(listed_dataset) == expected
    assert all(isinstance(listed_dataset[i], Protein) for i in range(expected))


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
    long_entry_dataset: ProteinDataset,
) -> None:
    """A 100-residue entry comes back cut to ``max_seq_length = 8``.

    Not parametrized: padding is the test above.

    Args:
        long_entry_dataset: 8-residue dataset over a longer entry.
    """
    sample = long_entry_dataset[0]
    assert sample.aatype.shape == (8,)
    assert sample.atom_positions.shape == (8, 37, 3)


@pytest.mark.parametrize(
    "pickle_state",
    list(PickleState),
    ids=[case.name for case in PickleState],
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
    shard_dataset: ProteinShardDataset,
) -> None:
    """The metadata sidecar records the token budget and shard size used.

    Not parametrized: the fields are copied from the budget as is.

    Args:
        shard_dataset: Two-entry shard dataset.
    """
    dataset = shard_dataset
    meta = ShardMetadata.model_validate_json(
        dataset.shard_metadata_path.read_text(),
    )
    assert meta.token_budget == dataset.token_budget
    assert meta.shard_size == dataset.n_proteins_in_shard
    assert meta.n_shards >= 1


@pytest.mark.parametrize(
    "names_hash_scenario",
    list(NamesHashScenario),
    ids=[case.name for case in NamesHashScenario],
)
def test_shard_metadata_names_hash_identifies_the_name_set(
    names_hash_scenario: NamesHashScenario,
    names_hashes: NamesHashes,
) -> None:
    """Two builds share a names hash exactly when they shard the same names.

    The hash is over the sorted names, so order does not matter.

    Args:
        names_hash_scenario: The two builds' names and whether they match.
        names_hashes: Both builds' hashes.
    """
    same = names_hashes.first == names_hashes.second
    assert same == names_hash_scenario.value.same_hash


def test_shard_dataset_reuses_existing_metadata(
    metadata_mtimes: MetadataMtimes,
) -> None:
    """A second dataset in the same shard directory leaves the shards alone.

    The metadata file is not rewritten, so its mtime is unchanged. Not
    parametrized: the cache check reads one file.

    Args:
        metadata_mtimes: Metadata mtimes around the second build.
    """
    assert metadata_mtimes.after == metadata_mtimes.before


@pytest.mark.parametrize(
    "json_form",
    list(JsonForm),
    ids=[case.name for case in JsonForm],
)
def test_parse_protein_centres_ca_and_maps_unknown_residues_to_x(
    shard_dataset: ProteinShardDataset,
    unknown_residue_json: object,
) -> None:
    """Parsing gives a CA centroid at the origin and ``X`` for unknown codes.

    The JSON may arrive as bytes or as a decoded dict, as webdataset hands
    over either; both must parse the same way.

    Args:
        shard_dataset: Shard dataset to parse with.
        unknown_residue_json: Entry ``p1`` with five unknown codes, as a
            dict or as bytes.
    """
    parsed = shard_dataset.parse_protein({"json": unknown_residue_json})
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
    bucketed_train_args: TrainArgs,
) -> None:
    """The bucketed training loader yields FeaturizedBatch, not raw proteins.

    Not parametrized: one batch shows the collate is wired in.

    Args:
        bucketed_cfg: Loader configuration.
        bucketed_train_args: Arguments over a five-protein dataset.
    """
    train_loader, _, _ = make_bucketed_data_loaders(
        cfg=bucketed_cfg,
        extra_train_args=bucketed_train_args,
    )
    assert isinstance(next(iter(train_loader)), FeaturizedBatch)


@pytest.mark.parametrize(
    "debug_cache_scenario",
    list(DebugCacheScenario),
    ids=[case.name for case in DebugCacheScenario],
)
def test_bucketed_debug_run_caps_train_split_at_252(
    debug_train_size: int,
    debug_cache_scenario: DebugCacheScenario,
) -> None:
    """``debug_run`` trains on the first 252 proteins of a 300-protein split.

    With a shard cache already built for all 300 proteins in the same
    directory, the debug run must still cap at 252 rather than reuse the
    full cache.

    Args:
        debug_train_size: Train-set size of the debug-run loader.
        debug_cache_scenario: The cap ``make_bucketed_data_loaders``
            applies.
    """
    assert debug_train_size == debug_cache_scenario.value.expected


@pytest.mark.parametrize(
    "manifest_scenario",
    list(ManifestScenario),
    ids=[case.name for case in ManifestScenario],
)
def test_dataset_splits_manifest_parses_splits_and_cath_nodes(
    manifest_scenario: ManifestScenario,
    manifest: DatasetSplitsManifest,
) -> None:
    """The manifest reads the three splits, ``cath_nodes``, and ignores extras.

    Args:
        manifest_scenario: Fields added to the three splits.
        manifest: The parsed manifest.
    """
    assert (manifest.train, manifest.validation, manifest.test) == (
        ["p1"],
        ["p2"],
        ["p3"],
    )
    assert manifest.cath_nodes == manifest_scenario.value.cath_nodes


@pytest.fixture
def expected_t_normalized(
    featurized_item: FeaturizedItem,
    tcfg: TrainConfig,
) -> Float[torch.Tensor, "N_res N_res"]:
    """Build the template weight the featurizer must give ``featurized_item``.

    ``t_hat`` is per residue, but the featurizer draws one noise level per
    item, so the first residue's level sets the expected weight.

    Args:
        featurized_item: Item whose ``t_hat`` sets the noise level.
        tcfg: Supplies the noise schedule.

    Returns:
        ``normalized_time_from_sigma(t_hat)`` at every residue pair.
    """
    expected = normalized_time_from_sigma(featurized_item.t_hat[0], tcfg.noise)
    return torch.full_like(featurized_item.t_normalized, expected.item())


def test_featurized_template_weight_is_normalized_noise_level(
    featurized_item: FeaturizedItem,
    expected_t_normalized: Float[torch.Tensor, "N_res N_res"],
) -> None:
    """``t_normalized`` is the normalized noise level of ``t_hat`` everywhere.

    The featurizer used to draw the template weight from ``U(0, 1)``
    independently of ``t_hat``, which made it pure noise to the template
    embedder. It must now equal the sampler's definition at every residue
    pair. Not parametrized: the item's noise level is the only input.

    Args:
        featurized_item: Item under test.
        expected_t_normalized: The sampler's weight for its ``t_hat``.
    """
    assert torch.equal(featurized_item.t_normalized, expected_t_normalized)


@pytest.fixture
def noise_schedule() -> NoiseScheduleParams:
    """Provide the production noise schedule parameters.

    Returns:
        The default ``NoiseScheduleParams``.
    """
    return NoiseScheduleParams()


@dataclasses.dataclass(frozen=True)
class SigmaBoundCase:
    """A noise level relative to ``sigma_data`` and its normalized time.

    Attributes:
        sigma_over_data: Noise level in units of ``sigma_data``.
        expected: Normalized time the noise level must map to.
    """

    sigma_over_data: float
    expected: float


class SigmaBoundScenario(enum.Enum):
    """Noise levels at and past the schedule's bounds.

    The production schedule spans ``sigma_min = 4e-4`` to
    ``sigma_max = 160`` in units of ``sigma_data``.

    Members:
        above_sigma_max_clamps_to_1: Far above ``sigma_max``.
        below_sigma_min_clamps_to_0: Far below ``sigma_min``.
        at_sigma_max: Exactly ``sigma_max``.
        at_sigma_min: Exactly ``sigma_min``.
    """

    above_sigma_max_clamps_to_1 = SigmaBoundCase(1e3, 1.0)
    below_sigma_min_clamps_to_0 = SigmaBoundCase(1e-6, 0.0)
    at_sigma_max = SigmaBoundCase(160.0, 1.0)
    at_sigma_min = SigmaBoundCase(4e-4, 0.0)


@pytest.fixture
def bound_sigma(
    sigma_bound_scenario: SigmaBoundScenario,
    noise_schedule: NoiseScheduleParams,
) -> Float[torch.Tensor, ""]:
    """Build the scenario's noise level in absolute units.

    Args:
        sigma_bound_scenario: Noise level relative to ``sigma_data``; the
            requesting test parametrizes it.
        noise_schedule: Supplies ``sigma_data``.

    Returns:
        A scalar noise level.
    """
    return torch.tensor(
        sigma_bound_scenario.value.sigma_over_data * noise_schedule.sigma_data,
    )


@pytest.mark.parametrize(
    "sigma_bound_scenario",
    list(SigmaBoundScenario),
    ids=[case.name for case in SigmaBoundScenario],
)
def test_normalized_time_from_sigma_bounds(
    noise_schedule: NoiseScheduleParams,
    bound_sigma: Float[torch.Tensor, ""],
    sigma_bound_scenario: SigmaBoundScenario,
) -> None:
    """Sigmas at or past the schedule's bounds map to exactly 0 or 1.

    The lognormal training distribution draws sigmas outside
    ``[sigma_min, sigma_max]``, so the inverse clamps rather than returning
    times outside [0, 1].

    Args:
        noise_schedule: Schedule whose bounds are tested.
        bound_sigma: Noise level at or past a bound.
        sigma_bound_scenario: Expected normalized time.
    """
    t = normalized_time_from_sigma(bound_sigma, noise_schedule)
    torch.testing.assert_close(
        t,
        torch.tensor(sigma_bound_scenario.value.expected),
        atol=1e-6,
        rtol=0,
    )


@dataclasses.dataclass(frozen=True)
class NullTemplateCase:
    """Shape requested from ``null_template_distogram`` in one scenario.

    Attributes:
        shape: Requested (B, N_res, N_res, n_templ_bins) shape, which the
            returned tensor must match exactly.
    """

    shape: torch.Size


class NullTemplateScenario(enum.Enum):
    """Null template shapes, varying batch, residue and bin counts.

    Members:
        single_sample: One 4-residue sample with the production 39 bins.
        batched_19_bins: Three 7-residue samples with 19 bins.
    """

    single_sample = NullTemplateCase(torch.Size((1, 4, 4, 39)))
    batched_19_bins = NullTemplateCase(torch.Size((3, 7, 7, 19)))


@pytest.fixture
def null_template(
    null_template_scenario: NullTemplateScenario,
) -> Float[torch.Tensor, "B N_res N_res n_templ_bins"]:
    """Build the scenario's null template distogram on the CPU.

    Args:
        null_template_scenario: Scenario holding the shape to request,
            parametrized directly by the requesting test.

    Returns:
        The output of ``null_template_distogram`` for that shape.
    """
    return null_template_distogram(null_template_scenario.value.shape, "cpu")


@pytest.mark.parametrize(
    "null_template_scenario",
    list(NullTemplateScenario),
    ids=[case.name for case in NullTemplateScenario],
)
def test_null_template_distogram_is_all_zero(
    null_template: Float[torch.Tensor, "B N_res N_res n_templ_bins"],
    null_template_scenario: NullTemplateScenario,
) -> None:
    """The null template has the requested shape, float dtype, and no mass.

    A real template is one-hot on the bin axis; the null must sum to zero
    there so the trunk can tell "no template" apart from any structure.

    Args:
        null_template: The null template built for the scenario.
        null_template_scenario: Scenario holding the requested shape.
    """
    assert null_template.shape == null_template_scenario.value.shape
    assert null_template.dtype == torch.float32
    assert not null_template.any()

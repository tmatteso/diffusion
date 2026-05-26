"""ClusterIndex: partitions a protein JSONL into 64 length-based cluster files."""

import json
from collections.abc import Mapping
from pathlib import Path


class ClusterIndex:
    """Partitions proteins from a source JSONL into per-length-cluster JSONL files.

    On construction, checks whether cluster files already exist. If not, scans the
    source JSONL, assigns each included protein to one of 64 regular clusters
    (width = token_budget // n_clusters residues each) or an overflow cluster, writes
    per-cluster JSONL files, and builds byte-offset arrays. If the files exist, reads
    them to rebuild the offset arrays without re-parsing protein data.

    The cluster directory is derived from the source path:
        <source_stem>_clusters/ sibling to <source>.jsonl

    Args:
        jsonl_path:   Path to the source JSONL protein dataset.
        names:        Names (entry["name"]) of proteins to include.
        token_budget: Maximum residues per batch; defines the upper edge of the last
                      regular cluster. Default 512.
        n_clusters:   Number of regular length clusters. Default 64.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        names: list[str],
        token_budget: int = 512,
        n_clusters: int = 64,
    ) -> None:
        self._jsonl_path = Path(jsonl_path)
        self._names = names
        self._token_budget = token_budget
        self._n_clusters = n_clusters
        self._cluster_dir = self._jsonl_path.parent / (self._jsonl_path.stem + "_clusters")

        # Representative lengths: regular clusters have rep_len = bin_width * (k+1).
        # Overflow cluster (index n_clusters) has rep_len = token_budget + 1 so that
        # the greedy-packing overflow check `rep_len > token_budget` fires correctly.
        bin_width = token_budget // n_clusters
        self.cluster_rep_len: list[int] = [bin_width * (k + 1) for k in range(n_clusters)] + [
            token_budget + 1
        ]

        self.flat_to_cluster: list[int] = []
        self.flat_to_local: list[int] = []
        self.cluster_offsets: list[list[int]] = []

        if self._cache_exists():
            self._load_from_cache()
        else:
            self._build_and_cache()

    # ------------------------------------------------------------------
    # Public helpers

    def assign_cluster(self, seq_len: int) -> int:
        """Return cluster id (0..n_clusters-1 regular, n_clusters overflow) for seq_len.

        Args:
            seq_len: Number of residues in the protein.

        Returns:
            Cluster id in [0, n_clusters].
        """
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if seq_len > self._token_budget:
            return self._n_clusters
        bin_width = self._token_budget // self._n_clusters
        return min((seq_len - 1) // bin_width, self._n_clusters - 1)

    def cluster_file(self, k: int) -> Path:
        """Return the path to cluster k's JSONL file.

        Args:
            k: Cluster id (0..n_clusters).

        Returns:
            Path to cluster_k.jsonl inside the cluster directory.
        """
        return self._cluster_dir / f"cluster_{k:03d}.jsonl"

    def __len__(self) -> int:
        """Return the total number of included proteins across all clusters."""
        return len(self.flat_to_cluster)

    # ------------------------------------------------------------------
    # Cache management

    def _manifest_path(self) -> Path:
        """Return the path to the names-manifest file inside the cluster directory."""
        return self._cluster_dir / "names.manifest"

    def _cache_exists(self) -> bool:
        """Return True if all cluster JSONL files exist and were built with the current names."""
        if not all(self.cluster_file(k).exists() for k in range(self._n_clusters + 1)):
            return False
        if not self._manifest_path().exists():
            return False
        return json.loads(self._manifest_path().read_text()) == sorted(self._names)

    def _build_and_cache(self) -> None:
        """Partition source JSONL by length, write cluster files, build offset arrays."""
        self._cluster_dir.mkdir(parents=True, exist_ok=True)
        name_set = set(self._names)

        # Collect raw lines per cluster without loading full protein data.
        cluster_lines: list[list[bytes]] = [[] for _ in range(self._n_clusters + 1)]
        with open(self._jsonl_path, "rb") as f:
            for raw_line in f:
                entry: Mapping[str, object] = json.loads(raw_line)
                if entry["name"] not in name_set:
                    continue
                seq_len = len(entry["seq"])  # type: ignore[arg-type]
                k = self.assign_cluster(seq_len)
                cluster_lines[k].append(raw_line)

        # Write cluster files and record byte offsets.
        self.cluster_offsets = []
        for k in range(self._n_clusters + 1):
            offsets: list[int] = []
            with open(self.cluster_file(k), "wb") as f:
                pos = 0
                for raw_line in cluster_lines[k]:
                    offsets.append(pos)
                    f.write(raw_line)
                    pos += len(raw_line)
            self.cluster_offsets.append(offsets)

        self._manifest_path().write_text(json.dumps(sorted(self._names)))
        self._build_flat_index()

    def _load_from_cache(self) -> None:
        """Read cluster files to rebuild byte-offset arrays. Does not parse protein data."""
        self.cluster_offsets = []
        for k in range(self._n_clusters + 1):
            offsets: list[int] = []
            byte_pos = 0
            with open(self.cluster_file(k), "rb") as f:
                for raw_line in f:
                    offsets.append(byte_pos)
                    byte_pos += len(raw_line)
            self.cluster_offsets.append(offsets)

        self._build_flat_index()

    def _build_flat_index(self) -> None:
        """Populate flat_to_cluster and flat_to_local from cluster_offsets."""
        self.flat_to_cluster = []
        self.flat_to_local = []
        for k, offsets in enumerate(self.cluster_offsets):
            for local_idx in range(len(offsets)):
                self.flat_to_cluster.append(k)
                self.flat_to_local.append(local_idx)

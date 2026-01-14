import os
import json
from pathlib import Path
from typing import Optional, Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Standard atom order for residues (37 atoms total for all-atom representation)
# Based on common protein structure representations
ATOM_TYPES = [
    'N', 'CA', 'C', 'O',  # Backbone atoms (all residues)
    'CB',  # Beta carbon (all except Glycine)
    # Side chain atoms (residue-specific, padded with zeros if missing)
    'CG', 'CG1', 'CG2', 'OG', 'OG1', 'OG2', 'SG',
    'CD', 'CD1', 'CD2', 'OD1', 'OD2', 'SD',
    'CE', 'CE1', 'CE2', 'CE3', 'OE1', 'OE2', 'NE', 'NE1', 'NE2',
    'CZ', 'CZ2', 'CZ3', 'NZ', 'OH',
    'ND1', 'ND2',
    'NH1', 'NH2',
    'OXT',  # C-terminal oxygen
]

assert len(ATOM_TYPES) == 37, f"Expected 37 atom types, got {len(ATOM_TYPES)}"


def parse_pdb_file(pdb_path: str, atom_types: list[str] = ATOM_TYPES) -> torch.Tensor:
    """
    Parse a PDB file and extract atom coordinates.

    Args:
        pdb_path: Path to PDB file
        atom_types: List of atom types to extract (in order)

    Returns:
        coords: Tensor of shape [n_residues, n_atoms, 3]
                Missing atoms are filled with zeros
    """
    atom_type_to_idx = {atom: i for i, atom in enumerate(atom_types)}
    n_atoms = len(atom_types)

    # Read PDB file and extract ATOM lines
    residues = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                res_num = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())

                if atom_name in atom_type_to_idx:
                    if res_num not in residues:
                        residues[res_num] = np.zeros((n_atoms, 3), dtype=np.float32)

                    atom_idx = atom_type_to_idx[atom_name]
                    residues[res_num][atom_idx] = [x, y, z]

    # Sort residues by residue number and stack
    if not residues:
        raise ValueError(f"No valid atoms found in {pdb_path}")

    sorted_res_nums = sorted(residues.keys())
    coords_list = [residues[res_num] for res_num in sorted_res_nums]
    coords = np.stack(coords_list, axis=0)  # [n_residues, n_atoms, 3]

    return torch.from_numpy(coords)


class ProteinStructureDataset(Dataset):
    """
    Dataset for loading protein structures from PDB files.

    Returns atom coordinates in the format [n_residues, n_atoms, 3].

    For large datasets (>10k files), uses lazy filtering to avoid slow initialization.
    Optionally supports metadata caching to speed up filtering.
    """

    def __init__(
        self,
        pdb_dir: Union[str, Path],
        atom_types: list[str] = ATOM_TYPES,
        transform: Optional[Callable] = None,
        file_pattern: str = "*.pdb",
        max_residues: Optional[int] = 256,
        lazy_filter: bool = True,
        cache_metadata: bool = True,
        metadata_cache_file: Optional[str] = None,
    ):
        """
        Args:
            pdb_dir: Directory containing PDB files
            atom_types: List of atom types to extract
            transform: Optional transform to apply to coordinates
            file_pattern: Glob pattern to match PDB files
            max_residues: Maximum number of residues allowed. Files with more residues are skipped.
                         Set to None to disable filtering.
            lazy_filter: If True, filter during __getitem__ instead of __init__ (much faster for large datasets).
                        If False, filter during __init__ (slower but provides accurate dataset size).
            cache_metadata: If True, cache residue counts to disk for faster subsequent initialization.
            metadata_cache_file: Path to metadata cache file. If None, uses "{pdb_dir}/.metadata_cache.json"
        """
        self.pdb_dir = Path(pdb_dir)
        self.atom_types = atom_types
        self.transform = transform
        self.max_residues = max_residues
        self.lazy_filter = lazy_filter
        self.cache_metadata = cache_metadata

        # Set up metadata cache
        if metadata_cache_file is None:
            self.metadata_cache_file = self.pdb_dir / ".metadata_cache.json"
        else:
            self.metadata_cache_file = Path(metadata_cache_file)

        # Load metadata cache if available
        self.metadata = self._load_metadata_cache() if cache_metadata else {}

        # Find all PDB files
        all_pdb_files = sorted(list(self.pdb_dir.glob(file_pattern)))

        if len(all_pdb_files) == 0:
            raise ValueError(f"No PDB files found in {pdb_dir} matching {file_pattern}")

        # Fast initialization: lazy filtering
        if lazy_filter:
            self.pdb_files = all_pdb_files
            print(f"Found {len(self.pdb_files)} PDB files in {pdb_dir}")
            if max_residues is not None:
                print(f"Using lazy filtering (max_residues={max_residues})")
                print(f"Files exceeding {max_residues} residues will be skipped during iteration")
        else:
            # Slow but accurate: filter during initialization
            self.pdb_files = []
            excluded_count = 0

            for pdb_file in all_pdb_files:
                try:
                    n_residues = self._get_residue_count(pdb_file)
                    if max_residues is None or n_residues <= max_residues:
                        self.pdb_files.append(pdb_file)
                    else:
                        excluded_count += 1
                except Exception as e:
                    print(f"Warning: Could not parse {pdb_file.name}: {e}")
                    excluded_count += 1

            print(f"Found {len(all_pdb_files)} PDB files in {pdb_dir}")
            if max_residues is not None:
                print(f"Excluded {excluded_count} files with >{max_residues} residues or parsing errors")
            print(f"Using {len(self.pdb_files)} PDB files")

            if len(self.pdb_files) == 0:
                raise ValueError(f"No valid PDB files remaining after filtering")

            # Save updated metadata cache
            if self.cache_metadata:
                self._save_metadata_cache()

    def _load_metadata_cache(self) -> dict:
        """Load metadata cache from disk."""
        if self.metadata_cache_file.exists():
            try:
                with open(self.metadata_cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load metadata cache: {e}")
        return {}

    def _save_metadata_cache(self):
        """Save metadata cache to disk."""
        try:
            with open(self.metadata_cache_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            print(f"Warning: Could not save metadata cache: {e}")

    def _get_residue_count(self, pdb_path: Path) -> int:
        """
        Get residue count from cache or by counting.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Number of residues
        """
        filename = pdb_path.name

        # Check cache first
        if filename in self.metadata:
            return self.metadata[filename]['n_residues']

        # Count residues
        n_residues = self._count_residues(pdb_path)

        # Update cache
        if self.cache_metadata:
            self.metadata[filename] = {'n_residues': n_residues}

        return n_residues

    def _count_residues(self, pdb_path: Path) -> int:
        """
        Count the number of residues in a PDB file without parsing all atoms.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Number of residues
        """
        residue_numbers = set()
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    res_num = int(line[22:26].strip())
                    residue_numbers.add(res_num)
        return len(residue_numbers)

    def __len__(self) -> int:
        return len(self.pdb_files)

    def __getitem__(self, idx: int) -> Optional[dict]:
        """
        Returns:
            Dictionary containing:
                - coords: Tensor [n_residues, n_atoms, 3]
                - filename: Name of the PDB file
                - n_residues: Number of residues
            Or None if the file should be skipped (lazy filtering).
        """
        pdb_path = self.pdb_files[idx]

        # Lazy filtering: check size before parsing if max_residues is set
        if self.lazy_filter and self.max_residues is not None:
            try:
                n_residues = self._get_residue_count(pdb_path)
                if n_residues > self.max_residues:
                    return None  # Skip this file
            except Exception:
                return None  # Skip files that can't be parsed

        try:
            coords = parse_pdb_file(str(pdb_path), self.atom_types)
        except Exception as e:
            if self.lazy_filter:
                return None  # Skip files with parsing errors
            else:
                raise RuntimeError(f"Error parsing {pdb_path}: {e}")

        # Check size after parsing (double check)
        if self.max_residues is not None and coords.shape[0] > self.max_residues:
            if self.lazy_filter:
                return None
            else:
                raise ValueError(f"File {pdb_path.name} has {coords.shape[0]} residues (max: {self.max_residues})")

        if self.transform is not None:
            coords = self.transform(coords)

        return {
            'coords': coords,
            'filename': pdb_path.name,
            'n_residues': coords.shape[0],
        }


def packed_collate_fn(batch: list[Optional[dict]]) -> Optional[dict]:
    """
    Collate function using sequence packing (Krell et al. 2022).

    No padding - returns list of variable-length sequences for efficient processing.
    Filters out None values (skipped samples from lazy filtering).

    Args:
        batch: List of dictionaries from ProteinStructureDataset (may contain None)

    Returns:
        Dictionary with:
            - coords: List of tensors, each [n_residues_i, n_atoms, 3]
            - lengths: [batch_size] (sequence lengths)
            - filenames: List of filenames
        Or None if all samples were filtered out.
    """
    # Filter out None values (skipped samples)
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None  # All samples were filtered out

    coords_list = [item['coords'] for item in batch]
    filenames = [item['filename'] for item in batch]
    lengths = torch.tensor([item['n_residues'] for item in batch])

    return {
        'coords': coords_list,
        'lengths': lengths,
        'filenames': filenames,
    }


def create_dataloader(
    pdb_dir: Union[str, Path],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    file_pattern: str = "*.pdb",
    max_residues: Optional[int] = 256,
    lazy_filter: bool = True,
    cache_metadata: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for protein structures with sequence packing.

    Args:
        pdb_dir: Directory containing PDB files
        batch_size: Batch size
        shuffle: Whether to shuffle the dataset
        num_workers: Number of worker processes for data loading
        transform: Optional transform to apply to coordinates
        file_pattern: Glob pattern to match PDB files
        max_residues: Maximum number of residues allowed. Files with more residues are skipped.
                     Set to None to disable filtering.
        lazy_filter: If True, filter during iteration (fast init). If False, filter during init (slow).
        cache_metadata: If True, cache residue counts to disk for faster subsequent runs.

    Returns:
        DataLoader instance
    """
    dataset = ProteinStructureDataset(
        pdb_dir=pdb_dir,
        transform=transform,
        file_pattern=file_pattern,
        max_residues=max_residues,
        lazy_filter=lazy_filter,
        cache_metadata=cache_metadata,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=packed_collate_fn,
    )

    return dataloader


# Common transforms
class CenterProtein:
    """Center protein coordinates at origin."""

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [n_residues, n_atoms, 3]
        Returns:
            Centered coordinates
        """
        # Use CA atoms (index 1) for centering
        ca_coords = coords[:, 1, :]  # [n_residues, 3]
        center = ca_coords.mean(dim=0, keepdim=True)  # [1, 3]
        coords = coords - center.unsqueeze(1)  # Broadcast to [n_residues, n_atoms, 3]
        return coords


class RandomRotation3D:
    """
    Apply random 3D rotation to protein coordinates.

    Generates a random rotation matrix using uniformly distributed rotations.
    """

    def __init__(self, deterministic: bool = False):
        """
        Args:
            deterministic: If True, use a fixed seed for reproducibility
        """
        self.deterministic = deterministic

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [n_residues, n_atoms, 3]
        Returns:
            Rotated coordinates
        """
        # Generate random rotation matrix using QR decomposition of random matrix
        # This ensures uniform distribution over SO(3)
        random_matrix = torch.randn(3, 3, dtype=coords.dtype, device=coords.device)
        q, r = torch.linalg.qr(random_matrix)

        # Ensure proper rotation (det = 1, not reflection with det = -1)
        d = torch.diag(r)
        rotation_matrix = q * d.sign().unsqueeze(0)

        # Apply rotation: coords @ R^T
        original_shape = coords.shape
        coords_flat = coords.reshape(-1, 3)  # [n_residues * n_atoms, 3]
        coords_rotated = coords_flat @ rotation_matrix.T
        coords_rotated = coords_rotated.reshape(original_shape)

        return coords_rotated


class RandomTranslation3D:
    """
    Apply random translation to protein coordinates.

    Translations are sampled uniformly from a specified range.
    """

    def __init__(self, translation_range: float = 5.0):
        """
        Args:
            translation_range: Maximum translation distance in Angstroms
                              Translation is sampled uniformly from [-range, +range]
        """
        self.translation_range = translation_range

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [n_residues, n_atoms, 3]
        Returns:
            Translated coordinates
        """
        # Sample random translation vector
        translation = torch.rand(3, dtype=coords.dtype, device=coords.device) * 2 - 1
        translation = translation * self.translation_range  # Scale to range

        # Apply translation
        coords_translated = coords + translation.unsqueeze(0).unsqueeze(0)

        return coords_translated


class RandomRotationTranslation3D:
    """
    Apply both random rotation and translation to protein coordinates.

    Combines RandomRotation3D and RandomTranslation3D for efficient augmentation.
    """

    def __init__(self, translation_range: float = 5.0):
        """
        Args:
            translation_range: Maximum translation distance in Angstroms
        """
        self.translation_range = translation_range

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [n_residues, n_atoms, 3]
        Returns:
            Rotated and translated coordinates
        """
        # Random rotation
        random_matrix = torch.randn(3, 3, dtype=coords.dtype, device=coords.device)
        q, r = torch.linalg.qr(random_matrix)
        d = torch.diag(r)
        rotation_matrix = q * d.sign().unsqueeze(0)

        # Apply rotation
        original_shape = coords.shape
        coords_flat = coords.reshape(-1, 3)
        coords_rotated = coords_flat @ rotation_matrix.T
        coords_rotated = coords_rotated.reshape(original_shape)

        # Random translation
        translation = torch.rand(3, dtype=coords.dtype, device=coords.device) * 2 - 1
        translation = translation * self.translation_range
        coords_final = coords_rotated + translation.unsqueeze(0).unsqueeze(0)

        return coords_final

# what are we doing now?
# I want to make sure we are computing the loss correctly.

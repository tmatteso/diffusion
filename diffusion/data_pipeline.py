import os
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
    """

    def __init__(
        self,
        pdb_dir: Union[str, Path],
        atom_types: list[str] = ATOM_TYPES,
        transform: Optional[Callable] = None,
        file_pattern: str = "*.pdb",
    ):
        """
        Args:
            pdb_dir: Directory containing PDB files
            atom_types: List of atom types to extract
            transform: Optional transform to apply to coordinates
            file_pattern: Glob pattern to match PDB files
        """
        self.pdb_dir = Path(pdb_dir)
        self.atom_types = atom_types
        self.transform = transform

        # Find all PDB files
        self.pdb_files = sorted(list(self.pdb_dir.glob(file_pattern)))

        if len(self.pdb_files) == 0:
            raise ValueError(f"No PDB files found in {pdb_dir} matching {file_pattern}")

        print(f"Found {len(self.pdb_files)} PDB files in {pdb_dir}")

    def __len__(self) -> int:
        return len(self.pdb_files)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            Dictionary containing:
                - coords: Tensor [n_residues, n_atoms, 3]
                - filename: Name of the PDB file
                - n_residues: Number of residues
        """
        pdb_path = self.pdb_files[idx]

        try:
            coords = parse_pdb_file(str(pdb_path), self.atom_types)
        except Exception as e:
            raise RuntimeError(f"Error parsing {pdb_path}: {e}")

        if self.transform is not None:
            coords = self.transform(coords)

        return {
            'coords': coords,
            'filename': pdb_path.name,
            'n_residues': coords.shape[0],
        }


def packed_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function using sequence packing (Krell et al. 2022).

    No padding - returns list of variable-length sequences for efficient processing.

    Args:
        batch: List of dictionaries from ProteinStructureDataset

    Returns:
        Dictionary with:
            - coords: List of tensors, each [n_residues_i, n_atoms, 3]
            - lengths: [batch_size] (sequence lengths)
            - filenames: List of filenames
    """
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

    Returns:
        DataLoader instance
    """
    dataset = ProteinStructureDataset(
        pdb_dir=pdb_dir,
        transform=transform,
        file_pattern=file_pattern,
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

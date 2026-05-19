# DNA / RNA molecule-type support in `atom_utils.py`

**Date:** 2026-05-19
**Scope:** `pallatom/helpers/atom_utils.py` and `pallatom/tests/helpers/test_atom_utils.py`

---

## Goal

Extend the `Protein` dataclass and `protein_from_pdb()` so that the parsed structure carries
explicit per-residue molecule-type information (protein / DNA / RNA). Also add the canonical
DNA and RNA monomer lookup tables that mirror OpenFold's `restype_3to1` / `restype_order`
convention.

---

## Decisions

| Question | Decision |
|----------|----------|
| Where to store molecule type? | Repurpose `Protein.b_factors` (shape unchanged: `num_res × num_atom_type`). Each row broadcasts a uniform float: `0.0`=protein, `1.0`=DNA, `2.0`=RNA. |
| Are real B-factors kept? | No. The model generates structures; real B-factors from PDB are not used downstream. |
| HETATM records? | `protein_from_pdb()` *explicitly* detects and skips them (water, ions, ligands). Only `ATOM` records produce residues in the output `Protein`. |
| D-prefix-less DNA? | `A`/`C`/`G` residues are disambiguated by checking for the `O2'` atom (present in RNA ribose, absent in deoxyribose). Bare `T` is forced to DNA; bare `U` is forced to RNA. |
| One-letter codes for DNA/RNA? | Lowercase (`a/c/g/t` for DNA, `a/c/g/u` for RNA) to avoid collision with protein single-letter codes. |

---

## New global constants

All added to `atom_utils.py`, after the existing `restype_3to1` block.

```python
from types import MappingProxyType
from typing import Final

# Molecule type encoding — stored in b_factors (broadcast per residue to all atom slots)
MOL_TYPE_PROTEIN: Final[int] = 0
MOL_TYPE_DNA:     Final[int] = 1
MOL_TYPE_RNA:     Final[int] = 2

# DNA monomers (canonical PDB ATOM residue names)
DNA_RESTYPES: Final[list[str]] = ["DA", "DC", "DG", "DT"]
DNA_RESTYPE_ORDER: Final[MappingProxyType[str, int]] = MappingProxyType(
    {restype: i for i, restype in enumerate(DNA_RESTYPES)}
)
DNA_RESTYPE_3TO1: Final[MappingProxyType[str, str]] = MappingProxyType(
    {"DA": "a", "DC": "c", "DG": "g", "DT": "t"}
)

# RNA monomers (canonical PDB ATOM residue names)
RNA_RESTYPES: Final[list[str]] = ["A", "C", "G", "U"]
RNA_RESTYPE_ORDER: Final[MappingProxyType[str, int]] = MappingProxyType(
    {restype: i for i, restype in enumerate(RNA_RESTYPES)}
)
RNA_RESTYPE_3TO1: Final[MappingProxyType[str, str]] = MappingProxyType(
    {"A": "a", "C": "c", "G": "g", "U": "u"}
)
```

`MappingProxyType` enforces immutability at runtime (`TypeError` on `__setitem__`).
`Final` enforces it at the pyright level.

---

## New private helper: `_classify_mol_type`

A module-level private function, placed just before `protein_from_pdb`:

```python
def _classify_mol_type(resname: str, atom_names: frozenset[str]) -> int:
    """Return MOL_TYPE_* for a PDB residue given its name and observed atoms.

    Args:
        resname: Stripped residue name from the PDB ATOM record (e.g. "ALA", "DA", "A").
        atom_names: Set of atom names present in this residue (used to detect 2'-OH).

    Returns:
        MOL_TYPE_PROTEIN, MOL_TYPE_DNA, or MOL_TYPE_RNA.
    """
    if resname in restype_3to1:
        return MOL_TYPE_PROTEIN
    if resname in DNA_RESTYPE_3TO1 or resname == "T":
        return MOL_TYPE_DNA
    if resname == "U":
        return MOL_TYPE_RNA
    if resname in RNA_RESTYPE_3TO1:          # A / C / G — ambiguous
        return MOL_TYPE_RNA if "O2'" in atom_names else MOL_TYPE_DNA
    return MOL_TYPE_PROTEIN                  # unknown → default protein
```

Classification order:
1. Protein (exact 3-letter code match in `restype_3to1`)
2. Unambiguous DNA: canonical D-prefix names (`DA`/`DC`/`DG`/`DT`) or bare `T`
3. Unambiguous RNA: bare `U`
4. Ambiguous (`A`/`C`/`G`): RNA if `O2'` atom present (ribose 2'-hydroxyl), else DNA
5. Unknown: default to protein

---

## Changes to `protein_from_pdb()`

### 1. Explicit HETATM skip

```python
for _line in _fh:
    _rec = _line[:6].strip()
    if _rec == "HETATM":
        continue   # skip: water, ions, ligands, crystallographic reagents
    if _rec != "ATOM":
        continue
    # ... existing parsing unchanged
```

### 2. Set `b_factors` from molecule type

After all residue atoms are collected, in the per-residue loop:

```python
_mol = _classify_mol_type(
    _residue_name[_key],
    frozenset(_residue_atoms[_key].keys()),
)
_b_factors[_i, :] = float(_mol)
```

The `aatype` logic is unchanged — nucleotides already fall through to `20` (X) via
`restype_3to1.get(resname, "X")`.

---

## `to_pdb()` impact

`to_pdb()` writes `b_factors` to the PDB B-factor column (chars 61–66). With the new
semantics it will write `0.00`, `1.00`, or `2.00`. No structural change is needed —
this is acceptable since the codebase uses `to_pdb()` for model output, not for
preserving experimental B-factors.

---

## Test plan

Tests are written as module-level `pytest` functions using `tmp_path` to materialise
minimal inline PDB strings. A shared `_make_atom_line()` helper produces correctly
formatted 80-character PDB ATOM / HETATM records.

### Constant tests

| Test | Assertion |
|------|-----------|
| `test_mol_type_constants` | `MOL_TYPE_PROTEIN==0`, `MOL_TYPE_DNA==1`, `MOL_TYPE_RNA==2` |
| `test_dna_restype_constants` | keys `DA/DC/DG/DT`, values `a/c/g/t`, order `0–3` |
| `test_rna_restype_constants` | keys `A/C/G/U`, values `a/c/g/u`, order `0–3` |

### `_classify_mol_type` unit tests

| Test | Input | Expected |
|------|-------|----------|
| `test_classify_mol_type_protein` | `"ALA"`, any atoms | `MOL_TYPE_PROTEIN` |
| `test_classify_mol_type_dna_canonical` | `"DA"`, any atoms | `MOL_TYPE_DNA` |
| `test_classify_mol_type_dna_bare_t` | `"T"`, any atoms | `MOL_TYPE_DNA` |
| `test_classify_mol_type_rna_u` | `"U"`, no atoms | `MOL_TYPE_RNA` |
| `test_classify_mol_type_rna_via_o2prime` | `"A"`, atoms `{"O2'"}` | `MOL_TYPE_RNA` |
| `test_classify_mol_type_dna_no_prefix_no_o2prime` | `"A"`, atoms `{"C1'"}` | `MOL_TYPE_DNA` |
| `test_classify_mol_type_unknown_defaults_protein` | `"UNK"`, any | `MOL_TYPE_PROTEIN` |

### `protein_from_pdb()` integration tests

| Test | PDB content | Assertion |
|------|-------------|-----------|
| `test_protein_from_pdb_protein_only` | 2× ALA ATOM | b_factors all `0.0`, aatype matches |
| `test_protein_from_pdb_dna_canonical` | 2× DA + 2× DT ATOM | b_factors all `1.0`, aatype all `20` |
| `test_protein_from_pdb_rna_canonical` | 2× A + 2× U ATOM (with `O2'`) | b_factors all `2.0`, aatype all `20` |
| `test_protein_from_pdb_dna_no_prefix` | A/C/G/T ATOM without `O2'` | b_factors all `1.0` |
| `test_protein_from_pdb_protein_dna_multichain` | chain A=ALA, chain B=DA | chain_index `[0,0,1,1]`, b_factors `[0,0,1,1]` |
| `test_protein_from_pdb_multi_chain_protein` | chain A + chain B protein | chain_index `[0,0,1,1]` |
| `test_protein_from_pdb_residue_index` | RESSEQ 5 and 10 | residue_index `[5, 10]` |
| `test_protein_from_pdb_no_atoms_raises` | no ATOM records | `ValueError` |
| `test_protein_from_pdb_ignores_water_hetatm` | ALA ATOM + HOH HETATM | N_res=1, no water residue |
| `test_protein_from_pdb_ignores_ion_hetatm` | ALA ATOM + MG HETATM | N_res=1 |
| `test_protein_from_pdb_ignores_ligand_hetatm` | ALA ATOM + ATP HETATM | N_res=1 |
| `test_protein_from_pdb_correct_residue_count_with_hetatm` | 3 ATOM residues + 2 HETATM | N_res=3 |

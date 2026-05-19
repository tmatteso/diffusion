# DNA / RNA Molecule-Type Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable DNA/RNA lookup constants and molecule-type encoding to
`atom_utils.py` so that `protein_from_pdb()` populates `Protein.b_factors` with
`0.0`/`1.0`/`2.0` (protein/DNA/RNA) instead of real B-factors, with HETATM records
explicitly detected and skipped.

**Architecture:** Three layered changes in a single module: (1) new immutable global
constants (`MappingProxyType` + `Final`) for DNA/RNA monomer lookup; (2) a new private
classifier `_classify_mol_type` that uses residue name and atom set to identify mol type;
(3) updates to `protein_from_pdb()` that replace the implicit ATOM-only filter with an
explicit HETATM skip and set `b_factors` from the classifier rather than raw PDB values.

**Tech Stack:** Python stdlib `types.MappingProxyType`, `typing.Final`; NumPy; pytest
with `tmp_path`; pre-commit hooks: black, ruff (D/ANN/I/FBT/PT/…), pyright (basic),
enforce-einops, enforce-jaxtyping, pytest, commitlint.

---

## Hook reminders (read before every commit)

- **commitlint**: header ≤ 72 chars, body lines ≤ 72 chars, prefix required
  (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`).
- **enforce-einops**: ban `.reshape(`, `.view(`, `.permute(`, `.unsqueeze(`,
  `.squeeze(`, `torch.einsum(`.
- **enforce-jaxtyping**: ban bare `torch.Tensor` / `Tensor` in annotations; wrap
  with `Float[torch.Tensor, "..."]` etc.
- **ruff D**: every public function and class needs a Google-style docstring.
  Test files are **not** exempt from `D` (only from `ANN`).
- **pyright**: `reportMissingParameterType` and `reportReturnType` are **errors**.
  `MappingProxyType[str, int]` requires explicit type args.
- **pytest**: all tests (old + new) must pass before every commit.

---

## File map

| File | Action | Responsibility |
|------|--------|---------------|
| `pallatom/helpers/atom_utils.py` | Modify | Add imports, constants, `_classify_mol_type`, update `protein_from_pdb` |
| `pallatom/tests/helpers/test_atom_utils.py` | Modify | Add `_pdb_atom_line` helper, all new test functions |

No new files are created.

---

## Task 1: Add mol-type and nucleotide constants

**Files:**
- Modify: `pallatom/helpers/atom_utils.py:1-5` (imports)
- Modify: `pallatom/helpers/atom_utils.py:136` (after `restype_3to1`)
- Modify: `pallatom/tests/helpers/test_atom_utils.py` (new imports + 3 test functions)

---

- [ ] **Step 1.1 — Write the failing constant tests**

Append these three test functions at the end of
`pallatom/tests/helpers/test_atom_utils.py`.

First, update the import block at the top of the test file. The full revised
`from helpers.atom_utils import (...)` block (isort-sorted, replacing the
existing one):

```python
from helpers.atom_utils import (
    ATOM5_CB,
    ATOM37_C,
    ATOM37_CA,
    ATOM37_CB,
    ATOM37_N,
    ATOM37_O,
    DNA_RESTYPE_3TO1,
    DNA_RESTYPE_ORDER,
    DNA_RESTYPES,
    MOL_TYPE_DNA,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
    PDB_MAX_CHAINS,
    Protein,
    RNA_RESTYPE_3TO1,
    RNA_RESTYPE_ORDER,
    RNA_RESTYPES,
    _chain_end,
    atom37_to_atom5,
    atom37_to_cb,
    atom_types,
    center_positions,
    get_cb_coords,
    make_fixed_size,
    make_np_example,
    pseudo_cb,
    restype_num,
    to_pdb,
)
```

Also add `import pathlib` to the stdlib section (before `from collections.abc`):

```python
from collections.abc import Mapping
import pathlib
```

Then append the three constant tests:

```python
# ---------------------------------------------------------------------------
# Molecule-type and nucleotide constants
# ---------------------------------------------------------------------------


def test_mol_type_constants() -> None:
    """MOL_TYPE_* constants encode protein=0, DNA=1, RNA=2."""
    assert MOL_TYPE_PROTEIN == 0
    assert MOL_TYPE_DNA == 1
    assert MOL_TYPE_RNA == 2


def test_dna_restype_constants() -> None:
    """DNA_RESTYPE_* constants have the expected keys, values, and ordering."""
    from types import MappingProxyType

    assert list(DNA_RESTYPES) == ["DA", "DC", "DG", "DT"]
    assert isinstance(DNA_RESTYPE_ORDER, MappingProxyType)
    assert dict(DNA_RESTYPE_ORDER) == {"DA": 0, "DC": 1, "DG": 2, "DT": 3}
    assert isinstance(DNA_RESTYPE_3TO1, MappingProxyType)
    assert dict(DNA_RESTYPE_3TO1) == {"DA": "a", "DC": "c", "DG": "g", "DT": "t"}


def test_rna_restype_constants() -> None:
    """RNA_RESTYPE_* constants have the expected keys, values, and ordering."""
    from types import MappingProxyType

    assert list(RNA_RESTYPES) == ["A", "C", "G", "U"]
    assert isinstance(RNA_RESTYPE_ORDER, MappingProxyType)
    assert dict(RNA_RESTYPE_ORDER) == {"A": 0, "C": 1, "G": 2, "U": 3}
    assert isinstance(RNA_RESTYPE_3TO1, MappingProxyType)
    assert dict(RNA_RESTYPE_3TO1) == {"A": "a", "C": "c", "G": "g", "U": "u"}
```

- [ ] **Step 1.2 — Run tests to confirm they fail**

```bash
pytest pallatom/tests/helpers/test_atom_utils.py::test_mol_type_constants \
       pallatom/tests/helpers/test_atom_utils.py::test_dna_restype_constants \
       pallatom/tests/helpers/test_atom_utils.py::test_rna_restype_constants -v
```

Expected: **ImportError / collection error** — the symbols don't exist yet.

- [ ] **Step 1.3 — Add imports and constants to `atom_utils.py`**

Add two imports to the stdlib section at the top of
`pallatom/helpers/atom_utils.py` (after `import dataclasses`,
before `from collections.abc`):

```python
import dataclasses
from collections.abc import Mapping, MutableMapping
from types import MappingProxyType
from typing import Final
```

Then, immediately after line 136 (`restype_3to1 = {v: k for k, v in restype_1to3.items()}`),
insert:

```python
# Molecule type encoding stored in Protein.b_factors (broadcast to all atom slots per residue).
# 0.0 = amino-acid residue, 1.0 = DNA nucleotide, 2.0 = RNA nucleotide.
MOL_TYPE_PROTEIN: Final[int] = 0
MOL_TYPE_DNA: Final[int] = 1
MOL_TYPE_RNA: Final[int] = 2

# DNA monomers — PDB ATOM residue names for deoxyribonucleotides.
# Mirrors OpenFold's restype_3to1 / restype_order naming convention.
DNA_RESTYPES: Final[list[str]] = ["DA", "DC", "DG", "DT"]
DNA_RESTYPE_ORDER: Final[MappingProxyType[str, int]] = MappingProxyType(
    {restype: i for i, restype in enumerate(DNA_RESTYPES)}
)
DNA_RESTYPE_3TO1: Final[MappingProxyType[str, str]] = MappingProxyType(
    {"DA": "a", "DC": "c", "DG": "g", "DT": "t"}
)

# RNA monomers — PDB ATOM residue names for ribonucleotides.
RNA_RESTYPES: Final[list[str]] = ["A", "C", "G", "U"]
RNA_RESTYPE_ORDER: Final[MappingProxyType[str, int]] = MappingProxyType(
    {restype: i for i, restype in enumerate(RNA_RESTYPES)}
)
RNA_RESTYPE_3TO1: Final[MappingProxyType[str, str]] = MappingProxyType(
    {"A": "a", "C": "c", "G": "g", "U": "u"}
)
```

- [ ] **Step 1.4 — Run tests to confirm they pass**

```bash
pytest pallatom/tests/helpers/test_atom_utils.py::test_mol_type_constants \
       pallatom/tests/helpers/test_atom_utils.py::test_dna_restype_constants \
       pallatom/tests/helpers/test_atom_utils.py::test_rna_restype_constants -v
```

Expected: **3 passed**.

- [ ] **Step 1.5 — Run the full suite to confirm no regressions**

```bash
pytest pallatom/tests/helpers/test_atom_utils.py -v
```

Expected: all previously passing tests still pass, 3 new tests pass.

- [ ] **Step 1.6 — Commit**

```bash
git add pallatom/helpers/atom_utils.py \
        pallatom/tests/helpers/test_atom_utils.py
git commit -m "$(cat <<'EOF'
feat: add mol-type and DNA/RNA constants to atom_utils

MOL_TYPE_{PROTEIN,DNA,RNA} = 0/1/2 for b_factors encoding.
DNA_RESTYPE_{ORDER,3TO1} and RNA_RESTYPE_{ORDER,3TO1} mirror
the OpenFold restype_3to1/restype_order convention. All dicts
are MappingProxyType for runtime immutability.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `_classify_mol_type` private function

**Files:**
- Modify: `pallatom/helpers/atom_utils.py` (new function before `protein_from_pdb`)
- Modify: `pallatom/tests/helpers/test_atom_utils.py` (import + 7 unit tests)

---

- [ ] **Step 2.1 — Update the test-file import to include `_classify_mol_type`**

Add `_classify_mol_type` to the `from helpers.atom_utils import (...)` block
(it sorts after `_chain_end` alphabetically):

```python
from helpers.atom_utils import (
    ATOM5_CB,
    ATOM37_C,
    ATOM37_CA,
    ATOM37_CB,
    ATOM37_N,
    ATOM37_O,
    DNA_RESTYPE_3TO1,
    DNA_RESTYPE_ORDER,
    DNA_RESTYPES,
    MOL_TYPE_DNA,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
    PDB_MAX_CHAINS,
    Protein,
    RNA_RESTYPE_3TO1,
    RNA_RESTYPE_ORDER,
    RNA_RESTYPES,
    _chain_end,
    _classify_mol_type,
    atom37_to_atom5,
    atom37_to_cb,
    atom_types,
    center_positions,
    get_cb_coords,
    make_fixed_size,
    make_np_example,
    pseudo_cb,
    restype_num,
    to_pdb,
)
```

- [ ] **Step 2.2 — Write the failing unit tests**

Append after the RNA constants tests block:

```python
# ---------------------------------------------------------------------------
# _classify_mol_type
# ---------------------------------------------------------------------------


def test_classify_mol_type_protein() -> None:
    """_classify_mol_type returns MOL_TYPE_PROTEIN for amino-acid residue names."""
    assert _classify_mol_type("ALA", frozenset()) == MOL_TYPE_PROTEIN
    assert _classify_mol_type("GLY", frozenset()) == MOL_TYPE_PROTEIN
    assert _classify_mol_type("VAL", frozenset()) == MOL_TYPE_PROTEIN


def test_classify_mol_type_dna_canonical() -> None:
    """_classify_mol_type returns MOL_TYPE_DNA for D-prefix nucleotide names."""
    assert _classify_mol_type("DA", frozenset()) == MOL_TYPE_DNA
    assert _classify_mol_type("DC", frozenset()) == MOL_TYPE_DNA
    assert _classify_mol_type("DG", frozenset()) == MOL_TYPE_DNA
    assert _classify_mol_type("DT", frozenset()) == MOL_TYPE_DNA


def test_classify_mol_type_dna_bare_t() -> None:
    """_classify_mol_type returns MOL_TYPE_DNA for bare T (D-prefix-less thymine)."""
    assert _classify_mol_type("T", frozenset()) == MOL_TYPE_DNA
    assert _classify_mol_type("T", frozenset({"C1'", "C2'"})) == MOL_TYPE_DNA


def test_classify_mol_type_rna_u() -> None:
    """_classify_mol_type returns MOL_TYPE_RNA for U regardless of atoms present."""
    assert _classify_mol_type("U", frozenset()) == MOL_TYPE_RNA
    assert _classify_mol_type("U", frozenset({"C1'"})) == MOL_TYPE_RNA


def test_classify_mol_type_rna_via_o2prime() -> None:
    """_classify_mol_type returns MOL_TYPE_RNA for A/C/G when O2' atom is present."""
    assert _classify_mol_type("A", frozenset({"O2'", "C1'"})) == MOL_TYPE_RNA
    assert _classify_mol_type("C", frozenset({"O2'"})) == MOL_TYPE_RNA
    assert _classify_mol_type("G", frozenset({"O2'"})) == MOL_TYPE_RNA


def test_classify_mol_type_dna_no_prefix_no_o2prime() -> None:
    """_classify_mol_type returns MOL_TYPE_DNA for A/C/G when no O2' atom is present."""
    assert _classify_mol_type("A", frozenset({"C1'", "C2'"})) == MOL_TYPE_DNA
    assert _classify_mol_type("C", frozenset()) == MOL_TYPE_DNA
    assert _classify_mol_type("G", frozenset({"P"})) == MOL_TYPE_DNA


def test_classify_mol_type_unknown_defaults_protein() -> None:
    """_classify_mol_type returns MOL_TYPE_PROTEIN for unrecognised residue names."""
    assert _classify_mol_type("UNK", frozenset()) == MOL_TYPE_PROTEIN
    assert _classify_mol_type("MSE", frozenset()) == MOL_TYPE_PROTEIN
    assert _classify_mol_type("", frozenset()) == MOL_TYPE_PROTEIN
```

- [ ] **Step 2.3 — Run tests to confirm they fail**

```bash
pytest pallatom/tests/helpers/test_atom_utils.py -k "classify_mol_type" -v
```

Expected: **ImportError / collection error** — `_classify_mol_type` doesn't exist yet.

- [ ] **Step 2.4 — Add `_classify_mol_type` to `atom_utils.py`**

Insert the following function immediately before `def protein_from_pdb` (currently
at line 449, which shifts down after the constants added in Task 1):

```python
def _classify_mol_type(resname: str, atom_names: frozenset[str]) -> int:
    """Return the molecule-type constant for a PDB residue.

    Uses residue name first; falls back to 2'-OH atom presence for the
    ambiguous single-letter nucleotides A/C/G, which appear in both
    D-prefix-less DNA and canonical RNA.

    Args:
        resname: Stripped residue name from the PDB ATOM record
            (e.g. "ALA", "DA", "A").
        atom_names: All atom names observed in this residue; used to
            detect the RNA 2'-hydroxyl oxygen "O2'".

    Returns:
        MOL_TYPE_PROTEIN (0), MOL_TYPE_DNA (1), or MOL_TYPE_RNA (2).
    """
    if resname in restype_3to1:
        return MOL_TYPE_PROTEIN
    if resname in DNA_RESTYPE_3TO1 or resname == "T":
        return MOL_TYPE_DNA
    if resname == "U":
        return MOL_TYPE_RNA
    if resname in RNA_RESTYPE_3TO1:  # A / C / G — ambiguous
        return MOL_TYPE_RNA if "O2'" in atom_names else MOL_TYPE_DNA
    return MOL_TYPE_PROTEIN  # unknown residue → treat as protein
```

- [ ] **Step 2.5 — Run the unit tests**

```bash
pytest pallatom/tests/helpers/test_atom_utils.py -k "classify_mol_type" -v
```

Expected: **7 passed**.

- [ ] **Step 2.6 — Run the full suite**

```bash
pytest pallatom/tests/helpers/test_atom_utils.py -v
```

Expected: all previously passing tests still pass, 7 new tests pass.

- [ ] **Step 2.7 — Commit**

```bash
git add pallatom/helpers/atom_utils.py \
        pallatom/tests/helpers/test_atom_utils.py
git commit -m "$(cat <<'EOF'
feat: add _classify_mol_type for protein/DNA/RNA detection

Classifies PDB residues using name lookup first, then 2'-OH
atom presence for ambiguous A/C/G nucleotides. Bare T forces
DNA; bare U forces RNA; all others default to protein.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update `protein_from_pdb` — HETATM skip and mol-type b_factors

**Files:**
- Modify: `pallatom/helpers/atom_utils.py` (`protein_from_pdb` body)
- Modify: `pallatom/tests/helpers/test_atom_utils.py` (import, helper, 12 integration tests)

---

- [ ] **Step 3.1 — Add `protein_from_pdb` to the test import block**

Add `protein_from_pdb` to the `from helpers.atom_utils import (...)` block
(sorts between `pseudo_cb` and `restype_num` alphabetically — place it after
`pseudo_cb`):

```python
from helpers.atom_utils import (
    ATOM5_CB,
    ATOM37_C,
    ATOM37_CA,
    ATOM37_CB,
    ATOM37_N,
    ATOM37_O,
    DNA_RESTYPE_3TO1,
    DNA_RESTYPE_ORDER,
    DNA_RESTYPES,
    MOL_TYPE_DNA,
    MOL_TYPE_PROTEIN,
    MOL_TYPE_RNA,
    PDB_MAX_CHAINS,
    Protein,
    RNA_RESTYPE_3TO1,
    RNA_RESTYPE_ORDER,
    RNA_RESTYPES,
    _chain_end,
    _classify_mol_type,
    atom37_to_atom5,
    atom37_to_cb,
    atom_types,
    center_positions,
    get_cb_coords,
    make_fixed_size,
    make_np_example,
    protein_from_pdb,
    pseudo_cb,
    restype_num,
    to_pdb,
)
```

- [ ] **Step 3.2 — Add the PDB record helper and write the failing integration tests**

Append the following to `pallatom/tests/helpers/test_atom_utils.py` after the
`_classify_mol_type` tests:

```python
# ---------------------------------------------------------------------------
# protein_from_pdb helpers
# ---------------------------------------------------------------------------


def _pdb_atom_line(
    record: str,
    serial: int,
    atom_name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float = 1.0,
    y: float = 2.0,
    z: float = 3.0,
) -> str:
    """Build an 80-character PDB ATOM or HETATM record for test fixtures.

    Args:
        record: Record type string, e.g. "ATOM" or "HETATM".
        serial: Atom serial number.
        atom_name: PDB atom name (e.g. "N", "CA", "O2'").
        resname: Residue name (e.g. "ALA", "DA", "HOH").
        chain: Single-character chain ID.
        resseq: Residue sequence number.
        x: Cartesian x coordinate in Angstroms.
        y: Cartesian y coordinate in Angstroms.
        z: Cartesian z coordinate in Angstroms.

    Returns:
        An 80-character string formatted as a PDB record.
    """
    atom_field = f" {atom_name:<3}" if len(atom_name) < 4 else atom_name[:4]
    return (
        f"{record:<6}{serial:>5} {atom_field} {resname:>3} {chain}{resseq:>4}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
    ).ljust(80)


# ---------------------------------------------------------------------------
# protein_from_pdb — integration tests
# ---------------------------------------------------------------------------


def test_protein_from_pdb_protein_only(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb sets b_factors to MOL_TYPE_PROTEIN for amino-acid residues."""
    pdb = tmp_path / "prot.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", "ALA", "A", 1) + "\n"
        + _pdb_atom_line("ATOM", 2, "CA", "ALA", "A", 1) + "\n"
        + _pdb_atom_line("ATOM", 3, "N", "GLY", "A", 2) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    assert prot.atom_positions.shape == (2, 37, 3)
    assert np.all(prot.b_factors == float(MOL_TYPE_PROTEIN))
    assert prot.aatype[0] == 0   # ALA → index 0 in restypes


def test_protein_from_pdb_dna_canonical(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb sets b_factors to MOL_TYPE_DNA for DA/DC/DG/DT residues."""
    pdb = tmp_path / "dna.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "P", "DA", "A", 1) + "\n"
        + _pdb_atom_line("ATOM", 2, "P", "DT", "A", 2) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    assert prot.atom_positions.shape == (2, 37, 3)
    assert np.all(prot.b_factors == float(MOL_TYPE_DNA))
    assert np.all(prot.aatype == 20)


def test_protein_from_pdb_rna_canonical(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb classifies A/U residues as RNA when O2' atom is present."""
    pdb = tmp_path / "rna.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "O2'", "A", "A", 1) + "\n"
        + _pdb_atom_line("ATOM", 2, "O2'", "U", "A", 2) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    assert prot.atom_positions.shape == (2, 37, 3)
    assert np.all(prot.b_factors == float(MOL_TYPE_RNA))
    assert np.all(prot.aatype == 20)


def test_protein_from_pdb_dna_no_prefix(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb classifies A/T residues as DNA when no O2' atom is present."""
    pdb = tmp_path / "dna_noprefix.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "C1'", "A", "A", 1) + "\n"
        + _pdb_atom_line("ATOM", 2, "C1'", "T", "A", 2) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    assert prot.atom_positions.shape == (2, 37, 3)
    assert np.all(prot.b_factors == float(MOL_TYPE_DNA))


def test_protein_from_pdb_protein_dna_multichain(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb assigns correct chain_index and b_factors in a protein+DNA complex."""
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", "ALA", "A", 1) + "\n"
        + _pdb_atom_line("ATOM", 2, "N", "GLY", "A", 2) + "\n"
        + _pdb_atom_line("ATOM", 3, "P", "DA", "B", 1) + "\n"
        + _pdb_atom_line("ATOM", 4, "P", "DT", "B", 2) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    assert prot.atom_positions.shape == (4, 37, 3)
    np.testing.assert_array_equal(prot.chain_index, [0, 0, 1, 1])
    assert np.all(prot.b_factors[:2] == float(MOL_TYPE_PROTEIN))
    assert np.all(prot.b_factors[2:] == float(MOL_TYPE_DNA))


def test_protein_from_pdb_multi_chain_protein(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb assigns distinct chain_index values for each PDB chain ID."""
    pdb = tmp_path / "twochains.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", "ALA", "A", 1) + "\n"
        + _pdb_atom_line("ATOM", 2, "N", "GLY", "A", 2) + "\n"
        + _pdb_atom_line("ATOM", 3, "N", "ALA", "B", 1) + "\n"
        + _pdb_atom_line("ATOM", 4, "N", "GLY", "B", 2) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    np.testing.assert_array_equal(prot.chain_index, [0, 0, 1, 1])


def test_protein_from_pdb_residue_index(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb sets residue_index to the PDB RESSEQ numbers."""
    pdb = tmp_path / "resseq.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", "ALA", "A", 5) + "\n"
        + _pdb_atom_line("ATOM", 2, "N", "ALA", "A", 10) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    np.testing.assert_array_equal(prot.residue_index, [5, 10])


def test_protein_from_pdb_no_atoms_raises(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb raises ValueError when the PDB file has no ATOM records."""
    pdb = tmp_path / "empty.pdb"
    pdb.write_text("REMARK empty structure\nHEADER test\n")
    with pytest.raises(ValueError, match="No ATOM records"):
        protein_from_pdb(str(pdb))


def test_protein_from_pdb_ignores_water_hetatm(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb excludes HOH HETATM records from the output Protein."""
    pdb = tmp_path / "water.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", "ALA", "A", 1) + "\n"
        + _pdb_atom_line("HETATM", 2, "O", "HOH", "A", 100) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    assert prot.atom_positions.shape[0] == 1


def test_protein_from_pdb_ignores_ion_hetatm(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb excludes ion HETATM records (MG, ZN, etc.) from the output."""
    pdb = tmp_path / "ion.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", "ALA", "A", 1) + "\n"
        + _pdb_atom_line("HETATM", 2, "MG", "MG", "A", 200) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    assert prot.atom_positions.shape[0] == 1


def test_protein_from_pdb_ignores_ligand_hetatm(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb excludes ligand HETATM records (ATP, HEM, etc.) from the output."""
    pdb = tmp_path / "ligand.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", "ALA", "A", 1) + "\n"
        + _pdb_atom_line("HETATM", 2, "N1", "ATP", "A", 300) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    assert prot.atom_positions.shape[0] == 1


def test_protein_from_pdb_correct_residue_count_with_hetatm(tmp_path: pathlib.Path) -> None:
    """protein_from_pdb counts only ATOM-record residues when HETATM records are present."""
    pdb = tmp_path / "mixed.pdb"
    pdb.write_text(
        _pdb_atom_line("ATOM", 1, "N", "ALA", "A", 1) + "\n"
        + _pdb_atom_line("ATOM", 2, "N", "GLY", "A", 2) + "\n"
        + _pdb_atom_line("ATOM", 3, "N", "SER", "A", 3) + "\n"
        + _pdb_atom_line("HETATM", 4, "O", "HOH", "A", 100) + "\n"
        + _pdb_atom_line("HETATM", 5, "O", "HOH", "A", 101) + "\n"
    )
    prot = protein_from_pdb(str(pdb))
    assert prot.atom_positions.shape[0] == 3
```

- [ ] **Step 3.3 — Run tests to confirm they fail**

```bash
pytest pallatom/tests/helpers/test_atom_utils.py -k "protein_from_pdb" -v
```

Expected: **12 failed** — `protein_from_pdb` not yet imported / behaviour not yet updated.
(Import errors for symbols added in this step are also acceptable at this stage.)

- [ ] **Step 3.4 — Update `protein_from_pdb` in `atom_utils.py`**

Replace the current `protein_from_pdb` body. The full function after editing:

```python
def protein_from_pdb(pdb_path: str) -> "Protein":
    """Parse a PDB file (ATOM records only) into a Protein using the atom37 layout.

    HETATM records (water, ions, ligands, crystallographic reagents) are
    explicitly detected and skipped; only ATOM records produce residues in
    the returned Protein. The b_factors field encodes molecule type rather
    than real PDB B-factors: 0.0 = protein, 1.0 = DNA, 2.0 = RNA.

    Args:
        pdb_path: Path to the PDB file to parse.

    Returns:
        A Protein with atom37-layout coordinates; b_factors encodes
        MOL_TYPE_PROTEIN / MOL_TYPE_DNA / MOL_TYPE_RNA per residue.

    Raises:
        ValueError: If the file contains no ATOM records.
    """
    _atom_type_idx = {name: i for i, name in enumerate(atom_types)}
    _seen_chains: dict[str, int] = {}

    _residue_atoms: MutableMapping[
        tuple[str, int, str], MutableMapping[str, tuple[float, float, float, float]]
    ] = {}
    _residue_name: MutableMapping[tuple[str, int, str], str] = {}
    _residue_chain: MutableMapping[tuple[str, int, str], str] = {}

    with open(pdb_path) as _fh:
        for _line in _fh:
            _rec = _line[:6].strip()
            if _rec == "HETATM":
                continue  # skip: water, ions, ligands, crystallographic reagents
            if _rec != "ATOM":
                continue
            _atom_name = _line[12:16].strip()
            _alt_loc = _line[16]
            if _alt_loc not in (" ", "A"):
                continue
            _resname = _line[17:20].strip()
            _chain_id = _line[21]
            _resseq = int(_line[22:26])
            _icode = _line[26]
            _x = float(_line[30:38])
            _y = float(_line[38:46])
            _z = float(_line[46:54])
            _bfac = float(_line[60:66]) if len(_line) > 66 else 0.0
            _key = (_chain_id, _resseq, _icode)
            if _key not in _residue_atoms:
                _residue_atoms[_key] = {}
                _residue_name[_key] = _resname
                _residue_chain[_key] = _chain_id
            if _atom_name not in _residue_atoms[_key]:
                _residue_atoms[_key][_atom_name] = (_x, _y, _z, _bfac)

    if not _residue_atoms:
        raise ValueError(f"No ATOM records found in {pdb_path}")

    _keys = sorted(_residue_atoms)
    N_res = len(_keys)
    _atom_positions = np.zeros((N_res, 37, 3), dtype=np.float64)
    _atom_mask = np.zeros((N_res, 37), dtype=np.float64)
    _b_factors = np.zeros((N_res, 37), dtype=np.float64)
    _aatype = np.zeros(N_res, dtype=np.intp)
    _residue_index = np.zeros(N_res, dtype=np.intp)
    _chain_index = np.zeros(N_res, dtype=np.intp)

    for _i, _key in enumerate(_keys):
        _cid = _residue_chain[_key]
        if _cid not in _seen_chains:
            _seen_chains[_cid] = len(_seen_chains)
        _chain_index[_i] = _seen_chains[_cid]
        _residue_index[_i] = _key[1]
        _one = restype_3to1.get(_residue_name[_key], "X")
        _aatype[_i] = restype_order.get(_one, 20)
        _mol = _classify_mol_type(
            _residue_name[_key],
            frozenset(_residue_atoms[_key].keys()),
        )
        _b_factors[_i, :] = float(_mol)
        for _aname, (_x, _y, _z, _bf) in _residue_atoms[_key].items():
            if _aname in _atom_type_idx:
                _idx = _atom_type_idx[_aname]
                _atom_positions[_i, _idx] = [_x, _y, _z]
                _atom_mask[_i, _idx] = 1.0

    return Protein(
        atom_positions=_atom_positions,
        aatype=_aatype,
        atom_mask=_atom_mask,
        residue_index=_residue_index,
        chain_index=_chain_index,
        b_factors=_b_factors,
    )
```

- [ ] **Step 3.5 — Run the integration tests**

```bash
pytest pallatom/tests/helpers/test_atom_utils.py -k "protein_from_pdb" -v
```

Expected: **12 passed**.

- [ ] **Step 3.6 — Run the full test suite**

```bash
pytest pallatom/tests/helpers/test_atom_utils.py -v
```

Expected: all previously passing tests + all 22 new tests pass (3 constant + 7
`_classify_mol_type` + 12 `protein_from_pdb`).

- [ ] **Step 3.7 — Commit**

```bash
git add pallatom/helpers/atom_utils.py \
        pallatom/tests/helpers/test_atom_utils.py
git commit -m "$(cat <<'EOF'
feat: update protein_from_pdb for DNA/RNA and HETATM handling

Replace implicit ATOM-only filter with explicit HETATM skip.
Set b_factors from _classify_mol_type (0=protein, 1=DNA, 2=RNA)
instead of raw PDB B-factors. All 22 new tests pass.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist

- [x] **Spec coverage**
  - MOL_TYPE_{PROTEIN,DNA,RNA} constants → Task 1 ✓
  - DNA_RESTYPE_{ORDER,3TO1,S} constants → Task 1 ✓
  - RNA_RESTYPE_{ORDER,3TO1,S} constants → Task 1 ✓
  - `MappingProxyType` runtime immutability → Task 1 ✓
  - `Final` pyright immutability → Task 1 ✓
  - `_classify_mol_type` all 5 branches → Task 2 ✓
  - D-prefix-less DNA via O2' check → Task 2 + Task 3 ✓
  - Bare T → DNA, bare U → RNA → Task 2 + Task 3 ✓
  - Explicit HETATM skip (water / ion / ligand) → Task 3 ✓
  - b_factors from mol type (not raw PDB) → Task 3 ✓
  - multi-chain chain_index test → Task 3 ✓
  - residue_index from RESSEQ → Task 3 ✓
  - no-atoms ValueError → Task 3 ✓

- [x] **Placeholder scan** — no TBD, TODO, or vague steps. All code blocks are complete.

- [x] **Type consistency**
  - `_classify_mol_type(resname: str, atom_names: frozenset[str]) -> int` used identically in
    Task 2 (definition) and Task 3 (call site with `frozenset(_residue_atoms[_key].keys())`). ✓
  - `_pdb_atom_line` returns `str`; all callers pass the result to `pdb.write_text(...)`. ✓
  - `DNA_RESTYPE_ORDER: Final[MappingProxyType[str, int]]` — explicit type args satisfy
    pyright `reportMissingTypeArgument`. ✓
  - `_bf` in the inner loop is still destructured (needed for 4-tuple unpacking) but no longer
    stored; the `_` prefix convention exempts it from ruff F841. ✓

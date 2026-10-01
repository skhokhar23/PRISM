"""Tests for src.sasa_utils, specifically the chain-isolation fix (2026-08-10).

Root cause under test: get_asa_complex() used to run FreeSASA on the whole
downloaded PDB file (every chain present, e.g. both receptor and ligand of
an already-bound complex), then only filter the *results* down to the
requested chain afterward. That measures each residue's accessibility with
its binding partner already in contact, which makes true interface residues
read as artificially buried. The fix detaches every non-requested chain
from the structure *before* the SASA calculation runs.
"""
import pytest

freesasa = pytest.importorskip("freesasa")
pytest.importorskip("Bio.PDB")

from src.sasa_utils import get_asa_complex  # noqa: E402


def _atom(serial, name, resname, chain, resseq, x, y, z, record="ATOM", element=None, altloc=" "):
    element = element if element is not None else name.strip()[0]
    return (
        f"{record:<6s}{serial:5d} {name:<4s}{altloc}{resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}  \n"
    )


def _write_pdb(path, atom_lines):
    with open(path, "w") as f:
        f.writelines(atom_lines)
        f.write("END\n")


def _chain_a_atoms():
    return [
        _atom(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0),
        _atom(2, "CA", "ALA", "A", 1, 1.5, 0.0, 0.0),
        _atom(3, "C", "ALA", "A", 1, 2.0, 1.4, 0.0),
        _atom(4, "O", "ALA", "A", 1, 1.3, 2.4, 0.0),
        _atom(5, "CB", "ALA", "A", 1, 2.2, -1.1, 1.1),
    ]


def _chain_b_atoms():
    # Close to chain A, as if in contact -- a plausible bound-partner
    # placement. Exact shielding magnitude doesn't matter for this test;
    # what matters is that requesting chain A alone gives the same answer
    # whether or not this chain physically exists in the same file.
    return [
        _atom(6, "N", "ALA", "B", 1, 2.4, -1.3, 1.4),
        _atom(7, "CA", "ALA", "B", 1, 2.6, -2.6, 1.9),
        _atom(8, "C", "ALA", "B", 1, 4.0, -3.1, 1.6),
        _atom(9, "O", "ALA", "B", 1, 4.9, -2.4, 1.9),
        _atom(10, "CB", "ALA", "B", 1, 2.4, -2.7, 3.4),
    ]


def _chain_c_atoms():
    # Far away and unrelated -- e.g. a second copy in the crystal's
    # asymmetric unit, not part of either requested chain set.
    return [_atom(11, "CA", "ALA", "C", 1, 200.0, 0.0, 0.0)]


def test_isolation_matches_chain_only_file(tmp_path):
    """Requesting chain A from a file that ALSO contains chain B must give
    the identical result to a file that never had chain B in it at all --
    proving the other chain is actually removed before SASA runs, not just
    filtered out of the results afterward."""
    root_with_partner = tmp_path / "with_partner"
    (root_with_partner / "pdbs").mkdir(parents=True)
    _write_pdb(root_with_partner / "pdbs" / "xxxx.pdb", _chain_a_atoms() + _chain_b_atoms())

    root_isolated_ref = tmp_path / "isolated_ref"
    (root_isolated_ref / "pdbs").mkdir(parents=True)
    _write_pdb(root_isolated_ref / "pdbs" / "xxxx.pdb", _chain_a_atoms())

    got = get_asa_complex("xxxxA", str(root_with_partner))["A"]
    want = get_asa_complex("xxxxA", str(root_isolated_ref))["A"]

    assert set(got.keys()) == set(want.keys())
    for res_num in got:
        assert got[res_num] == pytest.approx(want[res_num], abs=0.01)


def test_requesting_two_chains_together_keeps_both_drops_others(tmp_path):
    """Mirrors hotspot.py's usage: a template like `1a28AB` requests two
    chains together. Both must survive (mutual burial between them is the
    whole point), but an unrelated third chain in the same file must not."""
    root = tmp_path
    (root / "pdbs").mkdir()
    _write_pdb(root / "pdbs" / "xxxx.pdb", _chain_a_atoms() + _chain_b_atoms() + _chain_c_atoms())

    asa = get_asa_complex("xxxxAB", str(root))
    assert set(asa.keys()) == {"A", "B"}


def test_no_chain_suffix_keeps_all_chains(tmp_path):
    """No chain letters requested ('xxxx') means no filtering at all --
    matches the pre-existing behaviour for this case."""
    root = tmp_path
    (root / "pdbs").mkdir()
    _write_pdb(root / "pdbs" / "xxxx.pdb", _chain_a_atoms() + _chain_b_atoms())

    asa = get_asa_complex("xxxx", str(root))
    assert set(asa.keys()) == {"A", "B"}

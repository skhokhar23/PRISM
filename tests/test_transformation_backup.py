import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src import transformation as tform
from src.transformation import (
    _parse_vec,
    _hotspot_supported,
    _contact_supported,
    apply_tm_transform,
    merge_pdb_files,
    pair_has_acceptable_clashes,
    transformer,
)


def test_parse_vec_round_trip():
    assert _parse_vec("[1.0, 2.0, 3.0]", [0.0, 0.0, 0.0]) == [1.0, 2.0, 3.0]
    assert _parse_vec(None, [0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
    assert _parse_vec("garbage", [9.0]) == [9.0]
    assert _parse_vec([4.0, 5.0, 6.0], None) == [4.0, 5.0, 6.0]


def test_apply_tm_transform_identity(tmp_path):
    src = tmp_path / "in.pdb"
    src.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  \n"
        "ATOM      2  CA  ALA B   1       4.000   5.000   6.000  1.00  0.00           C  \n"
    )
    out = tmp_path / "out.pdb"
    apply_tm_transform(str(src), str(out), [0.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    content = out.read_text()
    lines = [l for l in content.splitlines() if l.startswith("ATOM")]
    assert len(lines) == 2
    assert "1.000   2.000   3.000" in lines[0]


def test_apply_tm_transform_chain_filter(tmp_path):
    src = tmp_path / "in.pdb"
    src.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  \n"
        "ATOM      2  CA  ALA B   1       4.000   5.000   6.000  1.00  0.00           C  \n"
    )
    out = tmp_path / "out.pdb"
    apply_tm_transform(str(src), str(out), [0.0, 0.0, 0.0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], keep_chains={"A"})
    content = out.read_text()
    assert "ALA A" in content
    assert "ALA B" not in content


def test_apply_tm_transform_translation(tmp_path):
    src = tmp_path / "in.pdb"
    src.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n")
    out = tmp_path / "out.pdb"
    apply_tm_transform(str(src), str(out), [10.0, -5.0, 0.5], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    content = out.read_text()
    assert "10.000  -5.000   0.500" in content


def test_merge_pdb_files(tmp_path):
    r = tmp_path / "r.pdb"
    l = tmp_path / "l.pdb"
    r.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n")
    l.write_text("ATOM      9  CA  ALA B   1       5.000   0.000   0.000  1.00  0.00           C  \n")
    out = tmp_path / "merged.pdb"
    merge_pdb_files(str(r), str(l), str(out))
    text = out.read_text()
    atoms = [line for line in text.splitlines() if line.startswith("ATOM")]
    assert len(atoms) == 2
    serials = [int(line[6:11].strip()) for line in atoms]
    assert serials == [1, 2]
    assert text.strip().endswith("END")


def test_clash_check(tmp_path):
    r = tmp_path / "r.pdb"
    l = tmp_path / "l.pdb"
    r.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n")
    l.write_text("ATOM      1  CA  ALA B   1      10.000   0.000   0.000  1.00  0.00           C  \n")
    assert pair_has_acceptable_clashes(str(r), str(l)) is True

    l.write_text("ATOM      1  CA  ALA B   1       0.500   0.000   0.000  1.00  0.00           C  \n")
    assert pair_has_acceptable_clashes(str(r), str(l)) is False


def test_hotspot_filter_missing_returns_true(monkeypatch, tmp_path):
    monkeypatch.setattr(tform, "HOTSPOT_DIR", str(tmp_path))
    assert _hotspot_supported("notpresentXY", "X", {"X.A.10": "A.A.1"}) is True


def test_hotspot_filter_with_match(monkeypatch, tmp_path):
    monkeypatch.setattr(tform, "HOTSPOT_DIR", str(tmp_path))
    (tmp_path / "tplAB.json").write_text(json.dumps({"A": [["10", "ALA"], ["20", "ALA"]], "B": []}))
    assert _hotspot_supported("tplAB", "A", {"A.A.10": "P.A.50"}) is True
    assert _hotspot_supported("tplAB", "A", {"A.A.99": "P.A.50"}) is False


def test_contact_filter_pair_match(monkeypatch, tmp_path):
    monkeypatch.setattr(tform, "CONTACTS_DIR", str(tmp_path))
    (tmp_path / "tplAB.json").write_text(json.dumps([[10, 200], [11, 201]]))
    rec_match = {"A.A.10": "R.A.5"}
    lig_match = {"B.A.200": "L.B.5"}
    assert _contact_supported("tplAB", rec_match, lig_match, "A", "B") is True

    rec_match = {"A.A.99": "R.A.5"}
    assert _contact_supported("tplAB", rec_match, lig_match, "A", "B") is False


def test_transformer_no_alignments(monkeypatch, tmp_path):
    monkeypatch.setattr(tform, "ALIGNMENT_DIR", str(tmp_path))
    monkeypatch.setattr(tform, "TRANSFORMATION_DIR", str(tmp_path / "tr"))
    monkeypatch.setattr(tform, "MERGE_DIR", str(tmp_path / "out"))
    os.makedirs(tmp_path / "tr", exist_ok=True)
    os.makedirs(tmp_path / "out", exist_ok=True)
    assert transformer(["a"], ["b"]) == []

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


# ---------------------------------------------------------------------------
# F10: refine more than candidates[0] (added 2026-08-XX)
# ---------------------------------------------------------------------------

def _fake_candidates(n, base_score=10.0):
    return [
        {"template": f"tpl{i}", "rec_chain": "A", "lig_chain": "B",
         "score": base_score - i, "contacts": 30, "output_pdb": f"out{i}.pdb"}
        for i in range(n)
    ]


def test_transformer_sends_top_k_candidates(monkeypatch):
    monkeypatch.setattr(tform, "process_pair_for_template",
                        lambda r, l: _fake_candidates(8))
    monkeypatch.setattr(tform, "TOP_K_REFINE", 3)

    result = transformer(["rec"], ["lig"])

    assert len(result) == 3
    # process_pair_for_template already returns candidates sorted by score
    # descending, so transformer() just takes the top TOP_K_REFINE in order.
    assert [row[2] for row in result] == ["tpl0", "tpl1", "tpl2"]


def test_transformer_top_k_1_reproduces_pre_f10_behaviour(monkeypatch):
    monkeypatch.setattr(tform, "process_pair_for_template",
                        lambda r, l: _fake_candidates(5))
    monkeypatch.setattr(tform, "TOP_K_REFINE", 1)

    result = transformer(["rec"], ["lig"])

    assert len(result) == 1
    assert result[0][2] == "tpl0"


def test_transformer_top_k_larger_than_candidate_pool(monkeypatch):
    monkeypatch.setattr(tform, "process_pair_for_template",
                        lambda r, l: _fake_candidates(2))
    monkeypatch.setattr(tform, "TOP_K_REFINE", 5)

    result = transformer(["rec"], ["lig"])

    # Only 2 candidates exist; asking for 5 must not crash or duplicate.
    assert len(result) == 2


# ---------------------------------------------------------------------------
# F9: exclude self-templates (added 2026-08-10)
# ---------------------------------------------------------------------------

def test_process_pair_excludes_self_template(monkeypatch, tmp_path):
    import pandas as pd
    monkeypatch.setattr(tform, "ALIGNMENT_DIR", str(tmp_path))

    # receptor/ligand pdb id is "1acb". One shared template is "1acbxy" --
    # the target's own structure -- and must be excluded; "9xyzab" is a
    # genuinely different structure and must survive.
    rec_df = pd.DataFrame([
        {"template": "1acbxy", "chain": "X", "tm_score": 0.9},
        {"template": "9xyzab", "chain": "A", "tm_score": 0.8},
    ])
    lig_df = pd.DataFrame([
        {"template": "1acbxy", "chain": "Y", "tm_score": 0.9},
        {"template": "9xyzab", "chain": "B", "tm_score": 0.8},
    ])
    rec_df.to_csv(tmp_path / "1acbE.csv", index=False)
    lig_df.to_csv(tmp_path / "1acbI.csv", index=False)

    seen_templates = []

    def fake_evaluate(receptor, ligand, template, rec, lig, stats=None):
        seen_templates.append(template)
        return {"template": template, "rec_chain": rec["chain"], "lig_chain": lig["chain"],
                "score": rec["tm_score"] + lig["tm_score"], "contacts": 30,
                "output_pdb": f"{template}.pdb"}

    monkeypatch.setattr(tform, "_evaluate", fake_evaluate)

    candidates = tform.process_pair_for_template("1acbE", "1acbI")

    assert "1acbxy" not in seen_templates
    assert "9xyzab" in seen_templates
    assert len(candidates) == 1
    assert candidates[0]["template"] == "9xyzab"


def test_process_pair_self_template_only_yields_nothing(monkeypatch, tmp_path):
    import pandas as pd
    monkeypatch.setattr(tform, "ALIGNMENT_DIR", str(tmp_path))

    rec_df = pd.DataFrame([{"template": "7ceixy", "chain": "X", "tm_score": 1.0}])
    lig_df = pd.DataFrame([{"template": "7ceixy", "chain": "Y", "tm_score": 1.0}])
    rec_df.to_csv(tmp_path / "7ceiA.csv", index=False)
    lig_df.to_csv(tmp_path / "7ceiB.csv", index=False)

    monkeypatch.setattr(tform, "_evaluate", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("_evaluate should never be called for a self-template")))

    candidates = tform.process_pair_for_template("7ceiA", "7ceiB")
    assert candidates == []


# ---------------------------------------------------------------------------
# F11 post-transform interface contact check (added 2026-08-07)
# ---------------------------------------------------------------------------

from src.transformation import (  # noqa: E402
    count_interface_contacts,
    pair_has_sufficient_contact,
    _read_protein_heavy_atoms,
)


def _atom(serial, name, resname, chain, resseq, x, y, z, record="ATOM", element=None, altloc=" "):
    element = element if element is not None else name.strip()[0]
    return (
        f"{record:<6s}{serial:5d} {name:<4s}{altloc}{resname:>3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}  \n"
    )


def test_contact_count_touching_and_separated(tmp_path):
    r = tmp_path / "r.pdb"
    l = tmp_path / "l.pdb"
    # Two receptor residues, two ligand residues, all within 4 A -> 4 pairs.
    r.write_text(
        _atom(1, " CA ", "ALA", "A", 1, 0.0, 0.0, 0.0)
        + _atom(2, " CA ", "ALA", "A", 2, 0.0, 0.0, 1.0)
    )
    l.write_text(
        _atom(1, " CA ", "ALA", "B", 1, 3.0, 0.0, 0.0)
        + _atom(2, " CA ", "ALA", "B", 2, 3.0, 0.0, 1.0)
    )
    assert count_interface_contacts(str(r), str(l)) == 4

    # Same chains pushed 40 A apart -> no contact at all. This is the case the
    # clash check let through before the fix.
    l.write_text(
        _atom(1, " CA ", "ALA", "B", 1, 43.0, 0.0, 0.0)
        + _atom(2, " CA ", "ALA", "B", 2, 43.0, 0.0, 1.0)
    )
    assert count_interface_contacts(str(r), str(l)) == 0
    assert pair_has_sufficient_contact(str(r), str(l), minimum=1) is False
    # minimum <= 0 restores the pre-fix behaviour
    assert pair_has_sufficient_contact(str(r), str(l), minimum=0) is True


def test_contact_count_is_per_residue_not_per_atom(tmp_path):
    """Many atoms of one residue in range must still count as one contact."""
    r = tmp_path / "r.pdb"
    l = tmp_path / "l.pdb"
    r.write_text(
        "".join(
            _atom(i, f" C{i} ", "LEU", "A", 1, 0.0, 0.0, float(i) * 0.1)
            for i in range(1, 6)
        )
    )
    l.write_text(
        "".join(
            _atom(i, f" C{i} ", "LEU", "B", 1, 3.0, 0.0, float(i) * 0.1)
            for i in range(1, 6)
        )
    )
    assert count_interface_contacts(str(r), str(l)) == 1


def test_contact_count_boundary(tmp_path):
    r = tmp_path / "r.pdb"
    l = tmp_path / "l.pdb"
    r.write_text(_atom(1, " CA ", "ALA", "A", 1, 0.0, 0.0, 0.0))
    l.write_text(_atom(1, " CA ", "ALA", "B", 1, 4.999, 0.0, 0.0))
    assert count_interface_contacts(str(r), str(l)) == 1
    l.write_text(_atom(1, " CA ", "ALA", "B", 1, 5.001, 0.0, 0.0))
    assert count_interface_contacts(str(r), str(l)) == 0


def test_waters_and_hydrogens_are_not_counted(tmp_path):
    """A water bridging the gap must not create a contact (the P16 mistake)."""
    r = tmp_path / "r.pdb"
    l = tmp_path / "l.pdb"
    r.write_text(
        _atom(1, " CA ", "ALA", "A", 1, 0.0, 0.0, 0.0)
        + _atom(2, " O  ", "HOH", "A", 900, 4.0, 0.0, 0.0, record="HETATM", element="O")
        + _atom(3, " H  ", "ALA", "A", 1, 4.5, 0.0, 0.0, element="H")
    )
    l.write_text(
        _atom(1, " O  ", "HOH", "B", 901, 5.0, 0.0, 0.0, record="HETATM", element="O")
        + _atom(2, " H  ", "ALA", "B", 1, 4.8, 0.0, 0.0, element="H")
        + _atom(3, " CA ", "ALA", "B", 1, 9.0, 0.0, 0.0)
    )
    # Only real heavy protein atoms remain, and they are 9 A apart.
    assert count_interface_contacts(str(r), str(l)) == 0

    # MSE is a real residue and must survive the filter.
    l.write_text(_atom(1, "SE  ", "MSE", "B", 1, 4.0, 0.0, 0.0, record="HETATM", element="SE"))
    assert count_interface_contacts(str(r), str(l)) == 1


def test_altloc_b_is_skipped(tmp_path):
    r = tmp_path / "r.pdb"
    l = tmp_path / "l.pdb"
    r.write_text(_atom(1, " CA ", "ALA", "A", 1, 0.0, 0.0, 0.0))
    l.write_text(_atom(1, " CA ", "ALA", "B", 1, 4.0, 0.0, 0.0, altloc="B"))
    assert _read_protein_heavy_atoms(str(l)) == []
    assert count_interface_contacts(str(r), str(l)) == 0


def test_missing_file_returns_zero(tmp_path):
    r = tmp_path / "r.pdb"
    r.write_text(_atom(1, " CA ", "ALA", "A", 1, 0.0, 0.0, 0.0))
    assert count_interface_contacts(str(r), str(tmp_path / "nope.pdb")) == 0


# ---------------------------------------------------------------------------
# Kabsch interface superposition (2026-08-11)
# ---------------------------------------------------------------------------

def test_kabsch_identical_gives_zero_rmsd():
    import numpy as np
    P = np.random.default_rng(1).normal(size=(12, 3)) * 8
    R, t, rmsd = tform.kabsch_superpose(P, P.copy())
    assert rmsd == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(R @ P.T + t.reshape(3, 1), P.T, atol=1e-8)


def test_kabsch_recovers_known_rigid_transform():
    import numpy as np
    rng = np.random.default_rng(2)
    P = rng.normal(size=(20, 3)) * 10
    th = 0.9
    R_true = np.array([[np.cos(th), -np.sin(th), 0.0],
                       [np.sin(th), np.cos(th), 0.0],
                       [0.0, 0.0, 1.0]])
    t_true = np.array([4.0, -2.0, 7.0])
    Q = (R_true @ P.T).T + t_true
    R, t, rmsd = tform.kabsch_superpose(P, Q)
    assert rmsd == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(R, R_true, atol=1e-8)
    assert np.allclose(t, t_true, atol=1e-8)
    # and the convention must match apply_tm_transform: new = R @ old + t
    assert np.allclose((R @ P.T).T + t, Q, atol=1e-8)


def test_kabsch_rejects_reflection():
    """A mirrored copy must NOT superpose to zero -- det(R) has to stay +1,
    otherwise we would silently accept a mirror-image structure."""
    import numpy as np
    rng = np.random.default_rng(3)
    P = rng.normal(size=(15, 3)) * 6
    M = P.copy()
    M[:, 2] *= -1.0
    R, _t, rmsd = tform.kabsch_superpose(P, M)
    assert rmsd > 1.0
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_interface_residues_by_side_splits_contact_list():
    a, b = tform._interface_residues_by_side("xxxxAB",
                                                      [[1, 10], [2, 11], [2, 12]])
    assert a == {1, 2}
    assert b == {10, 11, 12}


def test_interface_residues_by_side_ignores_malformed_pairs():
    a, b = tform._interface_residues_by_side("xxxxAB",
                                                      [[1, 10], [3], ["x", "y"], [2, 11]])
    assert a == {1, 2}
    assert b == {10, 11}


def test_kabsch_transform_returns_none_when_too_few_pairs(tmp_path, monkeypatch):
    """Below KABSCH_MIN_PAIRS the caller must fall back to TM-align."""
    monkeypatch.setattr(tform, "INTERFACES_DIR", str(tmp_path / "iface"))
    monkeypatch.setattr(tform, "SURFACE_DIR", str(tmp_path / "surf"))
    (tmp_path / "iface").mkdir()
    (tmp_path / "surf").mkdir()
    tform._CA_CACHE.clear()

    def ca(i, ch, num, x, y, z):
        return (f"ATOM  {i:5d}  CA  ALA {ch}{num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")

    (tmp_path / "iface" / "ttttAB_A_int.pdb").write_text(
        "".join(ca(i, "A", i, i * 3.8, 0, 0) for i in range(1, 4)) + "END\n")
    (tmp_path / "surf" / "zzzzE.asa.pdb").write_text(
        "".join(ca(i, "E", i, i * 3.8, 0, 0) for i in range(1, 4)) + "END\n")
    md = {f"A.A.{i}": f"E.A.{i}" for i in range(1, 4)}   # only 3 pairs, min is 5
    assert tform._kabsch_transform("zzzzE", "ttttAB", "A", md, {1, 2, 3}) is None
    tform._CA_CACHE.clear()


def test_kabsch_transform_restricts_to_interface_residues(tmp_path, monkeypatch):
    """Residues outside the template interface set must not influence the fit.

    Build a case where the interface residues superpose perfectly but the
    non-interface residues are badly displaced: restricting to the interface
    must still give RMSD ~ 0.
    """
    monkeypatch.setattr(tform, "INTERFACES_DIR", str(tmp_path / "iface"))
    monkeypatch.setattr(tform, "SURFACE_DIR", str(tmp_path / "surf"))
    (tmp_path / "iface").mkdir()
    (tmp_path / "surf").mkdir()
    tform._CA_CACHE.clear()

    def ca(i, ch, num, x, y, z):
        return (f"ATOM  {i:5d}  CA  ALA {ch}{num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")

    iface_res = list(range(1, 7))      # these agree exactly
    junk_res = list(range(7, 11))      # these are wildly off in the target
    tmpl = "".join(ca(i, "A", i, i * 3.8, 0, 0) for i in iface_res + junk_res) + "END\n"
    tgt = ("".join(ca(i, "E", i, i * 3.8, 0, 0) for i in iface_res)
           + "".join(ca(i, "E", i, i * 3.8, 40.0, 25.0) for i in junk_res) + "END\n")
    (tmp_path / "iface" / "ttttAB_A_int.pdb").write_text(tmpl)
    (tmp_path / "surf" / "zzzzE.asa.pdb").write_text(tgt)
    md = {f"A.A.{i}": f"E.A.{i}" for i in iface_res + junk_res}

    got = tform._kabsch_transform("zzzzE", "ttttAB", "A", md, set(iface_res))
    assert got is not None
    _t, _R, rmsd, n = got
    assert n == len(iface_res)
    assert rmsd == pytest.approx(0.0, abs=1e-6)

    # with NO interface restriction the junk residues drag the fit
    got_all = tform._kabsch_transform("zzzzE", "ttttAB", "A", md, set())
    assert got_all is not None
    assert got_all[2] > 1.0
    tform._CA_CACHE.clear()


# ---------------------------------------------------------------------------
# Candidate selection for refinement (union ranking, 2026-08-11)
# ---------------------------------------------------------------------------

def _cand(template, score, contacts, rec="A", lig="B"):
    return {"template": template, "score": score, "contacts": contacts,
            "rec_chain": rec, "lig_chain": lig, "output_pdb": f"{template}.pdb"}


def test_select_default_mode_is_unchanged_top_k(monkeypatch):
    monkeypatch.setattr(tform, "RANK_MODE", "tm_only")
    monkeypatch.setattr(tform, "TOP_K_REFINE", 3)
    cands = [_cand(f"t{i}", 1.0 - i * 0.1, i) for i in range(10)]
    picked = tform.select_for_refinement(cands)
    assert [c["template"] for c in picked] == ["t0", "t1", "t2"]


def test_select_union_captures_high_contact_candidate_that_scores_badly(monkeypatch):
    """The acb case: the good candidate is near-last by alignment score but
    top by contact count. tm_only must miss it; the union must catch it."""
    monkeypatch.setattr(tform, "TOP_K_REFINE", 3)
    cands = [_cand(f"t{i}", 1.0 - i * 0.01, 5) for i in range(50)]
    good = _cand("GOOD", 0.20, 999)          # worst score, best contacts
    cands.append(good)
    cands.sort(key=lambda c: c["score"], reverse=True)

    monkeypatch.setattr(tform, "RANK_MODE", "tm_only")
    assert "GOOD" not in [c["template"] for c in tform.select_for_refinement(cands)]

    monkeypatch.setattr(tform, "RANK_MODE", "union_tm_contacts")
    picked = [c["template"] for c in tform.select_for_refinement(cands)]
    assert "GOOD" in picked
    assert len(picked) <= 6


def test_select_union_keeps_score_leaders_too(monkeypatch):
    """brs/pcc regression: the union must not displace a candidate that is
    already top by alignment score."""
    monkeypatch.setattr(tform, "RANK_MODE", "union_tm_contacts")
    monkeypatch.setattr(tform, "TOP_K_REFINE", 3)
    best_by_score = _cand("BEST", 2.0, 1)
    cands = [best_by_score] + [_cand(f"t{i}", 0.5, 100 + i) for i in range(20)]
    picked = [c["template"] for c in tform.select_for_refinement(cands)]
    assert "BEST" in picked
    assert picked[0] == "BEST"          # score picks come first


def test_select_union_deduplicates(monkeypatch):
    """A candidate that is top by BOTH criteria must appear once, not twice."""
    monkeypatch.setattr(tform, "RANK_MODE", "union_tm_contacts")
    monkeypatch.setattr(tform, "TOP_K_REFINE", 3)
    both = _cand("BOTH", 9.0, 9999)
    cands = [both] + [_cand(f"t{i}", 1.0 - i * 0.1, i) for i in range(10)]
    picked = tform.select_for_refinement(cands)
    names = [c["template"] for c in picked]
    assert names.count("BOTH") == 1
    assert len(names) == len(set(names))


def test_select_union_distinguishes_same_template_different_chains(monkeypatch):
    """Dedup key must include the chain assignment -- the same template used in
    both orientations is two different candidates."""
    monkeypatch.setattr(tform, "RANK_MODE", "union_tm_contacts")
    monkeypatch.setattr(tform, "TOP_K_REFINE", 2)
    cands = [_cand("X", 1.0, 10, rec="A", lig="B"),
             _cand("X", 0.9, 900, rec="B", lig="A")]
    picked = tform.select_for_refinement(cands)
    assert len(picked) == 2


def test_select_handles_empty_and_zero_k(monkeypatch):
    monkeypatch.setattr(tform, "RANK_MODE", "union_tm_contacts")
    assert tform.select_for_refinement([]) == []
    assert tform.select_for_refinement([_cand("a", 1.0, 1)], k=0) == []


def test_select_union_never_returns_more_than_2k(monkeypatch):
    monkeypatch.setattr(tform, "RANK_MODE", "union_tm_contacts")
    monkeypatch.setattr(tform, "TOP_K_REFINE", 4)
    cands = [_cand(f"t{i}", i * 0.1, (50 - i)) for i in range(40)]
    assert len(tform.select_for_refinement(cands)) <= 8

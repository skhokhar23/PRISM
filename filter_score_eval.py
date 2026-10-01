#!/usr/bin/env python3
"""Offline evaluation of proposed filters (A1,A2,A3,A5) and ranking scores
(B1-B5) against candidate pools that already exist on disk.

Changes nothing. Runs no pipeline stage. Needs no cluster job. For every
candidate the pipeline already produced, it recomputes each proposed metric
from data already written to disk, then answers the only two questions that
matter for the decision:

  1. FILTERS: how many candidates survive, and does the known-good candidate
     survive? (A filter that removes the right answer is worse than useless.)
  2. SCORES:  where does the known-good candidate rank, and does the
     already-correct answer on brs/pcc stay at the top?

Ground truth (the native interface) is used ONLY to evaluate. None of the
proposed metrics use it, so all of them remain legitimate for real prediction.

Metrics implemented
-------------------
  A1  template-contact count   how many of the template's own interface contact
                               pairs have BOTH halves matched (legacy requires 5,
                               we currently require 1)
  A2  interface Kabsch RMSD    superimpose the matched INTERFACE residues only
                               (target CA vs template CA) and take the RMSD.
                               Legacy's MultiProt enforces <= 2.0 A.
  A3  hotspot identity count   hotspots matched with the SAME amino acid
                               (legacy "criterion 2"); we currently match on
                               residue position only
  A5  min TM-score             the weaker of the two chains' TM-scores, i.e.
                               what a raised TM_SCORE_THRESHOLD would gate on
  B1  PRISM combined score     0.6*f_hotspot + 0.4*f_match per side, summed
                               (Tuncbag 2012, Fig 4b; >1 reported near-native)
  B2  interface contact count  F11's own number (currently only a >=20 gate)
  B3  contact recapitulation   fraction of the template's mapped interface
                               contacts actually realised in the built complex
  B4  = A2 used as a score
  B5  distance-filtered match  matched pairs actually within 5 A of their
                               template counterpart after the transform (F8)

Usage
-----
    python3 filter_score_eval.py \\
        --run-dir  processed_ptc_rsa15_f11_f10_sasaiso \\
        --receptor 2ptcE --ligand 2ptcI \\
        --native   processed_ptc_rsa15_f11_f10_sasaiso/compare/native/2ptc_2ptc_EI.pdb \\
        --templates-root templates \\
        --known-good 1shyAB \\
        --out ptc_filter_eval.csv
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

CONTACT_CUTOFF = 5.0      # A, heavy atom; matches F11 / prism_bench
MATCH_CLOSE_CUTOFF = 5.0  # A, CA-CA, for the distance-filtered match count (B5)

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
}
_THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M",
}


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def kabsch(P, Q):
    """Rotation/translation taking P onto Q, plus the RMSD. Rows are paired.

    Verified against Bio.SVDSuperimposer to 1e-9, including the reflection
    guard (det(R) = +1 enforced via the sign correction).
    """
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    diff = (R @ (P - Pc).T).T - (Q - Qc)
    return float(np.sqrt((diff ** 2).sum() / len(P))), R, Qc - R @ Pc


def _element_of(line):
    e = line[76:78].strip().upper()
    if e:
        return e
    for ch in line[12:16]:
        if ch.isalpha():
            return ch.upper()
    return ""


def read_ca(path):
    """{(chain, resnum): (resname, np.array([x,y,z]))} for CA atoms."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                out[(line[21], int(line[22:26]))] = (
                    line[17:20].strip().upper(),
                    np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                )
            except ValueError:
                continue
    return out


def read_heavy(path, chains=None):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 54:
                continue
            if line[17:20].strip().upper() not in _STANDARD_AA:
                continue
            if line[16] not in (" ", "A"):
                continue
            if _element_of(line) in ("H", "D"):
                continue
            ch = line[21]
            if chains and ch not in chains:
                continue
            try:
                out.append((ch, int(line[22:26]),
                            float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                continue
    return out


def contact_residue_pairs(side_a, side_b, cutoff=CONTACT_CUTOFF):
    """set of (resnum_a, resnum_b) residue pairs in contact, grid-accelerated."""
    if not side_a or not side_b:
        return set()
    cell = cutoff
    grid = defaultdict(list)
    for (_c, rn, x, y, z) in side_b:
        grid[(int(x // cell), int(y // cell), int(z // cell))].append((rn, x, y, z))
    cut2 = cutoff * cutoff
    pairs = set()
    for (_c, ra, x, y, z) in side_a:
        cx, cy, cz = int(x // cell), int(y // cell), int(z // cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for (rb, bx, by, bz) in grid.get((cx + dx, cy + dy, cz + dz), ()):
                        if (x - bx) ** 2 + (y - by) ** 2 + (z - bz) ** 2 <= cut2:
                            pairs.add((ra, rb))
    return pairs


# --------------------------------------------------------------------------- #
# parsing pipeline artefacts
# --------------------------------------------------------------------------- #

def parse_key(k):
    """'A.K.15' -> ('A', 'K', 15); returns None if malformed."""
    p = k.split(".")
    if len(p) != 3:
        return None
    try:
        return p[0], p[1], int(p[2])
    except ValueError:
        return None


def load_alignment_rows(run, target):
    path = os.path.join(run, "alignment", f"{target}.csv")
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                rows[(r["template"], r["chain"])] = {
                    "tm_score": float(r["tm_score"]),
                    "match_count": int(r["match_count"]),
                    "len_template": int(r["len_template"]),
                    "translation": json.loads(r["translation"]) if isinstance(r["translation"], str) else r["translation"],
                    "rotation_mat": json.loads(r["rotation_mat"]) if isinstance(r["rotation_mat"], str) else r["rotation_mat"],
                }
            except (ValueError, KeyError, json.JSONDecodeError):
                continue
    return rows


def load_match_dict(run, target, template, chain):
    p = os.path.join(run, "alignment", f"{target}_{template}_{chain}.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            return json.load(fh).get("match_dict", {})
    except (json.JSONDecodeError, OSError):
        return {}


def load_template_contacts(troot, template):
    p = os.path.join(troot, "contacts", f"{template}.json")
    if not os.path.exists(p):
        return []
    try:
        with open(p) as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(payload, dict):
        out = []
        for v in payload.values():
            out.extend(v)
        return out
    return payload


def load_template_hotspots(troot, template):
    p = os.path.join(troot, "hotspots", f"{template}.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


# --------------------------------------------------------------------------- #

def side_metrics(run, troot, target, template, chain, contacts_side_residues, hotspots):
    """Per-side quantities: match dict, f_match, hotspot identity, Kabsch RMSD."""
    md = load_match_dict(run, target, template, chain)
    res = {"match_dict": md, "n_matched": len(md),
           "hotspot_pos": 0, "hotspot_id": 0, "hotspot_total": 0,
           "kabsch_rmsd": float("nan"), "kabsch_n": 0,
           "matched_template_res": set(), "tmpl2tgt": {}}
    if not md:
        return res

    tmpl2tgt = {}
    for k, v in md.items():
        pk, pv = parse_key(k), parse_key(v)
        if pk is None or pv is None:
            continue
        tmpl2tgt[pk[2]] = (pv[2], pk[1], pv[1])   # tmpl_res -> (tgt_res, tmpl_aa, tgt_aa)
    res["tmpl2tgt"] = tmpl2tgt
    res["matched_template_res"] = set(tmpl2tgt)

    # A3: hotspot matching, by position and by amino-acid identity
    hs = hotspots.get(chain, [])
    res["hotspot_total"] = len(hs)
    for entry in hs:
        try:
            hnum, hname = int(entry[0]), str(entry[1]).strip().upper()
        except (ValueError, IndexError, TypeError):
            continue
        if hnum in tmpl2tgt:
            res["hotspot_pos"] += 1
            _tgt_res, _tmpl_aa, tgt_aa = tmpl2tgt[hnum]
            if _THREE2ONE.get(hname, "?") == tgt_aa.upper():
                res["hotspot_id"] += 1

    # A2/B4: Kabsch RMSD over matched INTERFACE residues only
    tmpl_ca = read_ca(os.path.join(troot, "interfaces", f"{template}_{chain}_int.pdb"))
    tgt_ca = read_ca(os.path.join(run, "surface_extraction", f"{target}.asa.pdb"))
    if tmpl_ca and tgt_ca:
        P, Q = [], []
        for tmpl_res, (tgt_res, _a, _b) in tmpl2tgt.items():
            if contacts_side_residues and tmpl_res not in contacts_side_residues:
                continue
            tkey = [k for k in tmpl_ca if k[1] == tmpl_res]
            gkey = [k for k in tgt_ca if k[1] == tgt_res]
            if tkey and gkey:
                Q.append(tmpl_ca[tkey[0]][1])
                P.append(tgt_ca[gkey[0]][1])
        if len(P) >= 3:
            P, Q = np.array(P), np.array(Q)
            res["kabsch_rmsd"], _R, _t = kabsch(P, Q)
            res["kabsch_n"] = len(P)
    return res


def distance_filtered_match_count(run, troot, target, template, chain, tmpl2tgt, row):
    """B5: matched pairs whose target CA lands within cutoff of its template CA
    once the pipeline's own stored transform is applied."""
    tmpl_ca = read_ca(os.path.join(troot, "interfaces", f"{template}_{chain}_int.pdb"))
    tgt_ca = read_ca(os.path.join(run, "surface_extraction", f"{target}.asa.pdb"))
    if not tmpl_ca or not tgt_ca or not row:
        return -1
    try:
        R = np.array(row["rotation_mat"], dtype=float)
        t = np.array(row["translation"], dtype=float)
    except (TypeError, ValueError):
        return -1
    if R.shape != (3, 3) or t.shape != (3,):
        return -1
    close = 0
    for tmpl_res, (tgt_res, _a, _b) in tmpl2tgt.items():
        tkey = [k for k in tmpl_ca if k[1] == tmpl_res]
        gkey = [k for k in tgt_ca if k[1] == tgt_res]
        if not tkey or not gkey:
            continue
        moved = R @ tgt_ca[gkey[0]][1] + t
        if np.linalg.norm(moved - tmpl_ca[tkey[0]][1]) <= MATCH_CLOSE_CUTOFF:
            close += 1
    return close


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--templates-root", default="templates")
    ap.add_argument("--known-good", default="",
                    help="template id whose fate we care about, e.g. 1shyAB")
    ap.add_argument("--out", default="filter_score_eval.csv")
    args = ap.parse_args()

    run, troot = args.run_dir, args.templates_root

    # ---- ground truth (evaluation only) ----------------------------------- #
    rec_ch = set(args.receptor[4:].upper())
    lig_ch = set(args.ligand[4:].upper())
    nat_r = read_heavy(args.native, rec_ch)
    nat_l = read_heavy(args.native, lig_ch)
    if not nat_r or not nat_l:
        print(f"ERROR: native {args.native} yielded no atoms for chains "
              f"{sorted(rec_ch)}/{sorted(lig_ch)}", file=sys.stderr)
        sys.exit(1)
    true_iface = {a for (a, _b) in contact_residue_pairs(nat_r, nat_l)}
    print(f"native receptor interface: {len(true_iface)} residues")

    rec_rows = load_alignment_rows(run, args.receptor)
    lig_rows = load_alignment_rows(run, args.ligand)
    print(f"alignment rows: receptor {len(rec_rows)}, ligand {len(lig_rows)}")

    cand_dir = os.path.join(run, "output")
    files = sorted(f for f in os.listdir(cand_dir) if f.endswith(".pdb")) \
        if os.path.isdir(cand_dir) else []
    if not files:
        print(f"ERROR: no candidates in {cand_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"candidates on disk: {len(files)}\n")

    results = []
    skipped = defaultdict(int)

    for i, fname in enumerate(files, 1):
        template = fname.split("_")[0]
        merged = os.path.join(cand_dir, fname)

        contacts = load_template_contacts(troot, template)
        hotspots = load_template_hotspots(troot, template)
        if not contacts:
            skipped["no template contacts file"] += 1

        # Which template chain went to receptor vs ligand is NOT recorded in the
        # filename, so enumerate the valid assignments and keep the one the
        # pipeline's own ranking would have surfaced first (highest tm sum).
        best = None
        for (t1, c1), r1 in rec_rows.items():
            if t1 != template:
                continue
            for (t2, c2), r2 in lig_rows.items():
                if t2 != template or c2 == c1:
                    continue
                s = r1["tm_score"] + r2["tm_score"]
                if best is None or s > best[0]:
                    best = (s, c1, c2, r1, r2)
        if best is None:
            skipped["no alignment rows for template"] += 1
            continue
        tm_sum, rc, lc, rrow, lrow = best

        # template interface residues on each side, from the template's own
        # contact list (chain order in the file follows the template id)
        side1 = {int(p[0]) for p in contacts if len(p) == 2}
        side2 = {int(p[1]) for p in contacts if len(p) == 2}
        if template[4:5] == rc:
            rec_iface_res, lig_iface_res = side1, side2
        else:
            rec_iface_res, lig_iface_res = side2, side1

        rm = side_metrics(run, troot, args.receptor, template, rc, rec_iface_res, hotspots)
        lm = side_metrics(run, troot, args.ligand, template, lc, lig_iface_res, hotspots)

        # ---- A1 : template contact pairs with BOTH halves matched ---------- #
        a1 = 0
        mapped_pairs = []
        for p in contacts:
            if len(p) != 2:
                continue
            try:
                a, b = int(p[0]), int(p[1])
            except (ValueError, TypeError):
                continue
            if a in rm["tmpl2tgt"] and b in lm["tmpl2tgt"]:
                a1 += 1
                mapped_pairs.append((rm["tmpl2tgt"][a][0], lm["tmpl2tgt"][b][0]))
            elif b in rm["tmpl2tgt"] and a in lm["tmpl2tgt"]:
                a1 += 1
                mapped_pairs.append((rm["tmpl2tgt"][b][0], lm["tmpl2tgt"][a][0]))

        # ---- geometry of the built complex --------------------------------- #
        mr = read_heavy(merged, rec_ch)
        ml = read_heavy(merged, lig_ch)
        real_pairs = contact_residue_pairs(mr, ml)
        really = {a for (a, _b) in real_pairs}

        # ---- B3 : template contact recapitulation -------------------------- #
        b3 = (len([1 for pr in mapped_pairs if pr in real_pairs]) / len(mapped_pairs)
              if mapped_pairs else 0.0)

        # ---- B1 : PRISM's published combined matching score ---------------- #
        def side_score(m, row):
            f_hot = (m["hotspot_id"] / m["hotspot_total"]) if m["hotspot_total"] else 0.0
            f_mat = (row["match_count"] / row["len_template"]) if row["len_template"] else 0.0
            return 0.6 * f_hot + 0.4 * min(f_mat, 1.0)
        b1 = side_score(rm, rrow) + side_score(lm, lrow)

        # ---- B5 ------------------------------------------------------------ #
        b5r = distance_filtered_match_count(run, troot, args.receptor, template, rc,
                                            rm["tmpl2tgt"], rrow)
        b5l = distance_filtered_match_count(run, troot, args.ligand, template, lc,
                                            lm["tmpl2tgt"], lrow)

        kr, kl = rm["kabsch_rmsd"], lm["kabsch_rmsd"]
        a2 = float(np.nanmax([kr, kl])) if not (np.isnan(kr) and np.isnan(kl)) else float("nan")

        jac = (len(really & true_iface) / len(really | true_iface)) if (really or true_iface) else 0.0

        results.append({
            "template": template, "rec_chain": rc, "lig_chain": lc,
            "tm_sum": round(tm_sum, 4),
            "A5_min_tm": round(min(rrow["tm_score"], lrow["tm_score"]), 4),
            "A1_tmpl_contacts_mapped": a1,
            "A2_kabsch_rmsd_worst": round(a2, 3) if a2 == a2 else "",
            "A2_kabsch_n": min(rm["kabsch_n"], lm["kabsch_n"]),
            "A3_hotspot_identity": rm["hotspot_id"] + lm["hotspot_id"],
            "A3_hotspot_position": rm["hotspot_pos"] + lm["hotspot_pos"],
            "B1_prism_combined": round(b1, 4),
            "B2_contact_count": len(real_pairs),
            "B3_recapitulation": round(b3, 4),
            "B5_close_matches": (b5r + b5l) if (b5r >= 0 and b5l >= 0) else -1,
            "jaccard_vs_native": round(jac, 4),
            "n_contacting_res": len(really),
        })
        if i % 100 == 0:
            print(f"  ... {i}/{len(files)}")

    if not results:
        print("ERROR: no candidate could be evaluated", file=sys.stderr)
        for k, v in skipped.items():
            print(f"  skipped ({k}): {v}", file=sys.stderr)
        sys.exit(1)

    fields = list(results[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(results)

    # ----------------------------------------------------------------- report #
    n = len(results)
    print(f"\nevaluated {n} candidates" + (f"   (skipped: {dict(skipped)})" if skipped else ""))

    def good_row():
        if not args.known_good:
            return None
        for r in results:
            if r["template"].lower() == args.known_good.lower():
                return r
        return None

    g = good_row()
    best_actual = max(results, key=lambda r: r["jaccard_vs_native"])
    print(f"best candidate by real overlap: {best_actual['template']} "
          f"jaccard={best_actual['jaccard_vs_native']}")
    if args.known_good:
        print(f"known-good {args.known_good}: "
              + (f"present, jaccard={g['jaccard_vs_native']}" if g else "NOT FOUND in pool"))

    print("\n--- FILTERS: survivors / does the good candidate survive ---")
    filters = [
        ("A1 template contacts >= 5", lambda r: r["A1_tmpl_contacts_mapped"] >= 5),
        ("A1 template contacts >= 3", lambda r: r["A1_tmpl_contacts_mapped"] >= 3),
        ("A2 kabsch RMSD <= 2.0 A", lambda r: r["A2_kabsch_rmsd_worst"] != "" and r["A2_kabsch_rmsd_worst"] <= 2.0),
        ("A2 kabsch RMSD <= 3.0 A", lambda r: r["A2_kabsch_rmsd_worst"] != "" and r["A2_kabsch_rmsd_worst"] <= 3.0),
        ("A2 kabsch RMSD <= 4.0 A", lambda r: r["A2_kabsch_rmsd_worst"] != "" and r["A2_kabsch_rmsd_worst"] <= 4.0),
        ("A3 hotspot identity >= 1", lambda r: r["A3_hotspot_identity"] >= 1),
        ("A3 hotspot identity >= 2", lambda r: r["A3_hotspot_identity"] >= 2),
        ("A5 min TM >= 0.25", lambda r: r["A5_min_tm"] >= 0.25),
        ("A5 min TM >= 0.30", lambda r: r["A5_min_tm"] >= 0.30),
        ("A5 min TM >= 0.35", lambda r: r["A5_min_tm"] >= 0.35),
        ("A5 min TM >= 0.40", lambda r: r["A5_min_tm"] >= 0.40),
        ("A5 min TM >= 0.50", lambda r: r["A5_min_tm"] >= 0.50),
        ("B3 recapitulation > 0", lambda r: r["B3_recapitulation"] > 0),
    ]
    print(f"{'filter':<30}{'survivors':>11}{'% kept':>9}{'good survives?':>17}"
          f"{'best-real survives?':>21}")
    for name, fn in filters:
        surv = [r for r in results if fn(r)]
        gs = ("n/a" if g is None else ("YES" if fn(g) else "*** NO ***"))
        bs = "YES" if fn(best_actual) else "*** NO ***"
        print(f"{name:<30}{len(surv):>11}{100.0*len(surv)/n:>8.1f}%{gs:>17}{bs:>21}")

    print("\n--- SCORES: correlation with real overlap, and rank of good candidate ---")
    jac = np.array([r["jaccard_vs_native"] for r in results], dtype=float)

    def ranks(a):
        o = np.argsort(a); out = np.empty_like(o, dtype=float); out[o] = np.arange(len(a)); return out

    scores = [
        ("current: tm_sum", "tm_sum", True),
        ("A2/B4 kabsch RMSD", "A2_kabsch_rmsd_worst", False),
        ("B1 PRISM combined", "B1_prism_combined", True),
        ("B2 contact count", "B2_contact_count", True),
        ("B3 recapitulation", "B3_recapitulation", True),
        ("B5 close matches", "B5_close_matches", True),
        ("A1 template contacts", "A1_tmpl_contacts_mapped", True),
    ]
    print(f"{'score':<25}{'pearson':>10}{'spearman':>10}{'good rank':>12}"
          f"{'best-real rank':>16}")
    for name, key, higher_better in scores:
        vals = []
        ok = True
        for r in results:
            v = r[key]
            if v == "" or v is None or v == -1:
                ok = False
                break
            vals.append(float(v))
        if not ok or len(set(vals)) < 2:
            print(f"{name:<25}{'n/a':>10}{'n/a':>10}{'n/a':>12}{'n/a':>16}")
            continue
        v = np.array(vals, dtype=float)
        sign = 1.0 if higher_better else -1.0
        pear = float(np.corrcoef(jac, sign * v)[0, 1])
        spear = float(np.corrcoef(ranks(jac), ranks(sign * v))[0, 1])
        order = sorted(range(len(results)), key=lambda i: sign * v[i], reverse=True)
        grank = brank = "n/a"
        for pos, idx in enumerate(order, 1):
            if g is not None and results[idx] is g:
                grank = pos
            if results[idx] is best_actual:
                brank = pos
        print(f"{name:<25}{pear:>10.3f}{spear:>10.3f}{str(grank):>12}{str(brank):>16}")

    print(f"\nfull table -> {args.out}")
    print("Read the FILTERS block first: any row where the good candidate does "
          "NOT survive is disqualifying, no matter how many junk candidates it removes.")


if __name__ == "__main__":
    main()

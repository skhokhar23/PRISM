#!/usr/bin/env python3
"""Second-round offline evaluation: the strategies not covered by
filter_score_eval.py.

Three things get measured here, none of which need cluster time and none of
which use knowledge of the native answer (the native is used ONLY to score the
result afterwards):

  CONSENSUS      CONSRANK-style. Count how often each receptor-ligand residue
                 contact appears across the whole candidate ensemble, then
                 score each candidate by how well it reproduces the most
                 frequently observed contacts. Rationale for trying it here:
                 197/538 (ptc) and 125/600 (acb) candidates already touch the
                 real interface, so if a third of the pool independently
                 converges on the right patch while the rest scatter, the
                 conserved contacts ARE the interface.

  PRECISION      The other direction of the template-contact check. B3 in the
                 first round measured RECALL (of the contacts the template says
                 should form, how many did). This measures PRECISION (of the
                 contacts that actually formed, how many are between residues
                 the template considers interface residues at all). A candidate
                 that makes 30 contacts in completely the wrong place scores
                 high on neither, but only precision catches "contacts exist,
                 just nowhere near where they should be".

  FUSION         Combining rankings rather than inventing a new score. The
                 first round showed no single score works on all four pairs but
                 the failures are complementary: tm_sum ranks acb's good
                 candidate 451st while raw contact count ranks it 3rd, and vice
                 versa on brs/pcc. Union-of-top-K and reciprocal rank fusion
                 both exploit that without any weight tuning.

Reads the eval_<pair>.csv produced by filter_score_eval.py for the existing
per-candidate scores, and re-parses the candidate PDBs only for the two new
geometric measures.

Usage
-----
    python3 rank_fusion_eval.py \\
        --eval-csv eval_ptc.csv \\
        --run-dir  processed_ptc_rsa15_f11_f10_sasaiso \\
        --receptor 2ptcE --ligand 2ptcI \\
        --native   processed_ptc_rsa15_f11_f10_sasaiso/compare/native/2ptc_2ptc_EI.pdb \\
        --templates-root templates \\
        --known-good 1shyAB \\
        --out ptc_fusion.csv
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

CONTACT_CUTOFF = 5.0

_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
}


def _element_of(line):
    e = line[76:78].strip().upper()
    if e:
        return e
    for ch in line[12:16]:
        if ch.isalpha():
            return ch.upper()
    return ""


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


def contact_pairs(side_a, side_b, cutoff=CONTACT_CUTOFF):
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


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def template_interface_targets(run, target, template, chain, iface_res):
    """Target residue numbers that correspond to template INTERFACE residues."""
    md = load_json(os.path.join(run, "alignment", f"{target}_{template}_{chain}.json"),
                   {}).get("match_dict", {})
    out = set()
    for k, v in md.items():
        kp, vp = k.split("."), v.split(".")
        if len(kp) != 3 or len(vp) != 3:
            continue
        try:
            tmpl_res, tgt_res = int(kp[2]), int(vp[2])
        except ValueError:
            continue
        if not iface_res or tmpl_res in iface_res:
            out.add(tgt_res)
    return out


def ranks_desc(values):
    """1 = best (largest)."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    r = [0] * len(values)
    for pos, idx in enumerate(order, 1):
        r[idx] = pos
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--ligand", required=True)
    ap.add_argument("--native", required=True)
    ap.add_argument("--templates-root", default="templates")
    ap.add_argument("--known-good", default="")
    ap.add_argument("--out", default="fusion_eval.csv")
    args = ap.parse_args()

    run, troot = args.run_dir, args.templates_root
    rec_ch, lig_ch = set(args.receptor[4:].upper()), set(args.ligand[4:].upper())

    rows = list(csv.DictReader(open(args.eval_csv)))
    if not rows:
        print(f"ERROR: {args.eval_csv} is empty", file=sys.stderr)
        sys.exit(1)
    print(f"loaded {len(rows)} candidates from {args.eval_csv}")

    nat_r = read_heavy(args.native, rec_ch)
    nat_l = read_heavy(args.native, lig_ch)
    true_iface = {a for (a, _b) in contact_pairs(nat_r, nat_l)}
    print(f"native receptor interface: {len(true_iface)} residues")

    # ---- pass 1: contacts of every candidate, for consensus + precision ---- #
    print("parsing candidate geometry ...")
    cand_contacts, cand_precision = {}, {}
    for i, r in enumerate(rows, 1):
        tmpl, rc, lc = r["template"], r["rec_chain"], r["lig_chain"]
        merged = os.path.join(run, "output", f"{tmpl}_{args.receptor}_{args.ligand}.pdb")
        pairs = contact_pairs(read_heavy(merged, rec_ch), read_heavy(merged, lig_ch))
        cand_contacts[tmpl] = pairs

        contacts = load_json(os.path.join(troot, "contacts", f"{tmpl}.json"), [])
        a, b = set(), set()
        for p in contacts:
            if len(p) != 2:
                continue
            try:
                a.add(int(p[0])); b.add(int(p[1]))
            except (TypeError, ValueError):
                continue
        rec_iface, lig_iface = (a, b) if tmpl[4:5] == rc else (b, a)
        rec_tgt = template_interface_targets(run, args.receptor, tmpl, rc, rec_iface)
        lig_tgt = template_interface_targets(run, args.ligand, tmpl, lc, lig_iface)
        if pairs:
            good = sum(1 for (x, y) in pairs if x in rec_tgt and y in lig_tgt)
            cand_precision[tmpl] = good / len(pairs)
        else:
            cand_precision[tmpl] = 0.0
        if i % 150 == 0:
            print(f"  ... {i}/{len(rows)}")

    # ---- consensus: contact frequency across the ensemble ------------------ #
    freq = Counter()
    for pairs in cand_contacts.values():
        freq.update(pairs)
    n = len(cand_contacts)
    consensus = {}
    for tmpl, pairs in cand_contacts.items():
        consensus[tmpl] = (sum(freq[p] for p in pairs) / (len(pairs) * n)) if pairs else 0.0

    top_conserved = freq.most_common(15)
    hits = sum(1 for ((a, _b), _c) in top_conserved if a in true_iface)
    print(f"\n15 most conserved contacts across the ensemble: {hits}/15 involve a "
          f"REAL interface residue on the receptor")
    print("  (this is the single best indicator of whether consensus can work here)")

    # ---- assemble every score, compute ranks ------------------------------- #
    def col(key, cast=float):
        out = []
        for r in rows:
            try:
                out.append(cast(r[key]))
            except (ValueError, KeyError, TypeError):
                out.append(0.0)
        return out

    jac = col("jaccard_vs_native")
    scores = {
        "tm_sum": col("tm_sum"),
        "contact_count": col("B2_contact_count"),
        "close_matches": col("B5_close_matches"),
        "recapitulation": col("B3_recapitulation"),
        "prism_combined": col("B1_prism_combined"),
        "consensus": [consensus.get(r["template"], 0.0) for r in rows],
        "precision": [cand_precision.get(r["template"], 0.0) for r in rows],
    }
    rank = {k: ranks_desc(v) for k, v in scores.items()}

    # reciprocal rank fusion over the complementary pair, and over all
    K = 60.0
    scores["RRF_tm+contact"] = [
        1.0 / (K + rank["tm_sum"][i]) + 1.0 / (K + rank["contact_count"][i])
        for i in range(len(rows))]
    scores["RRF_all"] = [
        sum(1.0 / (K + rank[k][i]) for k in
            ("tm_sum", "contact_count", "close_matches", "consensus", "precision"))
        for i in range(len(rows))]
    rank["RRF_tm+contact"] = ranks_desc(scores["RRF_tm+contact"])
    rank["RRF_all"] = ranks_desc(scores["RRF_all"])

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["template", "jaccard"] + list(scores.keys()))
        for i, r in enumerate(rows):
            w.writerow([r["template"], jac[i]] + [round(scores[k][i], 6) for k in scores])

    gi = None
    if args.known_good:
        for i, r in enumerate(rows):
            if r["template"].lower() == args.known_good.lower():
                gi = i
                break
    bi = max(range(len(rows)), key=lambda i: jac[i])

    print(f"\nbest by real overlap: {rows[bi]['template']} (jaccard {jac[bi]:.3f})")
    if args.known_good:
        print(f"known-good {args.known_good}: "
              + (f"index {gi}, jaccard {jac[gi]:.3f}" if gi is not None else "NOT FOUND"))

    print(f"\n{'score':<22}{'pearson':>9}{'spearman':>10}{'good rank':>12}{'best rank':>12}")
    jarr = np.array(jac)

    def spearman(a, b):
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    for k, v in scores.items():
        varr = np.array(v, dtype=float)
        if varr.std() == 0 or jarr.std() == 0:
            print(f"{k:<22}{'n/a':>9}{'n/a':>10}{'n/a':>12}{'n/a':>12}")
            continue
        rk = rank[k] if k in rank else ranks_desc(v)
        print(f"{k:<22}{float(np.corrcoef(jarr, varr)[0,1]):>9.3f}"
              f"{spearman(jarr, varr):>10.3f}"
              f"{(rk[gi] if gi is not None else 0):>12}{rk[bi]:>12}")

    print(f"\n--- UNION strategies: is the good candidate captured? ---")
    print(f"{'strategy':<42}{'refined':>9}{'good captured?':>17}")
    combos = [
        ("top3 tm_sum", [("tm_sum", 3)]),
        ("top10 tm_sum (supervisor's setting)", [("tm_sum", 10)]),
        ("top3 tm_sum + top3 contact_count", [("tm_sum", 3), ("contact_count", 3)]),
        ("top5 tm_sum + top5 contact_count", [("tm_sum", 5), ("contact_count", 5)]),
        ("top3 tm_sum + top3 consensus", [("tm_sum", 3), ("consensus", 3)]),
        ("top3 tm_sum + top3 precision", [("tm_sum", 3), ("precision", 3)]),
        ("top3 each: tm_sum/contact/consensus",
         [("tm_sum", 3), ("contact_count", 3), ("consensus", 3)]),
        ("top5 each: tm_sum/contact/consensus/precision",
         [("tm_sum", 5), ("contact_count", 5), ("consensus", 5), ("precision", 5)]),
        ("top6 RRF_tm+contact", [("RRF_tm+contact", 6)]),
        ("top6 RRF_all", [("RRF_all", 6)]),
    ]
    for name, spec in combos:
        picked = set()
        for key, k in spec:
            rk = rank[key]
            picked |= {i for i in range(len(rows)) if rk[i] <= k}
        ok = "n/a" if gi is None else ("YES" if gi in picked else "*** NO ***")
        print(f"{name:<42}{len(picked):>9}{ok:>17}")

    print(f"\nfull table -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Groove-overlap diagnostic (2026-08-10).

Question this answers: for a pair like `acb` or `ptc`, where every refined
candidate has zero real contacts (see predictions_vs_native.csv), does ANY
candidate in the full pre-refine merged pool ever land on the receptor's
true interface at all -- or does nothing in the pool ever touch it?

This does not need PyRosetta and does not need a new pipeline run. It reads
the merged candidate PDBs the pipeline already wrote to disk (every merged
candidate is kept regardless of outcome -- see STAGE 3.6 in prism_bench.py's
report) and the native complex structure already used for scoring, and
checks each candidate's RECEPTOR-side residues against the true interface
using the same 5.0 A heavy-atom-contact definition used throughout this
project (CONTACT_CUTOFF in prism_bench.py, MIN_INTERFACE_CONTACTS's distance
cutoff in src/transformation.py).

Answer this gives:
  - if nothing in the full pool (hundreds to a thousand candidates) ever
    overlaps the true receptor interface, the problem is upstream of
    refinement and upstream of candidate selection -- it is at alignment.
    The correct template class is either absent from the library in usable
    form, or never survives the alignment stage's gates (F8 territory).
  - if some candidates DO overlap, even partially, but none of them made it
    into the top-K that got refined, the problem is the ranking rule itself
    (alignment score, F10) -- worth widening PRISM_TOP_K_REFINE or ranking
    by receptor-side interface overlap instead of alignment score.

Usage:
    python3 groove_diagnostic.py \\
        --native      PRISM/processed_.../compare/native/2ptc_2ptc_EI.pdb \\
        --receptor-chain E --ligand-chain I \\
        --candidates-dir PRISM/processed_ptc_rsa15_f11_f10_sasaiso/merged \\
        --out cmp_ptc_groove_diagnostic.csv

Before trusting the result on a new pair: confirm the chain IDs inside a
merged candidate file actually match --receptor-chain/--ligand-chain --
templates are transformed into the TARGET's frame, but relabeling
conventions can vary. Check with:
    python3 -c "from Bio.PDB import PDBParser; s=PDBParser(QUIET=True).get_structure('x','<file>'); print([c.id for c in s[0]])"
"""
import argparse
import csv
import glob
import os
import sys

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

CONTACT_CUTOFF = 5.0  # Angstrom, heavy atom -- matches prism_bench.py and F11


def _heavy_atoms_by_residue(chain):
    """residue_id -> list of heavy-atom coords, standard amino acids only."""
    out = {}
    for residue in chain:
        if not is_aa(residue, standard=True) and residue.get_resname() != "MSE":
            continue
        coords = [
            atom.coord for atom in residue
            if atom.element not in ("H", "D") and atom.get_altloc() in (" ", "A")
        ]
        if coords:
            out[residue.id[1]] = coords
    return out


def _contacting_residues(rec_atoms_by_res, lig_atoms_by_res, cutoff=CONTACT_CUTOFF):
    """Receptor residue numbers with >=1 heavy atom within cutoff of the ligand."""
    import numpy as np
    lig_all = None
    lig_lists = list(lig_atoms_by_res.values())
    if not lig_lists:
        return set()
    lig_all = np.concatenate([np.array(c) for c in lig_lists], axis=0)
    hits = set()
    cutoff2 = cutoff * cutoff
    for resnum, coords in rec_atoms_by_res.items():
        arr = np.array(coords)
        # broadcast distance-squared, bail out early per residue on first hit
        d2 = ((arr[:, None, :] - lig_all[None, :, :]) ** 2).sum(axis=2)
        if (d2 <= cutoff2).any():
            hits.add(resnum)
    return hits


def native_interface_residues(native_pdb, receptor_chain, ligand_chain):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("native", native_pdb)
    model = next(structure.get_models())
    if receptor_chain not in model or ligand_chain not in model:
        raise ValueError(
            f"native file {native_pdb} is missing chain "
            f"{receptor_chain!r} or {ligand_chain!r}; found {[c.id for c in model]}"
        )
    rec_atoms = _heavy_atoms_by_residue(model[receptor_chain])
    lig_atoms = _heavy_atoms_by_residue(model[ligand_chain])
    return _contacting_residues(rec_atoms, lig_atoms)


def candidate_receptor_residues(candidate_pdb, receptor_chain, ligand_chain):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("cand", candidate_pdb)
    except Exception as exc:
        return None, f"parse error: {exc}"
    model = next(structure.get_models())
    if receptor_chain not in model or ligand_chain not in model:
        return None, f"missing chain(s), found {[c.id for c in model]}"
    rec_atoms = _heavy_atoms_by_residue(model[receptor_chain])
    lig_atoms = _heavy_atoms_by_residue(model[ligand_chain])
    return _contacting_residues(rec_atoms, lig_atoms), None


def jaccard(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", required=True)
    ap.add_argument("--receptor-chain", required=True)
    ap.add_argument("--ligand-chain", required=True)
    ap.add_argument("--candidates-dir", required=True)
    ap.add_argument("--out", default="groove_diagnostic.csv")
    ap.add_argument("--limit", type=int, default=0,
                     help="0 = all candidates in the directory")
    ap.add_argument("--alignment-dir",
                     help="processed_.../alignment -- if given, reconstructs each "
                          "candidate's real alignment score (tm_score(receptor) + "
                          "tm_score(ligand), exactly as src/transformation.py "
                          "computes it) and ranks the whole pool by it, so you can "
                          "see exactly where a high-jaccard candidate actually sits "
                          "in the real ranking, not just whether it's in the top 10.")
    ap.add_argument("--receptor-target", help="e.g. 2ptcE -- required with --alignment-dir")
    ap.add_argument("--ligand-target", help="e.g. 2ptcI -- required with --alignment-dir")
    args = ap.parse_args()

    scores_by_template = {}
    if args.alignment_dir:
        import pandas as pd
        rec_csv = os.path.join(args.alignment_dir, f"{args.receptor_target}.csv")
        lig_csv = os.path.join(args.alignment_dir, f"{args.ligand_target}.csv")
        rec_df = pd.read_csv(rec_csv)
        lig_df = pd.read_csv(lig_csv)
        # Same score definition as src/transformation.py:375 --
        # tm_score(receptor) + tm_score(ligand), max over any chain pairing
        # sharing that template string (matches "template" already encoding
        # the chain letters used, e.g. "2xwaAB").
        rec_best = rec_df.groupby("template")["tm_score"].max()
        lig_best = lig_df.groupby("template")["tm_score"].max()
        shared = set(rec_best.index) & set(lig_best.index)
        for t in shared:
            scores_by_template[t] = float(rec_best[t]) + float(lig_best[t])
        print(f"loaded {len(scores_by_template)} template scores from "
              f"{rec_csv} + {lig_csv}")

    native_iface = native_interface_residues(
        args.native, args.receptor_chain, args.ligand_chain)
    if not native_iface:
        print(f"ERROR: native interface came back empty for chains "
              f"{args.receptor_chain}/{args.ligand_chain} in {args.native}. "
              f"Check chain IDs before trusting anything else.", file=sys.stderr)
        sys.exit(1)
    print(f"native receptor interface: {len(native_iface)} residues "
          f"({sorted(native_iface)})")

    files = sorted(glob.glob(os.path.join(args.candidates_dir, "*.pdb")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"ERROR: no .pdb files found in {args.candidates_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"scanning {len(files)} candidates in {args.candidates_dir} ...")

    rows = []
    n_any_overlap = 0
    n_errors = 0
    for i, f in enumerate(files, 1):
        cand_res, err = candidate_receptor_residues(
            f, args.receptor_chain, args.ligand_chain)
        name = os.path.basename(f)
        if err:
            n_errors += 1
            rows.append({"candidate": name, "template": name.split("_")[0],
                         "receptor_residues_matched": "", "overlap_count": "",
                         "receptor_contact_residue_count": "", "jaccard": "",
                         "score": "", "error": err})
            continue
        overlap = cand_res & native_iface
        j = jaccard(cand_res, native_iface)
        if overlap:
            n_any_overlap += 1
        template = name.split("_")[0]
        rows.append({
            "candidate": name,
            "template": template,
            "receptor_residues_matched": " ".join(str(r) for r in sorted(overlap)),
            "overlap_count": len(overlap),
            "receptor_contact_residue_count": len(cand_res),
            "jaccard": round(j, 4),
            "score": scores_by_template.get(template, ""),
            "error": "",
        })
        if i % 100 == 0:
            print(f"  ... {i}/{len(files)}")

    rows.sort(key=lambda r: (r["jaccard"] if r["jaccard"] != "" else -1), reverse=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "candidate", "template", "receptor_residues_matched",
            "overlap_count", "receptor_contact_residue_count", "jaccard",
            "score", "error"])
        w.writeheader()
        w.writerows(rows)

    if scores_by_template:
        # Exact rank-by-score within the SAME population used by the real
        # pipeline (process_pair_for_template's own accepted-candidate pool,
        # which is exactly what's sitting in --candidates-dir already).
        by_score = sorted(
            [r for r in rows if r["error"] == "" and r["score"] != ""],
            key=lambda r: r["score"], reverse=True)
        for rank, r in enumerate(by_score, 1):
            r["score_rank"] = rank
        print(f"\nscore-based ranking reconstructed for {len(by_score)} candidates "
              f"(current PRISM_TOP_K_REFINE cutoff would take the top few of these).")
        top_by_jaccard = [r for r in rows if r["error"] == "" and r["jaccard"] != ""]
        top_by_jaccard.sort(key=lambda r: r["jaccard"], reverse=True)
        print("where the best real-interface-overlap candidates actually rank by score:")
        for r in top_by_jaccard[:5]:
            rank = next((x["score_rank"] for x in by_score if x["candidate"] == r["candidate"]), "n/a")
            print(f"  {r['template']:<12s} jaccard={r['jaccard']:.3f}  score_rank={rank} of {len(by_score)}")

    # Does raw post-transform contact size (F11's own, already-computed signal,
    # uses no knowledge of the native answer) predict real interface overlap
    # any better than the alignment-score ranking currently used to pick
    # which candidates get refined? Quick correlation, no scipy dependency.
    scored = [r for r in rows if r["error"] == ""]
    if len(scored) >= 3:
        import numpy as np
        jvals = np.array([r["jaccard"] for r in scored], dtype=float)
        cvals = np.array([r["receptor_contact_residue_count"] for r in scored], dtype=float)
        if jvals.std() > 0 and cvals.std() > 0:
            pearson_r = float(np.corrcoef(jvals, cvals)[0, 1])
        else:
            pearson_r = float("nan")
        # Spearman = Pearson on ranks, computed by hand to avoid a scipy dependency.
        def _ranks(a):
            order = np.argsort(a)
            r = np.empty_like(order, dtype=float)
            r[order] = np.arange(len(a))
            return r
        if jvals.std() > 0 and cvals.std() > 0:
            spearman_r = float(np.corrcoef(_ranks(jvals), _ranks(cvals))[0, 1])
        else:
            spearman_r = float("nan")
        print(f"\ncontact-count vs real-interface-overlap correlation across all "
              f"{len(scored)} scored candidates:")
        print(f"  pearson r  = {pearson_r:.3f}")
        print(f"  spearman r = {spearman_r:.3f}")
        top10_by_contacts = sorted(scored, key=lambda r: r["receptor_contact_residue_count"],
                                    reverse=True)[:10]
        n_top10_contacts_touch_iface = sum(1 for r in top10_by_contacts if r["overlap_count"] > 0)
        print(f"  of the top 10 candidates BY CONTACT COUNT, {n_top10_contacts_touch_iface} "
              f"of 10 touch the real interface at all")

        if scores_by_template:
            svals = np.array([r["score"] for r in scored if r["score"] != ""], dtype=float)
            jvals2 = np.array([r["jaccard"] for r in scored if r["score"] != ""], dtype=float)
            if len(svals) >= 3 and svals.std() > 0 and jvals2.std() > 0:
                pearson_score = float(np.corrcoef(jvals2, svals)[0, 1])
                spearman_score = float(np.corrcoef(_ranks(jvals2), _ranks(svals))[0, 1])
                top10_by_score = sorted(
                    [r for r in scored if r["score"] != ""],
                    key=lambda r: r["score"], reverse=True)[:10]
                n_top10_score_touch_iface = sum(1 for r in top10_by_score if r["overlap_count"] > 0)
                print(f"\nALIGNMENT SCORE (the metric actually used for ranking today) "
                      f"vs real-interface-overlap, same {len(svals)} candidates:")
                print(f"  pearson r  = {pearson_score:.3f}")
                print(f"  spearman r = {spearman_score:.3f}")
                print(f"  of the top 10 candidates BY ALIGNMENT SCORE, "
                      f"{n_top10_score_touch_iface} of 10 touch the real interface at all")
                print(f"\n  >>> direct comparison: contacts r={pearson_r:.3f} vs "
                      f"score r={pearson_score:.3f} -- whichever is higher is the "
                      f"better-motivated ranking key on this data <<<")

    print()
    print(f"RESULT: {n_any_overlap} of {len(files)} candidates touch the real "
          f"receptor interface at all ({n_errors} could not be parsed).")
    if n_any_overlap == 0:
        print("Nothing in the full pool ever reaches the true interface. "
              "The problem is upstream of refinement AND upstream of candidate "
              "selection -- look at the alignment stage (F8), not F10.")
    else:
        print("Top 5 candidates by receptor-interface overlap:")
        shown = 0
        for r in rows:
            if r["jaccard"] == "" or r["jaccard"] == 0:
                continue
            print(f"  {r['template']:<12s} jaccard={r['jaccard']:.3f} "
                  f"overlap={r['overlap_count']} residues={r['receptor_residues_matched']}")
            shown += 1
            if shown >= 5:
                break
        print("If none of these made it into the refined set, the ranking rule "
              "(alignment score before refinement) is losing a real candidate -- "
              "worth widening PRISM_TOP_K_REFINE or ranking by this overlap "
              "measure instead of alignment score.")
    print(f"\nfull table written to {args.out}")


if __name__ == "__main__":
    main()

"""TM-align-based structural alignment stage.

Aligns each target surface (CA-only) to every requested template interface
(per chain) and produces per-target alignment summaries plus per-pair JSONs
that downstream stages need (translation/rotation + residue match dict).
"""

import os
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser

ALIGNMENT_DIR = "processed/alignment"
os.makedirs(ALIGNMENT_DIR, exist_ok=True)

MINIMUM_RESIDUE_MATCH_COUNT = 15
MINIMUM_RESIDUE_MATCH_PERCENTAGE = 50.0
#changed from 10.0 to 50.0
DIFF_PERCENTAGE = 20.0
TM_SCORE_THRESHOLD = 0.2
TEMPLATE_RESIDUE_COUNT = 50

TM_ALIGN_BIN = os.environ.get("TM_ALIGN_BIN", "external_tools/TMalign")


def check_alignment_passes_thresholds(alignment):
    match_count = alignment["match_count"]
    tm_score = alignment["tm_score"]
    len_template = alignment["len_template"]
    if len_template <= 0 or match_count < MINIMUM_RESIDUE_MATCH_COUNT or tm_score < TM_SCORE_THRESHOLD:
        return False
    match_score = (match_count / len_template) * 100.0
    if len_template > TEMPLATE_RESIDUE_COUNT:
        return match_score > (MINIMUM_RESIDUE_MATCH_PERCENTAGE - DIFF_PERCENTAGE)
    return match_score > MINIMUM_RESIDUE_MATCH_PERCENTAGE


def _surface_path(protein):
    candidates = [
        f"processed/surface_extraction/{protein}.asa.pdb",
        f"processed/surface_extraction/{protein}_asa.pdb",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def _interface_path(template, chain):
    return f"templates/interfaces/{template}_{chain}_int.pdb"


def _parse_matrix(matrix_path):
    translation = [0.0, 0.0, 0.0]
    rotation = [[0.0, 0.0, 0.0] for _ in range(3)]
    with open(matrix_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            try:
                row = int(tokens[0])
            except ValueError:
                continue
            if row in (0, 1, 2):
                translation[row] = float(tokens[1])
                rotation[row] = [float(tokens[2]), float(tokens[3]), float(tokens[4])]
    return translation, rotation


def _parse_tmalign_output(tm_path):
    tm_score_1 = tm_score_2 = 0.0
    match_count = len_target = len_template = 0
    seq1 = seq2 = ""
    seq_block = []

    with open(tm_path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith("Aligned length"):
            try:
                match_count = int(line.split("=")[1].split(",")[0].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("Length of Chain_1"):
            try:
                len_target = int(line.split(":")[1].split("residues")[0].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("Length of Chain_2"):
            try:
                len_template = int(line.split(":")[1].split("residues")[0].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("TM-score"):
            try:
                score = float(line.split()[1])
            except (ValueError, IndexError):
                continue
            if "Chain_1" in line:
                tm_score_1 = score
            elif "Chain_2" in line:
                tm_score_2 = score
        elif "denotes residue pairs" in line:
            seq_block = lines[i + 1: i + 4]

    if len(seq_block) >= 3:
        seq1 = seq_block[0].rstrip("\n")
        seq2 = seq_block[2].rstrip("\n")

    return {
        "match_count": match_count,
        "tm_score": max(tm_score_1, tm_score_2),
        "len_target": len_target,
        "len_template": len_template,
        "seq1": seq1,
        "seq2": seq2,
    }


def _extract_residue_ids(path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", path)
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    residues.append((chain.id, int(residue.id[1]), residue.get_resname()))
        break
    return residues


def _build_match_dict(seq1, seq2, target_residues, template_residues):
    """Map template interface residues -> target surface residues using the aligned sequences."""
    if not seq1 or not seq2:
        return {}
    match = {}
    i1 = i2 = 0
    for a, b in zip(seq1, seq2):
        ga = a == "-"; gb = b == "-"
        if not ga and not gb and i1 < len(target_residues) and i2 < len(template_residues):
            tch, tres, _ = target_residues[i1]
            mch, mres, _ = template_residues[i2]
            match[f"{mch}.{b}.{mres}"] = f"{tch}.{a}.{tres}"
        if not ga:
            i1 += 1
        if not gb:
            i2 += 1
    return match


def _run_single_alignment(protein, template, chain):
    protein_path = _surface_path(protein)
    interface_path = _interface_path(template, chain)
    matrix_path = os.path.join(ALIGNMENT_DIR, f"{protein}_{template}_{chain}_matrix.out")
    tm_path = os.path.join(ALIGNMENT_DIR, f"{protein}_{template}_{chain}_out.tm")
    json_path = os.path.join(ALIGNMENT_DIR, f"{protein}_{template}_{chain}.json")

    if not os.path.exists(protein_path) or not os.path.exists(interface_path):
        return None
    try:
        with open(tm_path, "w") as f:
            subprocess.run(
                [TM_ALIGN_BIN, protein_path, interface_path, "-m", matrix_path],
                stdout=f, stderr=subprocess.DEVNULL, check=True,
            )
        translation, rotation = _parse_matrix(matrix_path)
        parsed = _parse_tmalign_output(tm_path)
        target_residues = _extract_residue_ids(protein_path)
        template_residues = _extract_residue_ids(interface_path)
        match_dict = _build_match_dict(parsed["seq1"], parsed["seq2"], target_residues, template_residues)

        payload = {
            "protein": protein,
            "template": template,
            "chain": chain,
            "match_count": parsed["match_count"],
            "tm_score": parsed["tm_score"],
            "len_target": parsed["len_target"],
            "len_template": parsed["len_template"],
            "translation": translation,
            "rotation_mat": rotation,
            "match_dict": match_dict,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f)
        return payload
    except Exception as exc:
        print(f"TM-align failed for {protein} vs {template}_{chain}: {exc}")
        return None
    finally:
        for p in (matrix_path, tm_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def align(targets, templates, max_workers=None):
    tasks = []
    for target in targets:
        for template in templates:
            for chain in template[4:]:
                interface_path = _interface_path(template, chain)
                if not os.path.exists(interface_path):
                    continue
                try:
                    if os.path.getsize(interface_path) <= 0:
                        continue
                except OSError:
                    continue
                tasks.append((target, template, chain))

    if not tasks:
        print("No alignment tasks to run.")
        return

    workers = max_workers or min(32, os.cpu_count() or 4)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single_alignment, p, t, c): (p, t, c) for p, t, c in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="TM-align"):
            payload = fut.result()
            if payload is not None:
                rows.append({
                    "protein": payload["protein"],
                    "template": payload["template"],
                    "chain": payload["chain"],
                    "match_count": payload["match_count"],
                    "tm_score": payload["tm_score"],
                    "len_target": payload["len_target"],
                    "len_template": payload["len_template"],
                    "translation": json.dumps(payload["translation"]),
                    "rotation_mat": json.dumps(payload["rotation_mat"]),
                })

    if not rows:
        return

    rows_by_target = {}
    for row in rows:
        if check_alignment_passes_thresholds(row):
            rows_by_target.setdefault(row["protein"], []).append(row)

    for target, target_rows in rows_by_target.items():
        out_csv = os.path.join(ALIGNMENT_DIR, f"{target}.csv")
        fieldnames = ["protein", "template", "chain", "match_count", "tm_score",
                     "len_target", "len_template", "translation", "rotation_mat"]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(target_rows)
        df = pd.read_csv(out_csv).sort_values("tm_score", ascending=False)
        df.to_csv(out_csv, index=False)
        print(f"Wrote {len(target_rows)} alignments to {out_csv}")

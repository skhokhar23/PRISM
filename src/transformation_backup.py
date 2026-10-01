"""Transformation stage.

For each receptor-ligand pair, find templates where one template chain matches
the receptor surface and the other chain matches the ligand surface (in either
orientation), apply the corresponding rigid-body transforms to position both
proteins in the template frame, then filter candidates by:

  * hotspot support (the target residues hitting the template chain include at
    least one HotPoint residue)
  * inter-target Cα clash check
  * residue contact support (the template contact map has at least one inter-chain
    contact between residues that map to both sides)

The final accepted complex is written to `processed/output/` so the caller can
report the final PDB.
"""

import os
import json
import pandas as pd

from .utils import distance_calculator
from .pdb_download import split_target_id

ALIGNMENT_DIR = "processed/alignment"
HOTSPOT_DIR = "templates/hotspots"
CONTACTS_DIR = "templates/contacts"
TRANSFORMATION_DIR = "processed/transformation"
MERGE_DIR = "processed/output"
os.makedirs(TRANSFORMATION_DIR, exist_ok=True)
os.makedirs(MERGE_DIR, exist_ok=True)

CLASHING_DISTANCE = 3.0
MIN_HOTSPOT_MATCH = 1
MIN_CONTACT_MATCH = 1


def transformer(receptor_targets, ligand_targets):
    """Return list of `(receptor, ligand, template, output_pdb)` for accepted pairs."""
    accepted = []
    for receptor, ligand in zip(receptor_targets, ligand_targets):
        candidates = process_pair_for_template(receptor, ligand)
        if candidates:
            best = candidates[0]
            accepted.append((receptor, ligand, best["template"], best["output_pdb"]))
    return accepted


def _load_alignment(target):
    csv_path = os.path.join(ALIGNMENT_DIR, f"{target}.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _load_match_dict(protein, template, chain):
    path = os.path.join(ALIGNMENT_DIR, f"{protein}_{template}_{chain}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError:
            return {}
    return payload.get("match_dict", {})


def _load_hotspots(template):
    path = os.path.join(HOTSPOT_DIR, f"{template}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _load_contacts(template):
    path = os.path.join(CONTACTS_DIR, f"{template}.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, dict):
        out = []
        for v in payload.values():
            out.extend(v)
        return out
    return payload


def _hotspot_supported(template, chain, target_match_dict):
    hotspots = _load_hotspots(template)
    chain_hotspots = {str(num) for num, _name in hotspots.get(chain, [])}
    if not chain_hotspots:
        return True
    matched_template_residues = {key.split(".")[-1] for key in target_match_dict.keys()}
    return len(chain_hotspots & matched_template_residues) >= MIN_HOTSPOT_MATCH


def _contact_supported(template, receptor_match, ligand_match, chain_R, chain_L):
    contacts = _load_contacts(template)
    if not contacts:
        return True
    receptor_template_res = {int(k.split(".")[-1]) for k in receptor_match.keys() if k.startswith(f"{chain_R}.")}
    ligand_template_res = {int(k.split(".")[-1]) for k in ligand_match.keys() if k.startswith(f"{chain_L}.")}
    for pair in contacts:
        if len(pair) != 2:
            continue
        a, b = int(pair[0]), int(pair[1])
        if (a in receptor_template_res and b in ligand_template_res) or \
           (b in receptor_template_res and a in ligand_template_res):
            return True
    return False


def _parse_vec(val, default):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            return default
    return val if val is not None else default


def process_pair_for_template(receptor, ligand):
    """Find templates that explain the receptor-ligand interaction."""
    receptor_df = _load_alignment(receptor)
    ligand_df = _load_alignment(ligand)
    if receptor_df is None or ligand_df is None or receptor_df.empty or ligand_df.empty:
        return []

    candidates = []
    shared_templates = set(receptor_df["template"]).intersection(ligand_df["template"])
    for template in shared_templates:
        rec_rows = receptor_df[receptor_df["template"] == template].to_dict("records")
        lig_rows = ligand_df[ligand_df["template"] == template].to_dict("records")
        for rec in rec_rows:
            for lig in lig_rows:
                if rec["chain"] == lig["chain"]:
                    continue
                candidate = _evaluate(receptor, ligand, template, rec, lig)
                if candidate:
                    candidates.append(candidate)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def _evaluate(receptor, ligand, template, rec_align, lig_align):
    rec_chain = rec_align["chain"]
    lig_chain = lig_align["chain"]
    rec_match = _load_match_dict(receptor, template, rec_chain)
    lig_match = _load_match_dict(ligand, template, lig_chain)
    if not rec_match or not lig_match:
        return None

    if not _hotspot_supported(template, rec_chain, rec_match):
        return None
    if not _hotspot_supported(template, lig_chain, lig_match):
        return None
    if not _contact_supported(template, rec_match, lig_match, rec_chain, lig_chain):
        return None

    rec_pdb_id, rec_chains = split_target_id(receptor)
    lig_pdb_id, lig_chains = split_target_id(ligand)

    receptor_input = f"processed/pdbs/{rec_pdb_id}.pdb"
    ligand_input = f"processed/pdbs/{lig_pdb_id}.pdb"
    receptor_output = f"{TRANSFORMATION_DIR}/{template}_{receptor}_{ligand}_R.pdb"
    ligand_output = f"{TRANSFORMATION_DIR}/{template}_{receptor}_{ligand}_L.pdb"
    final_output = f"{MERGE_DIR}/{template}_{receptor}_{ligand}.pdb"

    rec_translation = _parse_vec(rec_align.get("translation"), [0.0, 0.0, 0.0])
    rec_rotation = _parse_vec(rec_align.get("rotation_mat"), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    lig_translation = _parse_vec(lig_align.get("translation"), [0.0, 0.0, 0.0])
    lig_rotation = _parse_vec(lig_align.get("rotation_mat"), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    apply_tm_transform(receptor_input, receptor_output, rec_translation, rec_rotation, keep_chains=set(rec_chains))
    apply_tm_transform(ligand_input, ligand_output, lig_translation, lig_rotation, keep_chains=set(lig_chains))

    if not pair_has_acceptable_clashes(receptor_output, ligand_output):
        return None

    merge_pdb_files(receptor_output, ligand_output, final_output)
    score = float(rec_align["tm_score"]) + float(lig_align["tm_score"])
    return {
        "receptor": receptor,
        "ligand": ligand,
        "template": template,
        "rec_chain": rec_chain,
        "lig_chain": lig_chain,
        "score": score,
        "output_pdb": final_output,
    }


def apply_tm_transform(input_pdb, output_pdb, translation, rotation_mat, keep_chains=None):
    try:
        with open(input_pdb) as in_f, open(output_pdb, "w") as out_f:
            for line in in_f:
                if line.startswith(("ATOM", "HETATM")):
                    if keep_chains and len(line) > 21 and line[21] not in keep_chains:
                        continue
                    try:
                        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    except ValueError:
                        out_f.write(line)
                        continue
                    nx = x * rotation_mat[0][0] + y * rotation_mat[0][1] + z * rotation_mat[0][2] + translation[0]
                    ny = x * rotation_mat[1][0] + y * rotation_mat[1][1] + z * rotation_mat[1][2] + translation[1]
                    nz = x * rotation_mat[2][0] + y * rotation_mat[2][1] + z * rotation_mat[2][2] + translation[2]
                    out_f.write(f"{line[:30]}{nx:8.3f}{ny:8.3f}{nz:8.3f}{line[54:]}")
                elif line.startswith("TER"):
                    out_f.write(line)
            out_f.write("END\n")
        return True
    except Exception as exc:
        print(f"Failed to transform {input_pdb}: {exc}")
        return False


def merge_pdb_files(receptor_path, ligand_path, output_path):
    serial = 1
    with open(output_path, "w") as out_f:
        for path in (receptor_path, ligand_path):
            with open(path) as fh:
                for line in fh:
                    if line.startswith(("ATOM", "HETATM")):
                        new_line = f"{line[:6]}{serial:5d}{line[11:]}"
                        out_f.write(new_line)
                        serial += 1
                    elif line.startswith("TER"):
                        out_f.write(line)
            out_f.write("TER\n")
        out_f.write("END\n")


def _read_ca_coords(path):
    coords = []
    if not os.path.exists(path):
        return coords
    with open(path) as fh:
        for line in fh:
            if line.startswith("ATOM") and " CA " in line[:20]:
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    continue
    return coords


def pair_has_acceptable_clashes(receptor_path, ligand_path):
    receptor_coords = _read_ca_coords(receptor_path)
    ligand_coords = _read_ca_coords(ligand_path)
    if not receptor_coords or not ligand_coords:
        return False
    for rc in receptor_coords:
        for lc in ligand_coords:
            if distance_calculator(rc, lc) < CLASHING_DISTANCE:
                return False
    return True

import os
import freesasa
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


def _resolve_pdb_path(target, pdb_root):
    """Resolve the path to a target's PDB file.

    `pdb_root` is the directory containing the `pdbs/` subfolder, e.g. `templates`
    or `processed`. Backwards-compatible fallback also accepts a direct path to
    the pdbs directory.
    """
    pdb_id = target[:4].lower()
    candidates = [
        os.path.join(pdb_root, "pdbs", f"{pdb_id}.pdb"),
        os.path.join(pdb_root, f"{pdb_id}.pdb"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not find PDB for {target!r} in any of: {candidates}"
    )


def get_asa_complex(target, pdb_root):
    """Compute relative ASA for the requested chains of a target.

    `target` is `{pdb_id}{chain_letters}` (e.g. `3i6eEF`). If no chain letters
    are provided, all chains are returned.
    Returns `{chain_id: {res_num: rel_asa_percent}}`.
    """
    pdb_path = _resolve_pdb_path(target, pdb_root)
    chains = [c.upper() for c in target[4:]]

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("target", pdb_path)
        result, _ = freesasa.calcBioPDB(structure)
        residue_areas = result.residueAreas()
    except Exception as exc:
        raise RuntimeError(f"FreeSASA failed for {pdb_path}: {exc}") from exc

    relative_asa = {}
    for chain_id, chain_dict in residue_areas.items():
        if chains and chain_id not in chains:
            continue
        relative_asa[chain_id] = {}
        for res_num, ra in chain_dict.items():
            try:
                relative_asa[chain_id][int(res_num)] = float(ra.relativeTotal) * 100.0
            except (TypeError, ValueError):
                continue
    return relative_asa


def get_asa_flat(target, pdb_root):
    """Return a flat dict keyed by `RESNAME_RESNUMBER_CHAINID`.

    Useful for the hotspot module that pairs residue ids with contact potentials.
    """
    pdb_path = _resolve_pdb_path(target, pdb_root)
    chains = [c.upper() for c in target[4:]]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", pdb_path)

    res_name_by_key = {}
    model = next(structure.get_models())
    for chain in model:
        if chains and chain.id not in chains:
            continue
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            res_name_by_key[(chain.id, int(residue.id[1]))] = residue.get_resname().strip()

    nested = get_asa_complex(target, pdb_root)
    flat = {}
    for chain_id, res_map in nested.items():
        for res_num, rel_asa in res_map.items():
            res_name = res_name_by_key.get((chain_id, res_num))
            if res_name is None:
                continue
            flat[f"{res_name}_{res_num}_{chain_id}"] = rel_asa
    return flat

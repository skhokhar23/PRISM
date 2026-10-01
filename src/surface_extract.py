import os
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

from .sasa_utils import get_asa_complex
from .utils import distance_calculator

RSATHRESHOLD = 15.0
NEIGHBOR_CA_DISTANCE = 6.0

SURFACE_EXTRACTION_DIR = "processed/surface_extraction"
os.makedirs(SURFACE_EXTRACTION_DIR, exist_ok=True)


def extract_surfaces(queries):
    failed_count = 0
    for protein in queries:
        try:
            ok = extract_surface(protein)
        except Exception as exc:
            print(f"extract_surface failed for {protein}: {exc}")
            ok = False
        if not ok:
            failed_count += 1
    return failed_count


def extract_surface(protein, pdb_root="processed"):
    """Write a CA-only PDB of surface (high-rASA + neighbour) residues for `protein`.

    Supports multi-chain targets such as `1fgnHL` (chains H,L).
    """
    asa_complex = get_asa_complex(protein, pdb_root)
    if not asa_complex:
        print(f"No ASA computed for {protein}")
        return False

    rsa_residues = {
        chain: {int(r) for r, v in residues.items() if v > RSATHRESHOLD}
        for chain, residues in asa_complex.items()
    }

    pdb_id = protein[:4].lower()
    requested_chains = set(protein[4:]) if len(protein) > 4 else set(asa_complex.keys())

    pdb_path = os.path.join(pdb_root, "pdbs", f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        print(f"PDB file not found: {pdb_path}")
        return False
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein, pdb_path)
    model = structure[0]

    chain_ca_data = {}
    chain_rsa_coords = {}

    for chain in model:
        if chain.id not in requested_chains:
            continue
        chain_ca_data[chain.id] = []
        chain_rsa_coords[chain.id] = []
        for residue in chain:
            if not is_aa(residue, standard=True) or "CA" not in residue:
                continue
            res_seq = int(residue.id[1])
            res_name = residue.get_resname()
            coords = list(residue["CA"].get_coord())
            chain_ca_data[chain.id].append((res_name, res_seq, coords))
            if res_seq in rsa_residues.get(chain.id, set()):
                chain_rsa_coords[chain.id].append(coords)

    surface_lines = []
    for chain_id, residues in chain_ca_data.items():
        rsa_coords = chain_rsa_coords.get(chain_id, [])
        if not rsa_coords:
            continue
        for res_name, res_seq, coords in residues:
            if res_seq in rsa_residues[chain_id]:
                surface_lines.append((res_name, chain_id, res_seq, coords))
                continue
            for rc in rsa_coords:
                if distance_calculator(coords, rc) <= NEIGHBOR_CA_DISTANCE:
                    surface_lines.append((res_name, chain_id, res_seq, coords))
                    break

    if not surface_lines:
        print(f"No surface CA lines found for {protein}")
        return False

    out_path = os.path.join(SURFACE_EXTRACTION_DIR, f"{protein}.asa.pdb")
    with open(out_path, "w") as f:
        prev_chain = None
        for i, (res_name, chain_id, res_seq, coords) in enumerate(surface_lines, start=1):
            if prev_chain is not None and chain_id != prev_chain:
                f.write("TER\n")
            f.write(_format_ca_pdb_line(i, res_name, chain_id, res_seq, *coords))
            prev_chain = chain_id
        f.write("TER\n")
    return True


def _format_ca_pdb_line(serial, res_name, chain_id, res_seq, x, y, z):
    return (
        f"ATOM  {serial:5d}  CA  {res_name:3s} {chain_id}{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
    )


if __name__ == "__main__":
    extract_surfaces(["3i6eE", "3i6eF"])

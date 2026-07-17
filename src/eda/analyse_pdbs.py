import os
import pandas as pd
from Bio.PDB import PDBParser

#changed from .pdb_download to src.pdb_download
#from src.pdb_download import download_pdb_file
#from src.utils import STANDARD_AA
from src.pdb_download import download_pdb_file
from src.utils import STANDARD_AA

TEMPLATES_PDBS_DIR = "templates/pdbs"
os.makedirs(TEMPLATES_PDBS_DIR, exist_ok=True)

def analyse_pdb(filepath):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", filepath)

    num_models = len(structure)

    residues_seen = set()
    chain_set = set()
    first_model = next(structure.get_models())
    for chain in first_model:
        chain_id = chain.id
        for residue in chain:
            resname = residue.get_resname().strip()
            if resname in STANDARD_AA:
                resseq = residue.id[1]
                residues_seen.add((chain_id, resseq))
                chain_set.add(chain_id)

    chain_names = ",".join(sorted(chain_set))
    num_chains = len(chain_set)
    total_amino_acids = len(residues_seen)

    # Get EXPDTA from file manually (since Biopython does not parse it)
    expdta = ""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("EXPDTA"):
                    expdta = line[10:80].strip()
                    break
    except Exception:
        pass

    # Structure type: NMR vs single
    exp_upper = expdta.upper()
    if "NMR" in exp_upper or "NUCLEAR MAGNETIC" in exp_upper or num_models > 1:
        structure_type = "NMR"
    else:
        structure_type = "single"

    return {
        "pdb_id": filepath.split("/")[-1].split(".")[0].lower(),
        "num_chains": num_chains,
        "chain_names": chain_names,
        "total_amino_acids": total_amino_acids,
        "structure_type": structure_type,
        "expdta": expdta,
        "num_models": num_models,
    }

def run_analysis():
    with open("templates/templates.txt", "r") as f:
        templates = [line.strip() for line in f.readlines() if len(line.strip()) > 0]

    results = []
    failed_templates = []
    for idx, template in enumerate(templates):
        try:
            if not os.path.exists(f"{TEMPLATES_PDBS_DIR}/{template[:4].lower()}.pdb"):
                print(f"Template {template} does not exist, start downloading...")
                download_pdb_file(template[:4].lower(), TEMPLATES_PDBS_DIR)
                if not os.path.exists(f"{TEMPLATES_PDBS_DIR}/{template[:4].lower()}.pdb"):
                    print(f"{idx+1}/{len(templates)}: Failed to download template {template}, skipping...")
                    failed_templates.append(template)
                    continue
            info = analyse_pdb(f"{TEMPLATES_PDBS_DIR}/{template[:4].lower()}.pdb")
            info["template"] = template
            results.append(info)
        except Exception as e:
            print(f"Failed to analyse template {template}: {e}")
            failed_templates.append(template)

    df = pd.DataFrame(results)
    df.to_csv("templates/templates_analysis.csv", index=False)
    filtered = df[(df["num_chains"] > 1) & (df["structure_type"] == "single")]
    filtered.to_csv("templates/templates_filtered.csv", index=False)

    templates_updatex_path = "templates/checked_templates.txt"
    with open(templates_updatex_path, "w", encoding="utf-8") as f:
        for tpl in filtered["template"]:
            f.write(f"{tpl}\n")

    missing_templates_path = "templates/missing_templates.txt"
    with open(missing_templates_path, "w", encoding="utf-8") as f:
        for tpl in failed_templates:
            f.write(f"{tpl}\n")

    # for tpl in failed_templates:
    #     pdb_path = f"{TEMPLATES_PDBS_DIR}/{tpl[:4].lower()}.pdb"
    #     if os.path.exists(pdb_path):
    #         try:
    #             os.remove(pdb_path)
    #             print(f"Deleted missing template PDB: {pdb_path}")
    #         except Exception as e:
    #             print(f"Could not delete {pdb_path}: {e}")

    return results, len(filtered), len(failed_templates)

if __name__ == "__main__":
    results, filtered_count, failed_count = run_analysis()
    print(f"Analysed {len(results)} template files.")
    print(f"Filtered {filtered_count} templates")
    print(f"Failed {failed_count} templates")


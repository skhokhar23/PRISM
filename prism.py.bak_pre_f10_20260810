import argparse
from src.pdb_download import pdb_downloader
from src.eda.analyse_pdbs import run_analysis
from src.template_generate import template_generator
from src.surface_extract import extract_surfaces
from src.alignment import align
from src.alignment_gtalign import align_gtalign
from src.transformation import transformer
from src.rosetta_refinement import refiner
from src.compare import compare_and_summarize

def main(args):
    print("[1/6] PDB download")
    receptor_targets, ligand_targets = pdb_downloader(args)
    targets = sorted(set(receptor_targets + ligand_targets))
    for r, l in zip(receptor_targets, ligand_targets):
        print(f"  {r} -> {l}")

    if args.generate_templates:
        print("[2a/6] Template analysis & filtering")
        results, filtered_count, failed_count = run_analysis()
        print(f"  analysed={len(results)} filtered={filtered_count} failed={failed_count}")

        print("[2b/6] Template artifact generation")
        templates = template_generator()
        print(f"  generated {len(templates)} templates")
    else:
        with open("templates/calculated_templates.txt") as f:
            templates = [line.strip() for line in f if line.strip()]
        print(f"[2/6] Loaded {len(templates)} pre-computed templates")

    if args.template_limit > 0:
        templates = templates[: args.template_limit]
        print(f"  using top {len(templates)} templates")

    print("[3/6] Target surface extraction")
    failed_surfaces = extract_surfaces(targets)
    if failed_surfaces:
        print(f"  WARNING: surface extraction failed for {failed_surfaces} target(s)")

    print(f"[4/6] Structural alignment ({args.aligner})")
    if args.aligner == "tmalign":
        align(targets, templates)
    else:
        align_gtalign(targets, templates, gtalign_path=args.gtalign_path)

    print("[5/6] Transformation + filtering")
    passed = transformer(receptor_targets, ligand_targets)
    print(f"  accepted {len(passed)} receptor-ligand candidates")
    for receptor, ligand, template, output in passed:
        print(f"  {receptor} + {ligand} via {template} -> {output}")

    compare_pairs = list(passed)
    if args.refine and passed:
        print("[6a/6] Rosetta refinement")
        refined = refiner(passed)
        for rec, lig, isc, tsc, op in refined:
            print(f"  refined {rec} + {lig}: int={isc} total={tsc} -> {op}")
        if refined:
            compare_pairs = [
                (rec, lig, tpl, out_pdb)
                for (rec, lig, tpl, _), (_, _, _, _, out_pdb) in zip(passed, refined)
            ]

    print("[6/6] Compare outputs vs native + DockQ")
    if compare_pairs:
        summary_csv, rows = compare_and_summarize(
            compare_pairs,
            dockq_no_align=args.dockq_no_align,
            n_jobs=args.compare_jobs,
        )
        print(f"  summary -> {summary_csv}")
        for row in rows:
            dockq = row["dockq"] if row["dockq"] is not None else "NA"
            rmsd = row["receptor_ca_rmsd"] if row["receptor_ca_rmsd"] is not None else "NA"
            print(
                f"  {row['template']}: output={row['output_pdb']} "
                f"receptor_RMSD={rmsd} DockQ={dockq}"
            )
    else:
        print("  no accepted outputs to compare")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_csv", type=str, default="inputs.csv")
    parser.add_argument("--generate_templates", action="store_true", help="Run template analysis & artifact generation before alignment.")
    parser.add_argument("--aligner", choices=["tmalign", "gtalign"], default="tmalign")
    parser.add_argument("--gtalign_path", default="gtalign")
    parser.add_argument("--template_limit", type=int, default=100, help="Limit number of templates aligned (0 = all)")
    parser.add_argument("--refine", action="store_true", help="Run Rosetta refinement on accepted candidates")
    parser.add_argument(
        "--dockq-no-align",
        action="store_true",
        help="Pass --no_align to DockQ in the final compare step (default: align)",
    )
    parser.add_argument(
        "--compare-jobs",
        type=int,
        default=1,
        help="Parallel workers for final compare/DockQ step",
    )
    args = parser.parse_args()
    main(args)

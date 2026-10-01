"""Rosetta refinement stage using PyRosetta.

Supports multi-chain receptor/ligand inputs. Accepted `passed_pairs` tuples may
be either:
  - 2-tuples `(receptor_path, ligand_path)`
  - 4-tuples `(receptor_id, ligand_id, template_id, complex_pdb_path)` as
    produced by `transformation.transformer`.
"""

import os
import shutil

from .contact import get_contacts_from_atom_lines

ROSETTA_INT_SCORE_THRESHOLD = float(os.environ.get("ROSETTA_INT_SCORE_THRESHOLD", -5.0))

ROSETTA_DIR = "processed/rosetta_refinement"
ENERGY_DIR = os.path.join(ROSETTA_DIR, "energies")
STRUCTURE_DIR = os.path.join(ROSETTA_DIR, "structures")
os.makedirs(ROSETTA_DIR, exist_ok=True)
os.makedirs(ENERGY_DIR, exist_ok=True)
os.makedirs(STRUCTURE_DIR, exist_ok=True)


def refiner(passed_pairs):
    summary_path = os.path.join(ROSETTA_DIR, "refinement_energies.txt")
    rows = []
    for entry in passed_pairs:
        if entry is None:
            continue
        receptor_path, ligand_path, left_chains, right_chains, combined_path = _resolve_entry(entry)
        if not receptor_path or not ligand_path:
            continue
        totalscore, intscore, out_path = calculate_energy(
            receptor_path, ligand_path,
            combined_path=combined_path,
            left_chains=left_chains,
            right_chains=right_chains,
        )
        if intscore != "-":
            rows.append((receptor_path, ligand_path, intscore, totalscore, out_path))

    with open(summary_path, "w") as fh:
        for rec, lig, isc, tsc, op in rows:
            fh.write(f"{rec}\t{lig}\t{isc}\t{tsc}\t{op}\n")
    return rows


def _resolve_entry(entry):
    """Accept legacy `(rec_path, lig_path)` and new `(rec_id, lig_id, template, combined_pdb)`."""
    if len(entry) == 2:
        rec_path, lig_path = entry
        return rec_path, lig_path, _extract_chain_ids(rec_path), _extract_chain_ids(lig_path), None
    if len(entry) == 4:
        rec_id, lig_id, template, combined = entry
        rec_path = f"processed/transformation/{template}_{rec_id}_{lig_id}_R.pdb"
        lig_path = f"processed/transformation/{template}_{rec_id}_{lig_id}_L.pdb"
        return rec_path, lig_path, _extract_chain_ids(rec_path), _extract_chain_ids(lig_path), combined
    return None, None, None, None, None


def calculate_energy(receptor_path, ligand_path, combined_path=None, left_chains=None, right_chains=None):
    try:
        import pyrosetta
        from pyrosetta.rosetta.core.import_pose import pose_from_file
        from pyrosetta.rosetta.protocols.docking import setup_foldtree
        from pyrosetta.rosetta.utility import Vector1
    except ImportError as exc:
        print(f"PyRosetta not available: {exc}")
        return "-", "-", "-"

    try:
        if combined_path is None or not os.path.exists(combined_path):
            combined_path = combine_pdb(receptor_path, ligand_path)
            if not combined_path:
                return "-", "-", "-"

        if not left_chains:
            left_chains = _extract_chain_ids(receptor_path)
        if not right_chains:
            right_chains = _extract_chain_ids(ligand_path)
        if not left_chains or not right_chains:
            print(f"Could not determine partner chains for {receptor_path} / {ligand_path}")
            return "-", "-", "-"
        partner_chains = f"{''.join(left_chains)}_{''.join(right_chains)}"

        out_name = os.path.splitext(os.path.basename(combined_path))[0]

        pyrosetta.init(extra_options="-ex1 -ex2aro -ignore_zero_occupancy false -detect_disulf false")
        pose = pose_from_file(combined_path)
        jump_num = Vector1(1)
        setup_foldtree(pose, partner_chains, jump_num)

        try:
            from pyrosetta.rosetta.protocols.docking import DockingPrepackProtocol
            prepack = DockingPrepackProtocol(jump_num[1])
            prepack.apply(pose)
        except (ImportError, AttributeError):
            _prepack_with_pack_rotamers(pose)

        prepacked_path = os.path.join(ROSETTA_DIR, f"{out_name}_0001.pdb")
        pose.dump_pdb(prepacked_path)

        try:
            from pyrosetta.rosetta.protocols.docking import DockMCMProtocol
            from pyrosetta import create_score_function
            scorefxn = create_score_function("ref2015")
            dock = DockMCMProtocol(jump_num[1], scorefxn, scorefxn)
            try:
                dock.set_local_refine(True)
            except AttributeError:
                pass
            dock.apply(pose)
        except Exception as exc:
            print(f"DockMCMProtocol failed: {exc}; falling back to perturb+min")
            _dock_local_refine_fallback(pose, jump_num[1])

        rosetta_out_path = os.path.join(STRUCTURE_DIR, f"{out_name}_refined.pdb")
        final_out_path = os.path.join(ROSETTA_DIR, f"{out_name}_refined.pdb")
        pose.dump_pdb(rosetta_out_path)

        totalscore, interaction_score = _compute_scores(pose, jump_num[1], out_name)

        if (os.path.exists(rosetta_out_path) and interaction_score != "-"
                and float(interaction_score) <= ROSETTA_INT_SCORE_THRESHOLD):
            shutil.copy2(rosetta_out_path, final_out_path)
            lines_l, lines_r = _extract_atom_lines_by_partners(final_out_path, left_chains, right_chains)
            int_res_path = f"{final_out_path}.intRes.txt"
            try:
                get_contacts_from_atom_lines(final_out_path, int_res_path, lines_l, lines_r)
            except Exception as exc:
                print(f"Failed to compute contact map post-refinement: {exc}")
            return str(totalscore), str(interaction_score), final_out_path

        return "-", "-", "-"

    except Exception as exc:
        print(f"Rosetta refinement failed: {exc}")
        return "-", "-", "-"


def _compute_scores(pose, jump_num_value, out_name):
    import pyrosetta
    score_path = os.path.join(ENERGY_DIR, "score.sc")
    renamed = os.path.join(ENERGY_DIR, f"{out_name}_score.sc")
    if os.path.exists(score_path):
        shutil.move(score_path, renamed)
        with open(renamed) as sf:
            for i, line in enumerate(sf):
                if i == 2:
                    parts = line.split()
                    if len(parts) >= 6:
                        return float(parts[1]), float(parts[5])
                    break
    try:
        from pyrosetta.rosetta.core.scoring import Interface
        scorefxn = pyrosetta.create_score_function("ref2015")
        scorefxn(pose)
        total = pose.energies().total_energy()
        interface = Interface(jump_num_value)
        interface.calculate(pose)
        intscore = interface.interface_energy(pose, scorefxn)
        return total, intscore
    except Exception:
        return "-", "-"


def _prepack_with_pack_rotamers(pose):
    from pyrosetta import create_score_function
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepackingRLT
    scorefxn = create_score_function("ref2015")
    tf = TaskFactory()
    tf.push_back(RestrictToRepackingRLT())
    packer = PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(pose)


def _dock_local_refine_fallback(pose, jump_num):
    from pyrosetta import create_score_function
    from pyrosetta.rosetta.protocols.rigid import RigidBodyPerturbMover
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover
    from pyrosetta.rosetta.core.kinematics import MoveMap
    scorefxn = create_score_function("ref2015")
    RigidBodyPerturbMover(jump_num, 3, 1).apply(pose)
    mm = MoveMap()
    mm.set_bb(False); mm.set_chi(True); mm.set_jump(True)
    MinMover(mm, scorefxn, "lbfgs_armijo_nonmonotone", 0.01, True).apply(pose)


def combine_pdb(receptor_path, ligand_path):
    try:
        base0 = os.path.splitext(os.path.basename(receptor_path))[0]
        base1 = os.path.splitext(os.path.basename(ligand_path))[0]
        combined_path = os.path.join(ROSETTA_DIR, f"{base0}_{base1}_combined.pdb")
        serial = 1
        with open(combined_path, "w") as out_f:
            for path in (receptor_path, ligand_path):
                with open(path) as fh:
                    for line in fh:
                        if line.startswith(("ATOM", "HETATM")):
                            out_f.write(f"{line[:6]}{serial:5d}{line[11:]}")
                            serial += 1
                out_f.write("TER\n")
            out_f.write("END\n")
        return combined_path
    except Exception as exc:
        print(f"combine_pdb failed: {exc}")
        return ""


def _extract_chain_ids(pdb_path):
    chain_ids = []
    seen = set()
    if not pdb_path or not os.path.exists(pdb_path):
        return chain_ids
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                ch = line[21]
                if ch not in seen:
                    seen.add(ch)
                    chain_ids.append(ch)
    return chain_ids


def _extract_atom_lines_by_partners(pdb_path, left_chains, right_chains):
    left = set(left_chains); right = set(right_chains)
    lines_0, lines_1 = [], []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")) or len(line) < 22:
                continue
            ch = line[21]
            if ch in left:
                lines_0.append(line)
            elif ch in right:
                lines_1.append(line)
    return lines_0, lines_1

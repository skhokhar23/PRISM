#!/bin/bash
#SBATCH --job-name=prism_new_batch
#SBATCH --account=cosbi
#SBATCH --partition=cosbi
#SBATCH --array=1-6%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/new_%a.log
#SBATCH --error=logs/new_%a.log

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" pairs_new.txt)
read -r PAIR TAG <<< "$LINE"

source "$(conda info --base)/bin/activate" prism_new
cd "$SLURM_SUBMIT_DIR"

echo "=== task ${SLURM_ARRAY_TASK_ID}: ${PAIR} -> processed_${TAG}"
date

if [ -d processed ]; then
  mv processed "processed_orphan_$(date +%s)"
fi

printf "Receptor,Ligand\n%s\n" "$PAIR" > "inputs_${TAG}.csv"
cat "inputs_${TAG}.csv"

python -u prism.py --inputs_csv "inputs_${TAG}.csv" --template_limit 0 --refine
RC=$?

if [ $RC -ne 0 ]; then
  echo "=== FAILED: prism.py exited with code $RC -- see traceback above"
  exit $RC
fi

mv processed "processed_${TAG}"
date

echo "=== refined structures:"
ls -1 "processed_${TAG}/rosetta_refinement/structures/" 2>/dev/null || echo "(none)"
echo "=== energies:"
cat "processed_${TAG}/rosetta_refinement/refinement_energies.txt" 2>/dev/null || echo "(none)"

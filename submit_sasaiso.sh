#!/bin/bash
# Test the SASA chain-isolation fix (src/sasa_utils.py, 2026-08-10) on the
# four already-established real pairs: brs/pcc as regression checks (both
# already correct, must stay correct), acb/ptc as the actual test (both
# currently wrong, blocked on the buried-cleft surface bug this fix
# targets). Histone pairs deliberately excluded -- this fix hasn't been
# validated yet, test on the known pairs first.
set -u

export PRISM_RUN_TAG="rsa15_f11_f10_sasaiso"

for pair in brs pcc acb ptc; do
    if [ ! -d "runs/${pair}" ]; then
        echo "SKIP ${pair}: runs/${pair} does not exist"
        continue
    fi
    JOBID=$(sbatch --parsable \
        --job-name="prism_${pair}_sasaiso" \
        --output="prism_${pair}_sasaiso_%j.log" \
        --export=ALL,PAIR="${pair}",PRISM_RUN_TAG="${PRISM_RUN_TAG}" \
        run_pair.slurm)
    echo "${pair}: submitted as job ${JOBID}"
done

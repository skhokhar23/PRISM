#!/bin/bash
# Submits one independent SLURM job per pair. Safe to run all at once --
# each pair has its own runs/<pair>/processed/ (see setup_runs.sh), so there
# is no shared-directory collision, unlike the earlier single-processed/
# scripts. Run setup_runs.sh first.
set -u

# Must match whatever the current state of the pipeline code actually is --
# update this whenever a change lands that's significant enough to need its
# own output name (see run_pair.sh). Currently: F7 + F11 + F10 + the SASA
# chain-isolation fix.
export PRISM_RUN_TAG="${PRISM_RUN_TAG:-rsa15_f11_f10_sasaiso}"

for pair in ptc brs acb pcc xl_h3h4 xl_h2ah2b dm_h3h4 dm_h2ah2b sc_asf1_h3; do
    if [ ! -d "runs/${pair}" ]; then
        echo "SKIP ${pair}: runs/${pair} does not exist -- run setup_runs.sh first"
        continue
    fi
    JOBID=$(sbatch --parsable \
        --job-name="prism_${pair}" \
        --output="prism_${pair}_%j.log" \
        --export=ALL,PAIR="${pair}",PRISM_RUN_TAG="${PRISM_RUN_TAG}" \
        run_pair.slurm)
    echo "${pair}: submitted as job ${JOBID}"
done

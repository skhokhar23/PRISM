#!/bin/bash
# Runs the new PRISM pipeline (RSA=15%, F11 contact check) for one pair,
# inside its own isolated runs/<pair>/ directory. Called by run_pair.slurm,
# not run directly. Requires setup_runs.sh to have been run first.
set -u

PAIR="${1:?usage: run_pair.sh <pair_id>}"
DIR="runs/${PAIR}"

if [ ! -d "$DIR" ]; then
    echo "FAIL: ${DIR} does not exist -- run setup_runs.sh first"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prism_new

# P22: this must NOT be set to a large value, or refinement keeps everything.
unset ROSETTA_INT_SCORE_THRESHOLD

# F10: how many top-alignment-score candidates get refined per pair, instead
# of just candidates[0]. Code default is 5; 3 has been the tested value so
# far. Raise it (export PRISM_TOP_K_REFINE=5 before calling this script)
# once there's a real timing number justifying the extra refinement cost.
export PRISM_TOP_K_REFINE="${PRISM_TOP_K_REFINE:-3}"

# Every run needs its own output name so it doesn't collide with or
# overwrite a previous run's results. This has already changed twice
# tonight (rsa15_f11 -> rsa15_f11_f10 -> ...), so it's now a required,
# explicit argument instead of a hardcoded suffix -- fails loudly if the
# caller forgets to set it, instead of silently reusing an old name.
: "${PRISM_RUN_TAG:?set PRISM_RUN_TAG before calling this script, e.g. PRISM_RUN_TAG=rsa15_f11_f10_sasaiso}"

cd "$DIR"
# Defensive: a prior run only moves processed/ out on success (see the mv at
# the bottom). If a previous attempt for this pair crashed before that, a
# stale processed/ could still be here -- clear it so this run starts clean.
rm -rf processed
echo "=== ${PAIR} : starting $(date) in $(pwd) (PRISM_TOP_K_REFINE=${PRISM_TOP_K_REFINE} PRISM_RUN_TAG=${PRISM_RUN_TAG}) ==="

python ../../prism.py --inputs_csv inputs.csv --template_limit 0 --refine \
    2>&1 | tee "../../run_${PAIR}_${PRISM_RUN_TAG}.log"
RC=${PIPESTATUS[0]}

echo "=== ${PAIR} : exit code ${RC} at $(date) ==="
if [ "$RC" -ne 0 ]; then
    echo "FAILED: ${PAIR} exited ${RC} -- see ../../run_${PAIR}_${PRISM_RUN_TAG}.log"
    exit "$RC"
fi

mv processed "../../processed_${PAIR}_${PRISM_RUN_TAG}"
echo "=== ${PAIR} : output in processed_${PAIR}_${PRISM_RUN_TAG} ==="

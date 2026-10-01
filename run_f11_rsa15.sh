#!/bin/bash
# Re-run the new PRISM pipeline on ptc with both fixes in place:
#   F7  RSA cutoff 25% -> 15%   (src/surface_extract.py)
#   F11 post-transform contact check  (src/transformation.py)
# Created 2026-08-07. acb was already run manually (move its `processed`
# output to processed_acb_rsa15_f11 separately) -- this script now covers
# only ptc, so it does not overwrite that manual run.
set -u

# P22: this must NOT be set to a large value, or refinement keeps everything.
unset ROSETTA_INT_SCORE_THRESHOLD

for PAIR in ptc; do
    OUT="processed_${PAIR}_rsa15_f11"

    if [ -d "$OUT" ]; then
        echo "SKIP $PAIR: $OUT already exists, remove or rename it first"
        continue
    fi

    rm -rf processed
    echo "=== $PAIR : starting $(date) ==="

    python prism.py --inputs_csv "inputs_${PAIR}.csv" \
                    --template_limit 0 \
                    --refine \
                    2>&1 | tee "run_${PAIR}_rsa15_f11.log"
    RC=${PIPESTATUS[0]}

    echo "=== $PAIR : exit code $RC at $(date) ==="
    if [ "$RC" -ne 0 ]; then
        echo "FAILED: $PAIR exited $RC -- see run_${PAIR}_rsa15_f11.log"
    fi

    mv processed "$OUT"
    echo "=== $PAIR : output in $OUT ==="
done

#!/bin/bash
# Set up one isolated working directory per pair under runs/<pair_id>/.
# Run ONCE from the PRISM repo root before submit_all.sh.
#
# Why: the pipeline writes everything under a relative "processed/" path, so
# two runs sharing one working directory stomp on each other (this is the
# bug that made SLURM arrays get throttled to %1 in the earlier F11 runs).
# Every path the code actually touches is relative -- confirmed by grepping
# every *_DIR constant in src/*.py -- so giving each pair its own directory,
# with read-only "templates/" and "external_tools/" symlinked back to the
# shared repo, isolates "processed/" per pair with zero pipeline code
# changes. templates/ and external_tools/ are never written to during a
# normal run (only --generate_templates writes there, which none of these
# runs use), so sharing them via symlink across concurrent jobs is safe.
set -euo pipefail

REPO_ROOT="$(pwd)"

declare -A PAIRS=(
  [ptc]="2ptcE,2ptcI"
  [brs]="1brsA,1brsD"
  [acb]="1acbE,1acbI"
  [pcc]="2pccA,2pccB"
  [xl_h3h4]="1kx5A,1kx5B"
  [xl_h2ah2b]="1kx5C,1kx5D"
  [dm_h3h4]="2pyoA,2pyoB"
  [dm_h2ah2b]="2pyoC,2pyoD"
  [sc_asf1_h3]="4eo5A,4eo5B"
)
# acb and pcc are new here -- their earlier F7+F11 runs used the old shared
# top-level processed/ (acb manually, before this isolation system existed;
# pcc was never run with F11 at all, only the RSA-only baseline). Folding
# them into runs/ now so every F10 pair goes through the same isolated path
# as the histone pairs, rather than mixing two run mechanisms in one batch.
# Safe to re-run this whole script -- mkdir -p / ln -sfn / the inputs.csv
# write are all idempotent, so brs/ptc/the histone pairs are untouched.

for pair in "${!PAIRS[@]}"; do
    dir="runs/${pair}"
    mkdir -p "$dir"
    ln -sfn "${REPO_ROOT}/templates" "$dir/templates"
    ln -sfn "${REPO_ROOT}/external_tools" "$dir/external_tools"
    IFS=',' read -r rec lig <<< "${PAIRS[$pair]}"
    printf "Receptor,Ligand\n%s,%s\n" "$rec" "$lig" > "$dir/inputs.csv"
    echo "set up ${dir}  (receptor=${rec} ligand=${lig})"
done

echo
echo "Verify before submitting anything:"
echo "  ls -la runs/*/templates runs/*/external_tools   # symlinks should resolve, not show broken red text"
echo "  cat runs/*/inputs.csv                            # confirm receptor/ligand pairs are correct"

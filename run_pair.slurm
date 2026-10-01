#!/bin/bash
#SBATCH --account=cosbi
#SBATCH --partition=cosbi
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
# job-name and output are set at submit time by submit_all.sh (sbatch
# --job-name/--output/--export), since #SBATCH directives can't expand the
# $PAIR shell variable.

bash run_pair.sh "$PAIR"

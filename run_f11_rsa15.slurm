#!/bin/bash
#SBATCH --account=cosbi
#SBATCH --partition=cosbi
#SBATCH --job-name=prism_f11_ptc
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=prism_f11_ptc_%j.log

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prism_new
bash run_f11_rsa15.sh

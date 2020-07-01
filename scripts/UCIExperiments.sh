#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name "DAG-NF-UCI"
#SBATCH --mem=6000M
#SBATCH --time=144:00:00
#SBATCH --output="DAG-NF-UCI-%j.out"

args=("$@")
supp_args=$(printf "%s "  "${args[@]}")

python UCIExperiments.py $supp_args
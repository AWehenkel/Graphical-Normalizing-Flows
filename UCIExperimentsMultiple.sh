#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name "DAG-NF-UCI"
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=144:00:00
#SBATCH --output="DAG-NF-UCI-%j.out"

args=("$@")
supp_args=$(printf "%s "  "${args[@]}")

source activate UMNN
for config in "${args[@]}"
do
    srun python UCIExperiments.py -load_config $config &
done

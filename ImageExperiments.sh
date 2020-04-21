#!/bin/bash
#SBATCH --gres=gpu:$1
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name "DAG-NF-$2"
#SBATCH --mem=12G
#SBATCH --time=144:00:00
#SBATCH --output="DAG-NF-$2-%j.out"

args=("$@")
supp_args=$(printf "%s "  "${args[@]:2}")

source activate UMNN
python CIFAR10Experiments.py -dataset $2 -nb_gpus $1 $supp_args
#!/bin/bash
#SBATCH --gres=gpu:$1
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name "DAG-NF-CIFAR10"
#SBATCH --mem=12G
#SBATCH --time=144:00:00
#SBATCH --output="DAG-NF-CIFAR10-%j.out"

args=("$@")
supp_args=$(printf "%s "  "${args[@]:1}")

source activate UMNN
python CIFAR10Experiments.py -nb_gpus
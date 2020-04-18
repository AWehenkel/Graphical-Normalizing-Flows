#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=2
#SBATCH --job-name "DAG-NF-CIFAR10"
#SBATCH --mem=12G
#SBATCH --time=144:00:00
#SBATCH --output="DAG-NF-CIFAR10-%j.out"
int_net=$5
int_net=(${int_net//,/ })
emb_net=$6
emb_net=(${emb_net//,/ })
args=("$@")
supp_args=$(printf "%s "  "${args[@]:6}")

source activate UMNN
python CIFAR10Experiments.py -b_size $1 -nb_epoch $2 -nb_steps_dual $3 -l1 $4 -int_net ${int_net[*]} -emb_net ${emb_net[*]} $supp_args
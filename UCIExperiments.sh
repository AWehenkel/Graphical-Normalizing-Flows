#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name "DAG-NF-UCI"
#SBATCH --mem=6000M
#SBATCH --time=144:00:00
#SBATCH --output="DAG-NF-UCI-%j.out"
int_net=$6
int_net=(${int_net//,/ })
emb_net=$7
emb_net=(${emb_net//,/ })
args=("$@")
supp_args=$(printf "%s "  "${args[@]:7}")

source activate UMNN
python UCIExperiments.py -dataset $1 -b_size $2 -nb_epoch $3 -nb_steps_dual $4 -l1 $5 -int_net ${int_net[*]} -emb_net ${emb_net[*]} $supp_args
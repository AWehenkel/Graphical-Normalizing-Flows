#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name "DAG-NF-UCI"
#SBATCH --mem=6000M
#SBATCH --time=72:00:00
#SBATCH --output="DAG-NF-UCI-%j.out"
int_net=$5
int_net=(${int_net//,/ })
emb_net=$6
emb_net=(${int_net//,/ })
python UCIExperiments.py -dataset $1 -nb_steps_dual $2 -max_l1 $3 -nb_epoch $4 -int_net ${int_net[*]} -emb_net ${emb_net[*]} -b_size $7
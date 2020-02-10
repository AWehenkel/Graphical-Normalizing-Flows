#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name "UMNN-DAG-NF-UCI"
#SBATCH --mem=6000M
#SBATCH --time=72:00:00
#SBATCH --output="UMNN-DAG-NF-UCI-%j.out"
int_net=$4
int_net=(${int_net//,/ })
emb_net=$5
emb_net=(${emb_net//,/ })
source activate UMNN
python UCIExperiments.py -dataset $1 -b_size $2 -nb_epoch $3 -int_net ${int_net[*]} -emb_net ${emb_net[*]} -UMNN_MAF
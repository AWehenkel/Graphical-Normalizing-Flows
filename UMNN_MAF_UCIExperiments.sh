#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name "DAG-NF-UCI"
#SBATCH --mem=6000M
#SBATCH --time=20:00:00
IN=$5
arrIN=(${IN//,/ })
python UCIExperiments.py -dataset $1 -nb_steps_dual $2 -max_l1 $3 -nb_epoch $4 -network ${arrIN[*]} -b_size $6 -UMNN_MAF
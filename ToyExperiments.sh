#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --job-name "DAG-NF-Toy"
#SBATCH --mem=6000M
#SBATCH --time=20:00:00
conda activate UMNN
python ToyExperiments.py -dataset $1 -nb_flow $2 $3 $4 $5
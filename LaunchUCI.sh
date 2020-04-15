#!/usr/bin/env bash
sbatch UCIExperiments.sh power 2500 10000 20 .0 150,150,150,150 100,100,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 1 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding
sbatch UCIExperiments.sh gas 10000 10000 20 2.0 100,100,100 100,100,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 1 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding
sbatch UCIExperiments.sh hepmass 100 10000 20 .0 200,200,200,200 512,512,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 1 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding
sleep 10
sbatch UCIExperiments.sh power 2500 10000 20 .0 150,150,150,150 100,100,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 1 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding -UMNN_MAF
sbatch UCIExperiments.sh gas 10000 10000 20 2.0 100,100,100 100,100,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 1 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding -UMNN_MAF
sbatch UCIExperiments.sh hepmass 100 10000 20 .0 200,200,200,200 512,512,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 1 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding -UMNN_MAF
sleep 10
sbatch UCIExperiments.sh power 2500 10000 20 .0 100,100,100 150,150,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 2 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding
sbatch UCIExperiments.sh gas 10000 10000 20 2.0 200,200,200 100,100,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 2 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding
sleep 10
sbatch UCIExperiments.sh power 2500 10000 20 .0 100,100,100 150,150,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 2 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding -UMNN_MAF
sbatch UCIExperiments.sh gas 10000 10000 20 2.0 200,200,200 100,100,30 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 2 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding -UMNN_MAF
sleep 10
sbatch UCIExperiments.sh power 10000 10000 20 .0 50,50,50 100,100,2 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 3 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-4 -hot_encoding -linear_net
sbatch UCIExperiments.sh gas 10000 10000 20 .0 200,200,200 300,300,300,300,2 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 3 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding -linear_net
sbatch UCIExperiments.sh hepmass 100 10000 20 .0 100,100,100,100 512,512,2 -min_pre_heating_epochs 0 -nb_steps 20 -nb_flow 3 -gumble_T .5  -weight_decay 1e-6 -learning_rate 1e-3 -hot_encoding -linear_net

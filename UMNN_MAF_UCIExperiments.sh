#!/bin/bash

python UCIExperiments.py -dataset $1 -b_size $2 -nb_epoch $3 -int_net ${int_net[*]} -emb_net ${emb_net[*]} -UMNN_MAF
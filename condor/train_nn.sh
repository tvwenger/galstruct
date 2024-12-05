#!/bin/bash

python learn_likelihood.py \
    --density_estimator $1 \
    -n $2 \
    --features $3 \
    --layers $4 \
    --training_batch_size $5 \
    --fixed Zsun 5.5 \
    --fixed roll 0.0 \
    --overwrite \
    nn_${1}_${2}n_${3}f_${4}l_${5}t.pkl

#!/bin/bash

python learn_likelihood.py \
    --density_estimator $1 \
    -n $2 \
    --features $3 \
    --layers $4 \
    --training_batch_size $5 \
    --fixed R0 8.15 \
    --fixed Upec 6.0 \
    --fixed Vpec 0.0 \
    --fixed Zsun 5.5 \
    --fixed roll 0.0 \
    --overwrite \
    nn.pkl

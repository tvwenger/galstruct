#!/bin/bash

python learn_likelihood.py \
    --density_estimator $1 \
    -n $2 \
    --features $3 \
    --layers $4 \
    --training_batch_size $5 \
    --fixed R0 8.15 \
    --fixed Usun 10.6 \
    --fixed Vsun 10.7 \
    --fixed Wsun 7.6 \
    --fixed Upec 6.0 \
    --fixed Vpec -4.3 \
    --fixed a2 0.96 \
    --fixed a3 1.62 \
    --fixed Zsun 5.5 \
    --fixed roll 0.0 \
    --overwrite \
    nn.pkl

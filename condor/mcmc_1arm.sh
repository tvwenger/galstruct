#!/bin/bash

tmpdir=`mktemp -d`
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python mcmc_posterior.py \
    synthetic \
    mcmc.pkl \
    $1 \
    -n 1_000 \
    --chains 8 \
    --ntune 1_000 \
    --ninit 100_000 \
    --target_accept 0.8 \
    --num_data 200 \
    --spiral_params 1.0 1.26 0.24 5.0 0.5 0.1 \
    --num_spirals 1 \
    --fixed q 1.0 \
    --fixed R0 8.15 \
    --fixed Upec 6.0 \
    --fixed Vpec 0.0 \
    --fixed Zsun 5.5 \
    --fixed roll 0.0

#!/bin/bash

tmpdir=`mktemp -d`
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python mcmc_posterior.py \
    synthetic \
    mcmc.pkl \
    $1 \
    -n 1_000 \
    --chains 12 \
    --ntune 1_000 \
    --ninit 100_000 \
    --target_accept 0.85 \
    --num_data 2000 \
    --fixed Zsun 5.5 \
    --fixed roll 0.0

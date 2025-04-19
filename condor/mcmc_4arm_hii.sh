#!/bin/bash

tmpdir=`mktemp -d`
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python mcmc_posterior.py \
    hii_data_sub.csv \
    mcmc.pkl \
    $1 \
    -n 1_000 \
    --chains 12 \
    --ntune 1_000 \
    --ninit 100_000 \
    --target_accept 0.8 \
    --fixed R0 8.15 \
    --fixed Upec 6.0 \
    --fixed Vpec 0.0 \
    --fixed Zsun 5.5 \
    --fixed roll 0.0

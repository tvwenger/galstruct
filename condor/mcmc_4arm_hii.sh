#!/bin/bash

tmpdir=`mktemp -d`
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python mcmc_posterior.py \
    hii_data.csv \
    mcmc.pkl \
    $1 \
    -n 500 \
    --chains 12 \
    --ntune 500 \
    --ninit 10_000 \
    --target_accept 0.80 \
    --fixed Zsun 5.5 \
    --fixed roll 0.0

#!/bin/bash

tmpdir=`mktemp -d`
PYTENSOR_FLAGS="base_compiledir=$tmpdir" python mcmc_posterior.py \
    synthetic \
    mcmc.pkl \
    $1 \
    --init "advi+adapt_diag" \
    -n 1_000 \
    --chains 8 \
    --ntune 1_000 \
    --ninit 100_000 \
    --target_accept 0.8 \
    --num_data 2000 \
    --fixed R0 8.15 \
    --fixed Usun 10.6 \
    --fixed Vsun 10.7 \
    --fixed Wsun 7.6 \
    --fixed Upec 6.0 \
    --fixed Vpec -4.3 \
    --fixed a2 0.96 \
    --fixed a3 1.62 \
    --fixed Zsun 5.5 \
    --fixed roll 0.0

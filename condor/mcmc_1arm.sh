#!/bin/bash

python mcmc_posterior.py \
    synthetic \
    mcmc.pkl \
    nn_nsf_2_097_152n_64f_8l_1_024t.pkl \
    -n 1_000 \
    --chains 12 \
    --ntune 1_000 \
    --ninit 100_000 \
    --target_accept 0.85 \
    --num_data 200 \
    --spiral_params 1.0 1.26 0.24 5.0 0.5 0.1 \
    --num_spirals 1 \
    --fixed q 1.0 \
    --fixed Zsun 5.5 \
    --fixed roll 0.0

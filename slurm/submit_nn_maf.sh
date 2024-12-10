#!/bin/bash
#SBATCH --chdir="/home/twenger/galstruct"
#SBATCH --job-name="train_nn_maf"
#SBATCH --output="logs/%x.%j.%N.out"
#SBATCH --error="logs/%x.%j.%N.err"
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --export=ALL
#SBATCH --time 72:00:00

eval "$(conda shell.bash hook)"
conda activate galstruct

# This is really inefficient because some networks converge early, yet we
# hog the resources until all are finished. Instead, we should have a script
# that submits each job separately.

srun --output logs/%x.%j.%N.nn_10k_maf_50f_5l_50t.out \
    --error logs/%x.%j.%N.nn_10k_maf_50f_5l_50t.err \
    --exclusive -N1 -n1 \
    python galstruct/learn_likelihood.py \
    -n 10_000 \
    --density_estimator maf \
    --features 50 \
    --layers 5 \
    --training_batch_size 50 \
    --overwrite \
    nn_10k_maf_50f_5l_50t.pkl &

srun --output logs/%x.%j.%N.nn_100k_maf_50f_5l_50t.out \
    --error logs/%x.%j.%N.nn_100k_maf_50f_5l_50t.err \
    --exclusive -N1 -n1  \
    python galstruct/learn_likelihood.py \
    -n 100_000 \
    --density_estimator maf \
    --features 50 \
    --layers 5 \
    --training_batch_size 50 \
    --overwrite \
    nn_100k_maf_50f_5l_50t.pkl & 

srun --output logs/%x.%j.%N.nn_1m_maf_50f_5l_50t.out \
    --error logs/%x.%j.%N.nn_1m_maf_50f_5l_50t.err \
    --exclusive -N1 -n1  \
    python galstruct/learn_likelihood.py \
    -n 1_000_000 \
    --density_estimator maf \
    --features 50 \
    --layers 5 \
    --training_batch_size 50 \
    --overwrite \
    nn_1m_maf_50f_5l_50t.pkl &

srun --output logs/%x.%j.%N.nn_10m_maf_50f_5l_50t.out \
    --error logs/%x.%j.%N.nn_10m_maf_50f_5l_50t.err \
    --exclusive -N1 -n1 \
    python galstruct/learn_likelihood.py \
    -n 10_000_000 \
    --density_estimator maf \
    --features 50 \
    --layers 5 \
    --training_batch_size 50 \
    --overwrite \
    nn_10m_maf_50f_5l_50t.pkl &

wait
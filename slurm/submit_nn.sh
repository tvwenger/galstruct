#!/bin/bash
#SBATCH --chdir="/home/twenger/galstruct"
#SBATCH --job-name="train_nn"
#SBATCH --output="logs/%x.%j.%N.out"
#SBATCH --error="logs/%x.%j.%N.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=twenger2@wisc.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --time 72:00:00

eval "$(conda shell.bash hook)"
conda activate galstruct

srun python python galstruct/learn_likelihood.py \
    -n 10_000 \
    --density_estimator maf \
    --features 50 \
    --layers 5 \
    --sim_batch_size 1 \
    --training_batch_size 50 \
    --overwrite \
    nn_10k_maf_50f_5l_50t.pkl

srun python python galstruct/learn_likelihood.py \
    -n 100_000 \
    --density_estimator maf \
    --features 50 \
    --layers 5 \
    --sim_batch_size 1 \
    --training_batch_size 50 \
    --overwrite \
    nn_100k_maf_50f_5l_50t.pkl

srun python python galstruct/learn_likelihood.py \
    -n 1_000_000 \
    --density_estimator maf \
    --features 50 \
    --layers 5 \
    --sim_batch_size 1 \
    --training_batch_size 50 \
    --overwrite \
    nn_1m_maf_50f_5l_50t.pkl

srun python python galstruct/learn_likelihood.py \
    -n 10_000_000 \
    --density_estimator maf \
    --features 50 \
    --layers 5 \
    --sim_batch_size 1 \
    --training_batch_size 50 \
    --overwrite \
    nn_10m_maf_50f_5l_50t.pkl
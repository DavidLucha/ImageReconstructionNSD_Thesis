#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=VAE_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10000
#SBATCH -o VAE_test_output.txt
#SBATCH -e VAE_test_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uqdlucha@uq.edu.au

module load anaconda
module load gcc/12.1.0
module load cuda/11.3.0
module load mvapich2
# module load openmpi3

source activate /scratch/qbi/uqdlucha/python/dvaegan

RUN_NAME=$(date +%Y%m%d-%H%M%S)

# Study 1
# Pretrain one network. Used in both GOD and NSD.
# echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
# srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --run_name RUN2_1_pretrain_$RUN_NAME --lr 0.0001 --gamma 1.0 --equilibrium_game y --backprop_method clip --num_workers 2 --epochs 100 --dataset both --loss_method Maria --optim_method RMS --seed 277603 --message "Proper run, using pretrain on 30k into stage 1 with only NSD. Pretrain."
# echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

echo "Running stage 1 at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage1_vae.py --run_name VAE_test_$RUN_NAME --lr 0.003 --backprop_method trad --num_workers 2 --epochs 200 --dataset NSD --seed 277603 --message "Testing VAE using the vqvae as settings"
echo "Stage 1 complete at $(date +%Y%m%d-%H%M%S)"
#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=WAE_Maria_New_Test
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=20000
#SBATCH -o WAE_Maria_New_Test_output.txt
#SBATCH -e WAE_Maria_New_Test_error.txt
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
echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain_WAE.py --run_name WAE_Maria_New_Test_$RUN_NAME --disc_loss Both --WAE_loss Both --lambda_WAE 1 --batch_size 64 --latent_dims 1024 --num_workers 2 --epochs 120 --dataset NSD --seed 277603 --message "Maria loss and weighting but with new BCE and other calcs"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
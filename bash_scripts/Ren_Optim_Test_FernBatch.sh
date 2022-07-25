#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=fern_test
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --mem=50000
#SBATCH -o fern_test_output.txt
#SBATCH -e fern_test_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:2
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
srun --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --num_workers 4 --epochs 100 --dataset both --run_name 003_1_$RUN_NAME --loss_method Ren --optim_method Adam --equilibrium_game y --message "Testing with Fernanda's settings"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

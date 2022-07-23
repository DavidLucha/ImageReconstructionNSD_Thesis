#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=ren_pretrain_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=20000
#SBATCH -o ren_pretrain_test_output.txt
#SBATCH -e ren_pretrain_test_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load anaconda
module load gcc/12.1.0
module load cuda/11.3.0
module load mvapich2

source activate /scratch/qbi/uqdlucha/python/dvaegan

RUN_NAME=$(date +%Y%m%d-%H%M%S)

# Study 1
# Pretrain one network. Used in both GOD and NSD.
echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --num_workers 2 --epochs 50 --dataset both --run_name 003_$RUN_NAME --loss_method Ren --optim_method Adam --message "Pretraining with Ren and Adam optim"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

echo "Running pretrain 2 at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --num_workers 8 --lr 0.001 --epochs 50 --dataset both --run_name 001_$RUN_NAME --loss_method Ren --optim_method Adam --message "Pretraining with Ren and Adam optim, less lr and more num_workers as test"
echo "Pretrain 2 complete at $(date +%Y%m%d-%H%M%S)"

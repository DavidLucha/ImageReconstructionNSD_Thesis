#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=loss_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10000
#SBATCH -o loss_test_output.txt
#SBATCH -e loss_test_error.txt
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
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --num_workers 2 --epochs 40 --dataset both --run_name 02_$RUN_NAME --gamma 0.2 --loss_method Ren --optim_method Adam --equilibrium_game n --message "gamma at 0.2, nw 2"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --num_workers 4 --epochs 40 --dataset both --run_name 04_$RUN_NAME --gamma 0.4 --loss_method Ren --optim_method Adam --equilibrium_game n --message "gamma at 0.4, nw 4"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --num_workers 6 --epochs 40 --dataset both --run_name 06_$RUN_NAME --gamma 0.6 --loss_method Ren --optim_method Adam --equilibrium_game n --message "gamma at 0.6, nw 6"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --num_workers 1 --epochs 40 --dataset both --run_name 08_$RUN_NAME --gamma 0.8 --loss_method Ren --optim_method Adam --equilibrium_game n --message "gamma at 0.8, nw 1"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
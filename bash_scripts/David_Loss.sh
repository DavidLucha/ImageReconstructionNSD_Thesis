#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=david_loss_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10000
#SBATCH -o david_loss_test_output.txt
#SBATCH -e david_loss_test_error.txt
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

# Pretrain one network. Used in both GOD and NSD.
echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --num_workers 2 --epochs 40 --dataset both --run_name David_Gamma1_$RUN_NAME --lr 0.0001 --loss_method David --optim_method RMS --equilibrium_game y --message "David loss, nw 2"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

# Pretrain one network. Used in both GOD and NSD.
echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --num_workers 2 --epochs 40 --dataset both --run_name David_Gamma5_$RUN_NAME --lr 0.0001 --loss_method David --gamma 5.0 --optim_method RMS --equilibrium_game y --message "David loss, gamma 5, nw 2"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"


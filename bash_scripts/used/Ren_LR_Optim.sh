#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=0001_gamma_LR_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10000
#SBATCH -o 0001_gamma_LR_test_output.txt
#SBATCH -e 0001_gamma_LR_test_error.txt
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
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --run_name LR_0001_gamma_5_$RUN_NAME --d_scale 0.0 --g_scale 0.0 --lr 0.0001 --gamma 5.0 --equilibrium_game n --backprop_method new --num_workers 8 --epochs 40 --dataset both --run_name Testing_Update_$RUN_NAME --loss_method Ren_Alt --optim_method Adam --message "new ren loss function recon weighting (gamma) at 5, lr at .0001, using adam"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

# Study 1
# Pretrain one network. Used in both GOD and NSD.
echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --run_name LR_0001_gamma_10_$RUN_NAME --d_scale 0.0 --g_scale 0.0 --lr 0.0001 --gamma 10.0 --equilibrium_game n --backprop_method new --num_workers 8 --epochs 40 --dataset both --run_name Testing_Update_$RUN_NAME --loss_method Ren_Alt --optim_method Adam --message "new ren loss function recon weighting (gamma) at 5, lr at .0001, using adam"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
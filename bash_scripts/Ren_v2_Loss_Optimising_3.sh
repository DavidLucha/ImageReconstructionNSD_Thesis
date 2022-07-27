#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=Ren_v2_loss_3_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10000
#SBATCH -o Ren_v2_loss_3_test_output.txt
#SBATCH -e Ren_v2_loss_3_test_error.txt
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
# srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --d_scale 0.0 --g_scale 0.0 --lr 0.0001 --gamma 1.0 --equilibrium_game n --backprop_method new --num_workers 2 --epochs 75 --dataset both --run_name Ren_Scalar_MSE_noEqui_Comb_$RUN_NAME --loss_method Ren_Alt --optim_method Combined --message "new ren loss function recon weighting (gamma) at 1, lr at .0001, no equilibrium, using combined encdec optimizers with adam"
# echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

# echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
# srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --d_scale 0.0 --g_scale 0.0 --lr 0.0001 --gamma 1.0 --equilibrium_game n --backprop_method new --num_workers 2 --epochs 75 --dataset both --run_name Ren_Scalar_MSE_noEqui_RMS_$RUN_NAME --loss_method Ren_Alt --optim_method RMS --message "new ren loss function recon weighting (gamma) at 1, lr at .0001, no equilibrium, using separate RMS optims"
# echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --d_scale 0.0 --g_scale 0.0 --lr 0.0001 --gamma 1.0 --equilibrium_game y --backprop_method new --num_workers 2 --epochs 75 --dataset both --run_name Ren_Scalar_MSE_wEqui_RMS_$RUN_NAME --loss_method Ren_Alt --optim_method RMS --message "new ren loss function recon weighting (gamma) at 1, lr at .0001, with equilibrium"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

# echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
# srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --d_scale 0.0 --g_scale 0.0 --lr 0.0001 --gamma 2.5 --equilibrium_game n --backprop_method new --num_workers 2 --epochs 75 --dataset both --run_name Ren_Scalar_weightedMSE_noEqui_$RUN_NAME --loss_method Ren_Alt --optim_method RMS --message "new ren loss function recon weighting (gamma) at 2.5, lr at .0001, with equilibrium"
# echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

# echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
# srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --lr 0.0003 --gamma 1.0 --equilibrium_game y --backprop_method new --num_workers 2 --epochs 100 --dataset both --run_name Ren_Scalar_MSE_wEqui_$RUN_NAME --loss_method Ren_Alt --optim_method RMS --message "new ren loss function recon weighting (gamma) at 1, lr at .0003, with equilibrium"
# echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
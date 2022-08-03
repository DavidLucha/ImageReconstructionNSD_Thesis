#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=pretrain_stage1_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=20000
#SBATCH -o pretrain_stage1_test_output.txt
#SBATCH -e pretrain_stage1_test_error.txt
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
echo "Running pretrain"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --epochs 100 --dataset both --run_name 0_$RUN_NAME --message "Pretraining 100 epochs - first proper test"
echo "Pretrain complete"
echo "Starting stage 1"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage1.py --epochs 200 --dataset both --run_name 1_$RUN_NAME --pretrained_net 0_$RUN_NAME --load_epoch 99 --message "First test of 200 epoch stage 1"
echo "Both stages complete"

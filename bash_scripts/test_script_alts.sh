#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=dvaegan_alt_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=5000
#SBATCH -o tensor_out_alts.txt
#SBATCH -e tensor_error_alts.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load anaconda
module load gcc/12.1.0
module load cuda/11.3.0
module load mvapich2

source activate /scratch/qbi/uqdlucha/python/dvaegan

RUN_NAME=$(date +%Y%m%d-%H%M%S)
# 20220712_172256
# Don't need the network train anymore. Just carry the RUN_NAME var throughout, and make the strings in Python. TODO: THIS <<<

# Study 1
# Pretrain one network. Used in both GOD and NSD.
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --data_root /scratch/qbi/uqdlucha/datasets/ --epochs 20 --dataset GOD --run_name 1$RUN_NAME --message "trying a batch size of 128" --batch_size 128
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --data_root /scratch/qbi/uqdlucha/datasets/ --epochs 20 --dataset GOD --run_name 2$RUN_NAME --message "trying a batch size of 32" --batch_size 32
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --data_root /scratch/qbi/uqdlucha/datasets/ --epochs 20 --dataset GOD --run_name 3$RUN_NAME --message "trying with David style loss" --loss_method David
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --data_root /scratch/qbi/uqdlucha/datasets/ --epochs 20 --dataset GOD --run_name 4$RUN_NAME --message "trying the original vaegan loss, bs 64" --loss_method Orig
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --data_root /scratch/qbi/uqdlucha/datasets/ --epochs 20 --dataset GOD --run_name 5$RUN_NAME --message "Trying the ren style loss, bs 64" --loss_method Ren
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --data_root /scratch/qbi/uqdlucha/datasets/ --epochs 20 --dataset GOD --run_name 6$RUN_NAME --message "Ren style sloss with .003 lr" --loss_method Ren --lr 0.003
echo Done!

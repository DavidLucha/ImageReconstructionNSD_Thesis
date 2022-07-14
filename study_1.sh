#!/bin/bash
#SBATCH -N 3
#SBATCH --job-name=jake_test_tensor_gpu
#SBATCH -n 3
#SBATCH -c 1
#SBATCH --mem=50000
#SBATCH -o tensor_out.txt
#SBATCH -e tensor_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:2

module load gnu7/7.2.0
module load cuda/9.2.148.1
module load anaconda/3.6
module load mvapich2
module load pmix/1.2.3

srun -n2 --mpi=pmi2 python3.6 benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py --num_gpus=2 --model resnet50 --batch_size 128

RUN_NAME=$(date +%Y%m%d-%H%M%S)
# 20220712_172256
# Don't need the network train anymore. Just carry the RUN_NAME var throughout, and make the strings in Python. TODO: THIS <<<

# Study 1
# Pretrain one network. Used in both GOD and NSD.
srun python3.6 scripts/deepReconPyTorch/pretrain.py --epochs 20 --dataset GOD --run_name $RUN_NAME

# Train Stage 1
# Pretrain on GOD images, pretrain on NSD images, non-pretrain on NSD images.
# TODO: Figure out how to get the names flowing across
# TODO: Make bool for pretrain - then create string using RUN_NAME
# Maybe a run_name
srun python3.6 scripts/deepReconPyTorch/train_stage1.py --epochs 20 --dataset GOD --run_name $RUN_NAME --pretrain True --load_epoch 20
srun python3.6 scripts/deepReconPyTorch/train_stage1.py --epochs 20 --dataset NSD --run_name $RUN_NAME --pretrain True --load_epoch 20
srun python3.6 scripts/deepReconPyTorch/train_stage1.py --epochs 20 --dataset NSD --run_name $RUN_NAME
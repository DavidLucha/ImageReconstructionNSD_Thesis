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

# Study 1
# Pretrain one network. Used in both GOD and NSD.
srun python3.6 scripts/deepReconPyTorch/pretrain.py --epochs 20 --dataset GOD

# Train Stage 1
# Pretrain on GOD images, pretrain on NSD images, non-pretrain on NSD images.
srun
srun
srun
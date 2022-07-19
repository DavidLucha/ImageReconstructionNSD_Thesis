#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=dvaegan_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=5000
#SBATCH -o tensor_out.txt
#SBATCH -e tensor_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load anaconda/3.6
source activate /opt/ohpc/pub/apps/pytorch_1.10
module load gnu/5.4.0
module load cuda/10.0.130
module load mvapich2

RUN_NAME=$(date +%Y%m%d-%H%M%S)
# 20220712_172256
# Don't need the network train anymore. Just carry the RUN_NAME var throughout, and make the strings in Python. TODO: THIS <<<

# Study 1
# Pretrain one network. Used in both GOD and NSD.
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --data_root /scratch/qbi/uqdlucha/datasets/ --epochs 20 --dataset GOD --run_name $RUN_NAME --message cuda_test
echo Done!

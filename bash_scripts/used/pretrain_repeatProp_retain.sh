#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=pretrain_repeatProp_retain_0001_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10000
#SBATCH -o pretrain_repeatProp_retain_0001_output.txt
#SBATCH -e pretrain_repeatProp_retain_0001_error.txt
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
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --run_name Pretrain_repeatProp_retain_0001_$RUN_NAME --lr 0.0001 --gamma 1.0 --equilibrium_game y --backprop_method no --num_workers 2 --epochs 150 --dataset both --loss_method David --optim_method RMS --message "This has the repeated foward passes like the OG VAEGAN I did (retain)"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
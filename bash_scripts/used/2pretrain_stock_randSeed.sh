#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=seedTest2_pretrain_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=10000
#SBATCH -o seedTest2_pretrain_output.txt
#SBATCH -e seedTest2_pretrain_error.txt
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
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --run_name seedTest2_pretrain_$RUN_NAME --lr 0.0001 --gamma 1.0 --equilibrium_game y --backprop_method trad --num_workers 2 --epochs 150 --dataset both --loss_method Maria --optim_method RMS --seed 0 --message "Testing seeds"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
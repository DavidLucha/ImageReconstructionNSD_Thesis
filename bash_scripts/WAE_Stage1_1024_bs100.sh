#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=WAE_1024_Stage1_bs100
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=20000
#SBATCH -o WAE_1024_Stage1_bs100_output.txt
#SBATCH -e WAE_1024_Stage1_bs100_error.txt
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
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain_WAE.py --run_name WAE_1024_Stage1_bs100_$RUN_NAME --batch_size 100 --latent_dims 1024 --num_workers 2 --epochs 100 --dataset NSD --seed 277603 --message "Final run through of pretrain (stage 1)"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
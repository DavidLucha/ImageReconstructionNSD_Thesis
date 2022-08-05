#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=1T50C
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=50000
#SBATCH -o 1T50C_output.txt
#SBATCH -e 1T50C_error.txt
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
srun -N 1 -p gpu --gres=gpu:1 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/pretrain.py --run_name 1T50C_$RUN_NAME --lr 0.0001 --gamma 1.0 --equilibrium_game y --backprop_method clip --num_workers 1 --epochs 100 --dataset both --loss_method Maria --optim_method RMS --seed 277603 --message "1 worker, 1 task, 50 CPU"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

# echo "Running stage 1 at $(date +%Y%m%d-%H%M%S)"
# srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage1.py --run_name RUN2_1_stage1_$RUN_NAME --pretrained_net RUN2_1_pretrain_20220802-225243  --load_epoch 99 --lr 0.0001 --backprop_method clip --num_workers 2 --epochs 200 --dataset NSD --seed 277603 --message "Proper run, using pretrain on 30k into stage 1 with only NSD. Stage 1."
# echo "Stage 1 complete at $(date +%Y%m%d-%H%M%S)"
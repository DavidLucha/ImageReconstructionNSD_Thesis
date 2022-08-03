#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=stage2_stage3_test
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=5000
#SBATCH -o stage2_stage3_test_output.txt
#SBATCH -e stage2_stage3_test_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1

module load anaconda
module load gcc/12.1.0
module load cuda/11.3.0
module load mvapich2

source activate /scratch/qbi/uqdlucha/python/dvaegan

# Update this
ST_1="1_20220723-010446"

# Study 1
# Testing stage 2/3 on GOD.
RUN_NAME=$(date +%Y%m%d-%H%M%S)
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2.py --epochs 2 --dataset GOD --subject 3 --vox_res 3mm --set_size max --run_name 2_$RUN_NAME --pretrained_net $ST_1 --load_epoch 1 --message "stage 2 test on GOD (subj 3)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage3.py --epochs 2 --dataset GOD --subject 3 --vox_res 3mm --set_size max --run_name 3_$RUN_NAME --stage_2_trained 2_$RUN_NAME --load_epoch 1 --message "stage 3 test on GOD (subj 3), loading from stage 2"

# Testing stage 2/3 on NSD.
RUN_NAME_2=$(date +%Y%m%d-%H%M%S)
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2.py --epochs 2 --dataset NSD --subject 6 --vox_res 1.8mm --set_size max --run_name 2_$RUN_NAME_2 --pretrained_net $ST_1 --load_epoch 1 --message "stage 2 test on NSD (subj 6)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage3.py --epochs 2 --dataset NSD --subject 6 --vox_res 1.8mm --set_size max --run_name 3_$RUN_NAME_2 --stage_2_trained 2_$RUN_NAME_2 --load_epoch 1 --message "stage 3 test on NSD (subj 6), loading from stage 2"

#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=WAE_fixST2_1024_0001_Test
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=20000
#SBATCH -o WAE_fixST2_1024_0001_Test_output.txt
#SBATCH -e WAE_fixST2_1024_0001_Test_error.txt
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
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2_WAE_iter.py --run_name WAE_fixST2_1024_0001_Test_$RUN_NAME --load_from pretrain --pretrained_net WAE_1024_Test_20220806-125520 --load_epoch 150 --disc_loss Both --WAE_loss Both --lambda_WAE 1 --batch_size 64 --lr_enc 0.0001 --lr_disc 0.00005 --valid_shuffle True --latent_dims 1024 --vox_res 1.8mm --set_size max --subject 1 --ROI VC --num_workers 2 --epochs 110 --iters 60000 --dataset NSD --seed 277603 --message "WAE test of stage 2 with the (i think) corrected discriminator function."
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
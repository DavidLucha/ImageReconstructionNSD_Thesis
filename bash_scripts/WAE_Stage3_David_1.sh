#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=WAE_Stage3_David_1_Test
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=20000
#SBATCH -o WAE_Stage3_David_1_Test_output.txt
#SBATCH -e WAE_Stage3_David_1_Test_error.txt
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
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage3_WAE_iter.py --run_name WAE_Stage3_David_1_Test_$RUN_NAME --load_from pretrain --st1_net WAE_Maria_512_New_Test_20220811-003030 --st1_load_epoch 20 --st2_net WAE_Stage2_David_1_Test_$RUN_NAME --st2_load_epoch final --disc_loss David --WAE_loss Both --lambda_WAE 1 --batch_size 64 --lr_dec 0.00003 --lr_disc 0.00001 --valid_shuffle True --latent_dims 512 --vox_res 1.8mm --set_size max --subject 1 --ROI VC --num_workers 2 --epochs 110 --iters 30000 --dataset NSD --seed 277603 --message "First double test using ideal david loss. Loading discrim from stage 1 and push cog latent to samp"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
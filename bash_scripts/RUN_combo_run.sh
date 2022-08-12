#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=Combo
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=20000
#SBATCH -o Combo_output.txt
#SBATCH -e Combo_error.txt
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

RUN_TIME=$(date +%Y%m%d-%H%M%S)
# SUBJ=1
# ------ CHANGE THESE ------- #
RUN_NAME="Combo"
VOX_RES="1.8mm"
BATCH_SIZE=100
SET_SIZE="max"
ROI="VC"
LOAD_FROM="pretrain"
STAGE_1_NET="WAE_1024_Test_20220806-125520"
STAGE_1_EPOCH="249"
# Maria default - but with the fixed real/fake thing.
# ------ CHANGE THESE ------- #
STAGE_3_NAME=WAE_Stage3_${SUBJ}_${RUN_NAME}_${RUN_TIME}

# for SUBJ in 1 2 3 4 5 6 7 8
for SUBJ in 2
do
  # Study 1
  # Run stage 2
  echo "Running stage 2 at $(date +%Y%m%d-%H%M%S)"
  srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2_WAE_iter.py --run_name WAE_Stage2_${SUBJ}_${RUN_NAME}_${RUN_TIME} --load_from ${LOAD_FROM} --standardize none --pretrained_net ${STAGE_1_NET} --load_epoch ${STAGE_1_EPOCH} --disc_loss Both --WAE_loss MSE --lambda_WAE 1 --lambda_GAN 10 --lambda_recon 1 --batch_size ${BATCH_SIZE} --lr_enc 0.001 --lr_disc 0.0005 --valid_shuffle True --latent_dims 1024 --lin_size 2048 --lin_layers 2 --clip_gradients False --vox_res 1.8mm --set_size max --subject ${SUBJ} --ROI VC --num_workers 2 --epochs 40 --iters 15000 --dataset NSD --seed 277603 --message "combination of Maria style fixed, with maria LR, but iwht no norm, unclipped gradients and bs 100, and higher model complexity"
  echo "Stage 2 complete at $(date +%Y%m%d-%H%M%S)"

  # Run stage 3
  # echo "Running stage 3 at $(date +%Y%m%d-%H%M%S)"
  # srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage3_WAE_iter.py --run_name ${STAGE_3_NAME} --load_from ${LOAD_FROM} --st1_net ${STAGE_1_NET} --st1_load_epoch ${STAGE_1_EPOCH} --st2_net WAE_Stage2_${SUBJ}_${RUN_NAME}_${RUN_TIME} --st2_load_epoch final --disc_loss Both --WAE_loss Both --lambda_WAE 1 --batch_size ${BATCH_SIZE} --lr_dec 0.00003 --lr_disc 0.00001 --valid_shuffle True --latent_dims 512 --vox_res 1.8mm --set_size max --subject ${SUBJ} --ROI VC --num_workers 2 --epochs 40 --iters 20000 --dataset NSD --seed 277603 --message "First double test using ideal david loss. Loading discrim from stage 1 and push cog latent to samp"
  # echo "Stage 3 complete at $(date +%Y%m%d-%H%M%S)"

  # tar the folder after complete
  # tar -czvf ${STAGE_3_NAME}.tar.gz /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/${STAGE_3_NAME}/
  # copy to afm
  # cp ${STAGE_3_NAME}.tar.gz /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/
done

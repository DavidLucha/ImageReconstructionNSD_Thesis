#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=Study1_Subj1
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=20000
#SBATCH -o Study1_Subj1_output.txt
#SBATCH -e Study1_Subj1_error.txt
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
# ------ CHANGE THESE ------- #
STUDY=1
SUBJ=1
VOX_RES="1.8mm"
BATCH_SIZE=32
SET_SIZE="max"
ROI="VC"
LOAD_FROM="pretrain"
STAGE_1_NET="WAE_1024_Test_20220806-125520"
STAGE_1_EPOCH="249"
RUN_NAME="Study${STUDY}_SUBJ0${SUBJ}_${VOX_RES}_${ROI}_${SET_SIZE}"
MESSAGE=""
# ------ CHANGE THESE ------- #
STAGE_2_NAME=${RUN_NAME}_Stage2_${RUN_TIME}
STAGE_3_NAME=${RUN_NAME}_Stage3_${RUN_TIME}


# Study 1
# Run stage 2
#TODO CHANGE EPOCH AND ITERATIONS
#TODO CHECK BATCHSIZE VAR IS WORKING
echo "Running stage 2 at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2_WAE_iter.py --run_name ${STAGE_2_NAME} --load_from ${LOAD_FROM} --pretrained_net ${STAGE_1_NET} --load_epoch ${STAGE_1_EPOCH} --standardize none --disc_loss Maria --WAE_loss Maria --lambda_WAE 1 --lambda_GAN 10 --lambda_recon 1 --batch_size ${BATCH_SIZE} --lr_enc 0.001 --lr_disc 0.0005 --valid_shuffle True --latent_dims 1024 --lin_size 2048 --lin_layers 2 --clip_gradients False --vox_res ${VOX_RES} --set_size ${SET_SIZE} --subject ${SUBJ} --ROI ${ROI} --num_workers 2 --epochs 12 --iters 8000 --dataset NSD --seed 277603 --message ${MESSAGE}
echo "Stage 2 complete at $(date +%Y%m%d-%H%M%S)"

# Run stage 3
echo "Running stage 3 at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage3_WAE_iter.py --run_name ${STAGE_3_NAME} --load_from ${LOAD_FROM} --st1_net ${STAGE_1_NET} --st1_load_epoch ${STAGE_1_EPOCH} --st2_net ${STAGE_2_NAME} --st2_load_epoch final --standardize none --disc_loss Maria --WAE_loss Maria --lambda_WAE 1 --lambda_GAN 10 --lambda_recon 1 --batch_size ${BATCH_SIZE} --lr_dec 0.001 --lr_disc 0.0005 --valid_shuffle True --latent_dims 1024 --lin_size 2048 --lin_layers 2 --clip_gradients False --vox_res ${VOX_RES} --set_size ${SET_SIZE} --subject ${SUBJ} --ROI ${ROI} --num_workers 2 --epochs 12 --iters 8000 --dataset NSD --seed 277603 --message ${MESSAGE}
echo "Stage 3 complete at $(date +%Y%m%d-%H%M%S)"

# tar the folder after complete
cd /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/
tar -czvf ${STAGE_3_NAME}.tar.gz /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/${STAGE_3_NAME}/
# copy to afm
cp ${STAGE_3_NAME}.tar.gz /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/
# copy only final network to network folder
cd ./${STAGE_3_NAME}/
cp ${STAGE_3_NAME}_final.pth /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/networks/


#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=RUN_1.8max_subj01_st2n3
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=20000
#SBATCH -o RUN_1.8max_subj01_st2n3_output.txt
#SBATCH -e RUN_1.8max_subj01_st2n3_error.txt
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
# SUBJ=1
VOX_RES="1.8mm"
SET_SIZE="max"
LOAD_FROM="stage_1"
LOAD_EP="199"

for SUBJ in 1 4
do
  # Study 1
  # Run stage 2
  echo "Running stage 2 at $(date +%Y%m%d-%H%M%S)"
  srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2_WAE_iter.py --run_name WAE_Stage2_David_1_Test_$RUN_NAME --load_from pretrain --pretrained_net WAE_Maria_512_New_Test_20220811-003030 --load_epoch 20 --disc_loss Both --WAE_loss Both --lambda_WAE 1 --batch_size 64 --lr_enc 0.0001 --lr_disc 0.00005 --valid_shuffle True --latent_dims 512 --vox_res 1.8mm --set_size max --subject 1 --ROI VC --num_workers 2 --epochs 150 --iters 30000 --dataset NSD --seed 277603 --message "First double test using ideal david loss. Fixing the weird flipped latent disc, and then using the recon loss calc from pretrain (manual instead of mse)"
  echo "Stage 2 complete at $(date +%Y%m%d-%H%M%S)"

  # Run stage 3
  echo "Running stage 3 at $(date +%Y%m%d-%H%M%S)"
  srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage3_WAE_iter.py --run_name WAE_Stage3_David_1_Test_$RUN_NAME --load_from pretrain --st1_net WAE_Maria_512_New_Test_20220811-003030 --st1_load_epoch 20 --st2_net WAE_Stage2_David_1_Test_$RUN_NAME --st2_load_epoch final --disc_loss David --WAE_loss Both --lambda_WAE 1 --batch_size 64 --lr_dec 0.00003 --lr_disc 0.00001 --valid_shuffle True --latent_dims 512 --vox_res 1.8mm --set_size max --subject 1 --ROI VC --num_workers 2 --epochs 150 --iters 15000 --dataset NSD --seed 277603 --message "First double test using ideal david loss. Loading discrim from stage 1 and push cog latent to samp"
  echo "Stage 3 complete at $(date +%Y%m%d-%H%M%S)"
done
# tar the folder after complete
# tar -czvf RUN_stage3_subj0${SUBJ}_${RUN_NAME}.tar.gz /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/Subj_0${SUBJ}/stage_3/RUN_stage3_subj0${SUBJ}_${RUN_NAME}/
# copy to afm
# cp RUN_stage3_subj0${SUBJ}_${RUN_NAME}.tar.gz /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/
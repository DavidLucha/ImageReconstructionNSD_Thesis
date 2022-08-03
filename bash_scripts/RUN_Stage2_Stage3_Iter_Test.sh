#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=iterRUN_1.8max_subj01_st2n3
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=20000
#SBATCH -o iterRUN_1.8max_subj01_st2n3_output.txt
#SBATCH -e iterRUN_1.8max_subj01_st2n3_error.txt
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
SUBJ=1
VOX_RES="1.8mm"
SET_SIZE="max"
LOAD_FROM="pretrain"
LOAD_EP="60"
PRE_NET="RUN1_1_stage1_only_20220802-225250"

# Study 1
# Run stage 2
echo "Running stage 2 at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2_iter.py --run_name iterRUN_1.8max_stage2_subj0${SUBJ}_${RUN_NAME} --vox_res ${VOX_RES} --set_size ${SET_SIZE} --subject ${SUBJ} --pretrained_net ${PRE_NET} --load_from ${LOAD_FROM} --load_epoch ${LOAD_EP} --lr 0.00001 --backprop_method clip --num_workers 2 --epochs 150 --iters 30000 --dataset NSD --loss_method Maria --seed 277603 --message "Stage 2, subject ${SUBJ}, at ${VOX_RES} voxel resolution, with set size of: ${SET_SIZE}"
echo "Stage 2 complete at $(date +%Y%m%d-%H%M%S)"

# Run stage 3
# echo "Running stage 3 at $(date +%Y%m%d-%H%M%S)"
# srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage3.py --run_name RUN_1.8max_stage3_subj0${SUBJ}_${RUN_NAME} --vox_res ${VOX_RES} --set_size ${SET_SIZE} --subject ${SUBJ} --pretrained_net ${PRE_NET} --load_from ${LOAD_FROM} --load_epoch ${LOAD_EP} --lr 0.00001 --backprop_method clip --num_workers 2 --epochs 150 --iters 15000 --dataset NSD --loss_method Maria --seed 277603 --message "Stage 2, subject ${SUBJ}, at ${VOX_RES} voxel resolution, with set size of: ${SET_SIZE}"
# echo "Stage 3 complete at $(date +%Y%m%d-%H%M%S)"

# tar the folder after complete
# tar -czvf RUN_stage3_subj0${SUBJ}_${RUN_NAME}.tar.gz /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/Subj_0${SUBJ}/stage_3/RUN_stage3_subj0${SUBJ}_${RUN_NAME}/
# copy to afm
# cp RUN_stage3_subj0${SUBJ}_${RUN_NAME}.tar.gz /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/
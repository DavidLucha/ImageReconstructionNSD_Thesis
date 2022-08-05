#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=optim_stage3_0001
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=50000
#SBATCH -o optim_stage3_0001_output.txt
#SBATCH -e optim_stage3_0001_error.txt
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
PRE_NET="optim_stage_2_0003_shuffle_20220805-122457"

# Study 1
# Run stage 2
# echo "Running stage 2 at $(date +%Y%m%d-%H%M%S)"
# srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2_iter.py --run_name optim_stage_2_0003_shuffle_${RUN_NAME} --vox_res ${VOX_RES} --set_size ${SET_SIZE} --subject ${SUBJ} --pretrained_net ${PRE_NET} --load_from ${LOAD_FROM} --load_epoch ${LOAD_EP} --lr_enc 0.0003 --lr_disc 0.00001 --valid_shuffle True --backprop_method trad --num_workers 2 --epochs 150 --iters 50000 --dataset NSD --seed 277603 --message "Stage 2, reduced discriminator lr, enc increased at .0003 with eval shuffle on"
# echo "Stage 2 complete at $(date +%Y%m%d-%H%M%S)"

# Run stage 3
echo "Running stage 3 at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage3_iter.py --run_name optim_stage_3_0001_subj0${SUBJ}_${RUN_NAME} --vox_res ${VOX_RES} --set_size ${SET_SIZE} --subject ${SUBJ} --stage_2_trained optim_stage_2_0003_shuffle_20220805-122457 --load_epoch final --lr_dec 0.0001 --lr_disc 0.00001 --valid_shuffle True --backprop_method trad --num_workers 2 --epochs 150 --iters 15000 --dataset NSD --seed 277603 --equilibrium_game y --message "Stage 3, subject ${SUBJ}, at ${VOX_RES} voxel resolution, with set size of: ${SET_SIZE}. Dec .0001, Dis .00001. Equi on."
echo "Stage 3 complete at $(date +%Y%m%d-%H%M%S)"
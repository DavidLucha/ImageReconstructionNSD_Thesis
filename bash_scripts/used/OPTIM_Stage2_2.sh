#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=optim_stage2_noclip_1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=20000
#SBATCH -o optim_stage2_noclip_1_output.txt
#SBATCH -e optim_stage2_noclip_1_error.txt
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
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2_iter.py --run_name optim_stage_2_noclip_${RUN_NAME} --vox_res ${VOX_RES} --set_size ${SET_SIZE} --subject ${SUBJ} --pretrained_net ${PRE_NET} --load_from ${LOAD_FROM} --load_epoch ${LOAD_EP} --lr_enc 0.0001 --lr_disc 0.0001 --backprop_method trad --num_workers 2 --epochs 150 --iters 50000 --dataset NSD --seed 277603 --message "Stage 2 optimize, both optims at .0001 with no clip"
echo "Stage 2 complete at $(date +%Y%m%d-%H%M%S)"
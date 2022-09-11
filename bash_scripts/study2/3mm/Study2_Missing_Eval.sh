#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=Study2_Subj1_Eval
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=20000
#SBATCH -o Study2_Subj1_Eval_output.txt
#SBATCH -e Study2_Subj1_Eval_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uqdlucha@uq.edu.au

module load anaconda
module load gcc/12.1.0
module load cuda/11.3.0
module load mvapich2
# module load openmpi3

VOX_RES="3mm"
BATCH_SIZE=100
SET_SIZE="max"
ROI="VC"

# EVAL
source activate /scratch/qbi/uqdlucha/python/dvaegan3_6
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size ${BATCH_SIZE} --st3_net 'Study2_SUBJ01_3mm_VC_max_Stage3_20220910-164249' --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res ${VOX_RES} --ROI ${ROI} --set_size ${SET_SIZE} --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

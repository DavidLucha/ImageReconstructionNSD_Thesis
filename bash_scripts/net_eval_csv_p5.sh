#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=P5_Eval
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=50000
#SBATCH -o P5_Eval_output.txt
#SBATCH -e P5_Eval_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uqdlucha@uq.edu.au

module load anaconda
module load gcc/12.1.0
module load cuda/11.3.0
module load mvapich2
# module load openmpi3

# source activate /scratch/qbi/uqdlucha/python/dvaegan
source activate /scratch/qbi/uqdlucha/python/dvaegan3_6

# Evaluation
echo "Running evaluation at $(date +%Y%m%d-%H%M%S)"

# 3mm
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ04_3mm_VC_1200_Stage3_20220905-000707" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 3mm --ROI VC --set_size 1200 --subject 4 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ04_3mm_VC_4000_Stage3_20220903-104357" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 3mm --ROI VC --set_size 4000 --subject 4 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ04_3mm_VC_7500_Stage3_20220903-090221" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 3mm --ROI VC --set_size 7500 --subject 4 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# SUBJ 5
# 1.8mm
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ05_1pt8mm_VC_1200_Stage3_20220901-141409" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size 1200 --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ05_1pt8mm_VC_4000_Stage3_20220901-105817" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size 4000 --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ05_1pt8mm_VC_7500_Stage3_20220901-141424" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size 7500 --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# 3mm
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ05_3mm_VC_1200_Stage3_20220902-013514" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 3mm --ROI VC --set_size 1200 --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ05_3mm_VC_4000_Stage3_20220901-174230" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 3mm --ROI VC --set_size 4000 --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ05_3mm_VC_7500_Stage3_20220901-202431" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 3mm --ROI VC --set_size 7500 --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# SUBJ 6
# 1.8mm
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ06_1pt8mm_VC_1200_Stage3_20220902-140110" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size 1200 --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ06_1pt8mm_VC_4000_Stage3_20220902-002601" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size 4000 --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ06_1pt8mm_VC_7500_Stage3_20220902-023445" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size 7500 --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
# 3mm
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ06_3mm_VC_1200_Stage3_20220903-022243" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 3mm --ROI VC --set_size 1200 --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ06_3mm_VC_4000_Stage3_20220902-071920" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 3mm --ROI VC --set_size 4000 --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study2_SUBJ06_3mm_VC_7500_Stage3_20220902-084404" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 3mm --ROI VC --set_size 7500 --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

echo "Evaluation complete at $(date +%Y%m%d-%H%M%S)"

# tar the folder after complete
# cd /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/
# tar -czvf ${STAGE_3_NAME}.tar.gz /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/${STAGE_3_NAME}/
# copy to afm
# cp ${STAGE_3_NAME}.tar.gz /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/
# copy only final network to network folder
# cd ./${STAGE_3_NAME}/
# cp ${STAGE_3_NAME}_final.pth /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/networks/


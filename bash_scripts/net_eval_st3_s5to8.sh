#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=ST3S5t8_Eval
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=50000
#SBATCH -o ST3S5t8_Eval_output.txt
#SBATCH -e ST3S5t8_Eval_error.txt
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
# Subject 5 Study 3 Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ05_1pt8mm_HVC_max_Stage3_20220825-133214" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ05_1pt8mm_V1_max_Stage3_20220824-151918" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1 --set_size max --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ05_1pt8mm_V2_max_Stage3_20220824-205238" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V2 --set_size max --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ05_1pt8mm_V3_max_Stage3_20220825-022720" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V3 --set_size max --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ05_1pt8mm_V4_max_Stage3_20220825-075836" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V4 --set_size max --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# Subject 6 Study 3 Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ06_1pt8mm_HVC_max_Stage3_20220825-140009" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ06_1pt8mm_V1_max_Stage3_20220824-153037" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1 --set_size max --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ06_1pt8mm_V2_max_Stage3_20220824-210807" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V2 --set_size max --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ06_1pt8mm_V3_max_Stage3_20220825-024558" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V3 --set_size max --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ06_1pt8mm_V4_max_Stage3_20220825-082325" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V4 --set_size max --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# Subject 7 Study 3 Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ07_1pt8mm_HVC_max_Stage3_20220825-134804" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 7 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ07_1pt8mm_V1_max_Stage3_20220824-152321" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1 --set_size max --subject 7 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ07_1pt8mm_V2_max_Stage3_20220824-210110" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V2 --set_size max --subject 7 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ07_1pt8mm_V3_max_Stage3_20220825-023706" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V3 --set_size max --subject 7 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ07_1pt8mm_V4_max_Stage3_20220825-081258" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V4 --set_size max --subject 7 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# Subject 8 Study 3 Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ08_1pt8mm_HVC_max_Stage3_20220825-151131" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 8 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ08_1pt8mm_V1_max_Stage3_20220824-160057" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1 --set_size max --subject 8 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ08_1pt8mm_V2_max_Stage3_20220824-214909" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V2 --set_size max --subject 8 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ08_1pt8mm_V3_max_Stage3_20220825-033754" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V3 --set_size max --subject 8 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ08_1pt8mm_V4_max_Stage3_20220825-092519" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V4 --set_size max --subject 8 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

echo "Evaluation complete at $(date +%Y%m%d-%H%M%S)"

# tar the folder after complete
# cd /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/
# tar -czvf ${STAGE_3_NAME}.tar.gz /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/${STAGE_3_NAME}/
# copy to afm
# cp ${STAGE_3_NAME}.tar.gz /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/
# copy only final network to network folder
# cd ./${STAGE_3_NAME}/
# cp ${STAGE_3_NAME}_final.pth /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/networks/


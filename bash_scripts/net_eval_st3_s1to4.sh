#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=ST3S1t4_Eval
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=50000
#SBATCH -o ST3S1t4_Eval_output.txt
#SBATCH -e ST3S1t4_Eval_error.txt
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
# Subject 1 Study 3 Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ01_1pt8mm_HVC_max_Stage3_20220823-154803" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ01_1pt8mm_V1_max_Stage3_20220822-173246" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1 --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ01_1pt8mm_V2_max_Stage3_20220822-230716" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V2 --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ01_1pt8mm_V3_max_Stage3_20220823-044337" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V3 --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ01_1pt8mm_V4_max_Stage3_20220823-101609" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V4 --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# Subject 2 Study 3 Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ02_1pt8mm_HVC_max_Stage3_20220823-160615" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ02_1pt8mm_V1_max_Stage3_20220822-174639" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1 --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ02_1pt8mm_V2_max_Stage3_20220822-232611" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V2 --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ02_1pt8mm_V3_max_Stage3_20220823-045920" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V3 --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ02_1pt8mm_V4_max_Stage3_20220823-103133" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V4 --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# Subject 3 Study 3 Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ03_1pt8mm_HVC_max_Stage3_20220823-162532" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 3 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ03_1pt8mm_V1_max_Stage3_20220822-175643" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1 --set_size max --subject 3 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ03_1pt8mm_V2_max_Stage3_20220822-233332" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V2 --set_size max --subject 3 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ03_1pt8mm_V3_max_Stage3_20220823-051201" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V3 --set_size max --subject 3 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ03_1pt8mm_V4_max_Stage3_20220823-104802" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V4 --set_size max --subject 3 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# Subject 4 Study 3 Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ04_1pt8mm_HVC_max_Stage3_20220823-162312" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 4 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ04_1pt8mm_V1_max_Stage3_20220822-175146" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1 --set_size max --subject 4 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ04_1pt8mm_V2_max_Stage3_20220822-232930" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V2 --set_size max --subject 4 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ04_1pt8mm_V3_max_Stage3_20220823-050943" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V3 --set_size max --subject 4 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ04_1pt8mm_V4_max_Stage3_20220823-104539" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V4 --set_size max --subject 4 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

echo "Evaluation complete at $(date +%Y%m%d-%H%M%S)"

# tar the folder after complete
# cd /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/
# tar -czvf ${STAGE_3_NAME}.tar.gz /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/${STAGE_3_NAME}/
# copy to afm
# cp ${STAGE_3_NAME}.tar.gz /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/
# copy only final network to network folder
# cd ./${STAGE_3_NAME}/
# cp ${STAGE_3_NAME}_final.pth /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/networks/


#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=P1_Eval
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=50000
#SBATCH -o P1_Eval_output.txt
#SBATCH -e P1_Eval_error.txt
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

# Study 1
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study1_SUBJ01_1pt8mm_VC_max_Stage3_20220817-112810" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study1_SUBJ02_1pt8mm_VC_max_Stage3_20220817-112727" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study1_SUBJ03_1pt8mm_VC_max_Stage3_20220817-112727" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 3 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study1_SUBJ04_1pt8mm_VC_max_Stage3_20220817-112727" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 4 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study1_SUBJ05_1pt8mm_VC_max_Stage3_20220817-112727" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 5 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study1_SUBJ06_1pt8mm_VC_max_Stage3_20220817-112730" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 6 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study1_SUBJ07_1pt8mm_VC_max_Stage3_20220817-172226" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 7 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study1_SUBJ08_1pt8mm_VC_max_Stage3_20220817-172325" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 8 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# FOR HPC
# Subj 1 Study 3
# Main Three
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ01_1pt8mm_V1_to_V3_max_Stage3_20220822-004912" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ01_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220822-062246" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ01_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220822-115814" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
# Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ01_1pt8mm_HVC_max_Stage3_20220823-154803" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 1 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

# Subj 2 Study 3
# Main Three
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ02_1pt8mm_V1_to_V3_max_Stage3_20220822-004914" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ02_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220822-062607" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ02_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220822-120754" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"
# Extra
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/net_evaluation_tabular.py --data_root '/scratch/qbi/uqdlucha/datasets/' --network_root '/scratch/qbi/uqdlucha/final_networks/' --batch_size 100 --st3_net "Study3_SUBJ02_1pt8mm_HVC_max_Stage3_20220823-160615" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI HVC --set_size max --subject 2 --num_workers 2 --dataset NSD --seed 277603 --message "hello"

echo "Evaluation complete at $(date +%Y%m%d-%H%M%S)"

# tar the folder after complete
# cd /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/
# tar -czvf ${STAGE_3_NAME}.tar.gz /scratch/qbi/uqdlucha/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/${ROI}/Subj_0${SUBJ}/stage_3/${STAGE_3_NAME}/
# copy to afm
# cp ${STAGE_3_NAME}.tar.gz /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/${SET_SIZE}/
# copy only final network to network folder
# cd ./${STAGE_3_NAME}/
# cp ${STAGE_3_NAME}_final.pth /afm02/Q3/Q3789/datasets/output/NSD/${VOX_RES}/networks/


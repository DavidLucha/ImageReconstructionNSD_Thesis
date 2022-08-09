#!/bin/bash

#SBATCH -N 1
#SBATCH --job-name=WAE_ST2_RMS_bs32_00030001_1024_2048lin2_Test
#SBATCH -n 2
#SBATCH -c 25
#SBATCH --mem=10000
#SBATCH -o WAE_ST2_RMS_bs32_00030001_1024_2048lin2_Test_output.txt
#SBATCH -e WAE_ST2_RMS_bs32_00030001_1024_2048lin2_Test_error.txt
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

# Study 1
# Pretrain one network. Used in both GOD and NSD.
echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage2_WAE_iter.py --run_name WAE_ST2_RMS_bs32_00030001_1024_2048lin2_Test_$RUN_NAME --load_from pretrain --pretrained_net WAE_1024_Test_20220806-125520 --optim_method RMS --load_epoch 150 --batch_size 32 --lr_enc 0.0003 --lr_disc 0.0001 --valid_shuffle True --lin_size 2048 --lin_layers 2 --latent_dims 1024 --vox_res 1.8mm --set_size max --subject 1 --num_workers 2 --epochs 60 --iters 200000 --dataset NSD --seed 277603 --message "Best stage 2 WAE but with 2048 neurons two layers, going int0 1024 and then out to latent dim size"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"

# Study 1
# Pretrain one network. Used in both GOD and NSD.
echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
srun -N 1 -p gpu --gres=gpu:1 --mpi=pmi2 python /clusterdata/uqdlucha/scripts/deepReconPyTorch/train_stage3_WAE_iter.py --run_name WAE_ST3_RMS_bs32_1024_Test_$RUN_NAME --load_from pretrain --st1_net WAE_1024_Test_20220806-125520 --st1_load_epoch 150 --st2_net WAE_ST2_RMS_bs32_00030001_1024_2048lin2_Test_$RUN_NAME --st2_load_epoch 59 --optim_method RMS --lr_dec 0.0003 --lr_disc 0.0001 --valid_shuffle True --lin_size 2048 --lin_layers 2 --latent_dims 1024 --vox_res 1.8mm --set_size max --subject 1 --num_workers 2 --epochs 200 --iters 60000 --dataset NSD --seed 277603 --message "WAE test, stage 3 default, loading in stage2 1024 evne though those are shit"
echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
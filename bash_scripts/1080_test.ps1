conda activate pytorch_1.12_b

$RUN_NAME = Get-Date -Format "dd-MM-yyyy_HH-mm"

# Study 1
# Pretrain one network. Used in both GOD and NSD.
$T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
echo "Running pretrain at $T_NOW"
# Study 1
# Pretrain one network. Used in both GOD and NSD.
python C:\Users\david\Python\deepReconPyTorch\pretrain_lr_finder_reduced.py --data_root 'D:/Lucha_Data/datasets/' --d_scale 0.0 --g_scale 0.0 --gamma 1.0 --equilibrium_game n --backprop_method new --num_workers 8 --epochs 100 --dataset both --run_name Ren_Scalar_MSE_noEqui_$RUN_NAME --loss_method Ren_Alt --optim_method Combined --message "new ren loss function recon weighting (gamma) at 1, lr at .0001, no equilibrium"
# python C:\Users\david\Python\deepReconPyTorch\pretrain.py --data_root 'D:/Lucha_Data/datasets/' --num_workers 8 --epochs 40 --dataset both --run_name no_gamma_$RUN_NAME --loss_method Ren_Alt --optim_method RMS --d_scale 0.0 --g_scale 0.0 --equilibrium_game y --message "Testing Ren alt loss with updated encoder weights"
$T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
echo "Pretrain complete at $T_NOW"

# $T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
# echo "Running pretrain at $T_NOW"
# python C:\Users\david\Python\deepReconPyTorch\pretrain.py --data_root 'D:/Lucha_Data/datasets/' --num_workers 8 --epochs 40 --dataset both --run_name gamma_256_$RUN_NAME --loss_method Ren_Alt --optim_method RMS --gamma 256.0 --d_scale 0.0 --g_scale 0.0 --equilibrium_game y --message "Testing Ren alt loss with updated encoder weights + channels for feature loss"
# $T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
# echo "Pretrain complete at $T_NOW"

# $T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
# echo "Running pretrain at $T_NOW"
# python C:\Users\david\Python\deepReconPyTorch\pretrain.py --data_root 'D:/Lucha_Data/datasets/' --num_workers 8 --epochs 40 --dataset both --run_name off_$RUN_NAME --loss_method Ren --optim_method Adam --equilibrium_game n --message "equil off"
# $T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
# echo "Pretrain complete at $T_NOW"
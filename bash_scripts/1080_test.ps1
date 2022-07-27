conda activate pytorch_1.12_b

$RUN_NAME = Get-Date -Format "dd-MM-yyyy_HH-mm"

# Study 1
# Pretrain one network. Used in both GOD and NSD.
$T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
echo "Running pretrain at $T_NOW"
# Study 1
# Pretrain one network. Used in both GOD and NSD.
python C:\Users\david\Python\deepReconPyTorch\pretrain.py --data_root 'D:/Lucha_Data/datasets/' --run_name Pretrain_Maria_$RUN_NAME --d_scale 0.0 --g_scale 0.0 --lr 0.0003 --gamma 1.0 --equilibrium_game y --backprop_method trad --num_workers 2 --epochs 400 --dataset both --loss_method Ren_Alt --optim_method RMS --message "400 epoch run, Maria basic."
# python C:\Users\david\Python\deepReconPyTorch\pretrain.py --data_root 'D:/Lucha_Data/datasets/' --d_scale 0.0 --g_scale 0.0 --lr 0.0001 --gamma 1.0 --equilibrium_game n --backprop_method new --num_workers 8 --epochs 75 --dataset both --run_name Testing_Update_$RUN_NAME --loss_method Ren_Alt --optim_method Adam

# python C:\Users\david\Python\deepReconPyTorch\pretrain_lr_finder_reduced.py --data_root 'D:/Lucha_Data/datasets/' --d_scale 0.0 --g_scale 0.0 --gamma 1.0 --equilibrium_game n --backprop_method new --num_workers 8 --epochs 100 --dataset both --run_name Ren_Scalar_MSE_noEqui_$RUN_NAME --loss_method Ren_Alt --optim_method Combined --message "new ren loss function recon weighting (gamma) at 1, lr at .0001, no equilibrium"
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

# echo "Running pretrain at $(date +%Y%m%d-%H%M%S)"
# echo "Pretrain complete at $(date +%Y%m%d-%H%M%S)"
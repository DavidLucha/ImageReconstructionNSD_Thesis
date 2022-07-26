conda activate dvaegan

$RUN_NAME = Get-Date -Format "dd-MM-yyyy_HH-mm"

# Study 1
# Pretrain one network. Used in both GOD and NSD.
$T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
echo "Running pretrain at $T_NOW"
python C:\Users\david\Python\deepReconPyTorch\pretrain.py --data_root 'D:/Lucha_Data/datasets/' --num_workers 8 --epochs 40 --dataset both --run_name on_$RUN_NAME --loss_method Ren --optim_method Adam --equilibrium_game y --message "equil on"
$T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
echo "Pretrain complete at $T_NOW"

$T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
echo "Running pretrain at $T_NOW"
python C:\Users\david\Python\deepReconPyTorch\pretrain.py --data_root 'D:/Lucha_Data/datasets/' --num_workers 8 --epochs 40 --dataset both --run_name off_$RUN_NAME --loss_method Ren --optim_method Adam --equilibrium_game n --message "equil off"
$T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
echo "Pretrain complete at $T_NOW"
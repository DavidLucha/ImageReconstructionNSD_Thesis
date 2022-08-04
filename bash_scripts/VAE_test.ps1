conda activate pytorch_1.12_b

$RUN_NAME = Get-Date -Format "dd-MM-yyyy_HH-mm"

# Study 1
# Pretrain one network. Used in both GOD and NSD.
$T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
echo "Running pretrain at $T_NOW"
python C:\Users\david\Python\deepReconPyTorch\train_stage1_vae.py --data_root 'D:/Lucha_Data/datasets/' --run_name VAE_Test_$RUN_NAME --lr 0.003 --backprop_method trad --num_workers 8 --epochs 200 --dataset both --loss_method Maria --optim_method RMS --message "400 epoch run, Maria basic, lr 0003"

$T_NOW = Get-Date -Format "dd-MM-yyyy_HH-mm"
echo "Pretrain complete at $T_NOW"


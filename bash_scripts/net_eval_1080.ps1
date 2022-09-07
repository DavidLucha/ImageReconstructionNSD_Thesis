# python C:\Users\david\Python\deepReconPyTorch\net_evaluation.py --st3_net "Study1_SUBJ01_1pt8mm_VC_max_Stage3_20220817-112810" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 1 --num_workers 8 --dataset NSD --seed 277603 --message "hello"
conda activate dvaegan
echo "hello there"

# Study 1
# python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study1_SUBJ01_1pt8mm_VC_max_Stage3_20220817-112810" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 1 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
# python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study1_SUBJ02_1pt8mm_VC_max_Stage3_20220817-112727" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 2 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
# python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study1_SUBJ03_1pt8mm_VC_max_Stage3_20220817-112727" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 3 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
# python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study1_SUBJ04_1pt8mm_VC_max_Stage3_20220817-112727" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 4 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
# python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study1_SUBJ05_1pt8mm_VC_max_Stage3_20220817-112727" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 5 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
# python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study1_SUBJ06_1pt8mm_VC_max_Stage3_20220817-112730" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 6 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
# python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study1_SUBJ07_1pt8mm_VC_max_Stage3_20220817-172226" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 7 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
# python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study1_SUBJ08_1pt8mm_VC_max_Stage3_20220817-172325" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI VC --set_size max --subject 8 --num_workers 4 --dataset NSD --seed 277603 --message "hello"

# FOR HPC
# Subj 1 Study 3
# Main Three
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ01_1pt8mm_V1_to_V3_max_Stage3_20220822-004912" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 1 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ01_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220822-062246" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 1 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ01_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220822-115814" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 1 --num_workers 4 --dataset NSD --seed 277603 --message "hello"

# Subj 2 Study 3
# Main Three
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ02_1pt8mm_V1_to_V3_max_Stage3_20220822-004914" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 2 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ02_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220822-062607" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 2 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ02_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220822-120754" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 2 --num_workers 4 --dataset NSD --seed 277603 --message "hello"

# Subj 3 Study 3
# Main Three
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ03_1pt8mm_V1_to_V3_max_Stage3_20220822-004914" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 3 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ03_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220822-063205" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 3 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ03_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220822-121550" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 3 --num_workers 4 --dataset NSD --seed 277603 --message "hello"

# Subj 4 Study 3
# Main Three
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ04_1pt8mm_V1_to_V3_max_Stage3_20220822-004916" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 4 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ04_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220822-062600" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 4 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ04_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220822-120743" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 4 --num_workers 4 --dataset NSD --seed 277603 --message "hello"

# Subj 5 Study 3
# Main Three
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ05_1pt8mm_V1_to_V3_max_Stage3_20220823-222716" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 5 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ05_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220824-040118" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 5 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ05_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220824-093837" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 5 --num_workers 4 --dataset NSD --seed 277603 --message "hello"

# Subj 6 Study 3
# Main Three
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ06_1pt8mm_V1_to_V3_max_Stage3_20220823-222717" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 6 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ06_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220824-040838" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 6 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ06_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220824-095018" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 6 --num_workers 4 --dataset NSD --seed 277603 --message "hello"

# Subj 7 Study 3
# Main Three
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ07_1pt8mm_V1_to_V3_max_Stage3_20220823-222716" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 7 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ07_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220824-040419" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 7 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ07_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220824-094407" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 7 --num_workers 4 --dataset NSD --seed 277603 --message "hello"

# Subj 8 Study 3
# Main Three
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ08_1pt8mm_V1_to_V3_max_Stage3_20220823-222716" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3 --set_size max --subject 8 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ08_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220824-041720" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_HVC --set_size max --subject 8 --num_workers 4 --dataset NSD --seed 277603 --message "hello"
python C:\Users\david\Python\deepReconPyTorch\net_evaluation_tabular.py --st3_net "Study3_SUBJ08_1pt8mm_V1_to_V3_n_rand_max_Stage3_20220824-100909" --st3_load_epoch final --latent_dims 1024 --lin_size 2048 --lin_layers 2 --save True --vox_res 1pt8mm --ROI V1_to_V3_n_rand --set_size max --subject 8 --num_workers 4 --dataset NSD --seed 277603 --message "hello"










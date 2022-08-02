import training_config
import time

batch_size = 64
# epochs = training_config.n_epochs
image_size = 100
num_workers = 1
dataset ='both'
recon_level = training_config.recon_level
run_name = time.strftime("%Y%m%d-%H%M%S")
data_root = 'D:/Lucha_Data/datasets/'


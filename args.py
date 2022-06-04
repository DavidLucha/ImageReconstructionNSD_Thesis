import training_config

input = ''
batch_size = training_config.batch_size
epochs = training_config.n_epochs
image_size = training_config.image_size
num_workers = training_config.num_workers
pretrained_net = training_config.pretrained_net
load_epoch = training_config.load_epoch
dataset ='GOD'
subset = '1.8mm'  # '1.8mm, 3mm, 5S_Small, 8S_Small,'
recon_level = training_config.recon_level
# add loss method
# add model import | probs temporary

# Stage 1
stage_1_net = training_config.pretrained_net


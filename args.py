import training_config

input = ''
batch_size = training_config.batch_size
epochs = training_config.n_epochs # TODO: Change based on stages
image_size = training_config.image_size
num_workers = training_config.num_workers
pretrained_net = training_config.pretrained_net
stage_1_trained = training_config.stage_1_trained
stage_2_trained = training_config.stage_2_trained
stage_3_trained = training_config.stage_3_trained  # FINAL MODEL
load_epoch = training_config.load_epoch
dataset ='GOD'
subset = '1.8mm'  # '1.8mm, 3mm, 5S_Small, 8S_Small,'
recon_level = training_config.recon_level
network_checkpoint = None # 'vaegan_20220613-014326'  # 'vaegan_20220610-153208' # None
checkpoint_epoch = 90
subject_no = 3  # Default three for GOD
# subject 0 is all subjects (for NSD)
# add loss method
# add model import | probs temporary

# Stage 1
stage_1_net = training_config.pretrained_net


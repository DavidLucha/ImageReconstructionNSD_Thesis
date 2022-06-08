"""____________________Config for Dual-VAE/GAN training___________________________"""
decoder_weights = ['gan_20210127-012348', 90] # This should change in stage 2 vs 3
pretrained_net = 'vaegan_20220606-130121'  # TODO: Change this
stage_1_trained = ''
stage_2_trained = 'vaegan_20220608-152046'
stage_3_trained = '' # FINAL MODEL
load_epoch = 20  # was 335
# TODO: Make sure network is saving final epoch
evaluate = False

image_crop = 375 # Not sure why this | will be different for COCO vs ImageNet
image_size = 100
latent_dim = 128 # was 512

device = 'cuda:0'  # cuda or cpu
device2 = 'cuda:3'
device3 = 'cuda:5'

patience = 0   # for early stopping, 0 = deactivate early stopping
data_split = 0.2
batch_size = 16  # pytorch vaegan =64 | according to Ren main (16)
learning_rate_s1 = 0.0001 # TODO: Stage 1 (should be .003 - but too drastic right now)
# NOTE: Origin VAE/GAE implementation uses 3e-4 but is dependent on batch size (64)
learning_rate = 0.0003 # Stage 2 & 3
weight_decay = 0 # 1e-7
n_epochs = 400 # 400 for Stage 1 & 2 | 61 for pretraining (had at 20 but idk)
n_epochs_s3 = 200 # Stage 3
num_workers = 8 # was 4
step_size = 30  # for scheduler
gamma = 0.1     # for scheduler
recon_level = 3
lambda_mse = 1e-6 # weight for style area - VAEGAN implementation
decay_lr = 0.98 # 0.98 (Maria) 0.75 (Lam)
decay_margin = 1# margin decay for the generator/discriminator game
decay_mse = 1 # mse weight decrease
decay_equilibrium = 1 # equilibrium decay for the generator/discriminator game
margin = 0.35 # margin for generator/discriminator game | 0.35 in new imp, 0.4 in orig
equilibrium = 0.68 # equilibrium for the generator/discriminator game
beta = 1.0 # beta factor for beta-vae |  MIGHT NOT NEED

kernel_size = 4
stride = 2
padding_mode = [1, 1, 1, 1, 1, 0]
dropout = 0.7

save_images = 5
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

"""____________________Data Config___________________________"""

LOGS_PATH = 'logs/'
# TRAIN_IMG_PATH = "D:/Honours/Object Decoding Dataset/images_passwd/images/training/"
# SAVE_PATH = "D:/Honours/Object Decoding Dataset/7387130/Subject Training Pickles/"
data_root = 'D:/Lucha_Data/datasets/'
# save_training_results = ''
TRAINED_NET_ROOT = ''

# GOD Data
# For Pretrain
# TODO: Download non-overlapping pretrain images
god_pretrain_imgs = 'GOD/images/pretrain_25k/'
# For Stage 1
god_s1_train_imgs = 'GOD/images/train/'
god_s1_valid_imgs = 'GOD/images/valid/'
# For Stage 2 and 3
god_train_data = 'GOD/GOD_Subject1_train_normed.pickle' #
god_valid_data = 'GOD/GOD_Subject1_valid_normed_avg.pickle' # Average
# god_valid_data = 'GOD/GOD_all_subjects_valid.pickle' # Not average (50img*35presentations)

# NSD Data
# Change this for NSD tests
nsd_mode = '1.8mm/'  # 3mm | 5S_Small etc.

# TODO: We will have to change imgs path depending on test/how we run sample size comparison
# For Stage 1
nsd_s1_train_imgs = 'NSD/images/train/'  # add + nsd_mode
nsd_s1_valid_imgs = 'NSD/images/valid/'
# For Stage 2 and 3
nsd_train_data = 'NSD/' + nsd_mode + 'NSD_all_subjects_train.pickle'
nsd_valid_data = 'NSD/' + nsd_mode + 'NSD_all_subjects_train.pickle'



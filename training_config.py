"""____________________Config for Dual-VAE/GAN training___________________________"""
# NU = not used
decoder_weights = ['gan_20210127-012348', 90] # This should change in stage 2 vs 3
pretrained_net = ['vaegan_20220613-014326', 90]  # TODO: Change this
stage_1_trained = ['vaegan_20220613-014326', 90] # TODO: Change this - currently a pretrain
stage_2_trained = ['vaegan_20220615-104028', 380] # TODO: Change this 'vaegan_20220608-193031'
stage_3_trained = '' # FINAL MODEL
load_epoch = 10  # was 335
# TODO: Make sure network is saving final epoch

evaluate = False

image_size = 100
latent_dim = 128 # was 512

device = 'cuda:0'  # cuda or cpu
device2 = 'cuda:3'  # NU
device3 = 'cuda:5'  # NU
num_workers = 8  # was 4, then 8 for good

batch_size = 64  # according to Ren code = 16
# Maria loss functions require 64 (sum loss)

learning_rate_pt = 0.0001
# NOTE: Original VAE/GAE implementation uses 3e-4 but is dependent on batch size (64)
learning_rate_s1 = 0.0001 # TODO: Change after loss functions change .0003
# Stage 1 lr should be .003 (according to Ren) - but too drastic right now given the sum loss (use 0.0001)
learning_rate = 0.00001  # Stage 2 & 3
weight_decay = 0  # Used in optimizers
decay_lr = 0.98 # 0.98 (Maria) 0.75 (Lam)

n_epochs_pt = 5 # 120
n_epochs = 400  # 400 for Stage 1 & 2
n_epochs_s3 = 200  # Stage 3

recon_level = 3 # TODO: Check this is getting the right layer
lambda_mse = 1e-6 # weight for style error in loss calculation

# For equilibrium game
decay_margin = 1# margin decay for the generator/discriminator game
decay_mse = 1 # mse weight decrease
decay_equilibrium = 1 # equilibrium decay for the generator/discriminator game
margin = 0.35 # margin for generator/discriminator game | 0.35 in new imp, 0.4 in orig
equilibrium = 0.68 # equilibrium for the generator/discriminator game

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# kernel_size = 4  # NU
# stride = 2  # NU
# padding_mode = [1, 1, 1, 1, 1, 0]  # NU
# dropout = 0.7  # NU
# gamma = 0.1  # NU
# save_images = 5  # NU

"""____________________Data Config___________________________"""

LOGS_PATH = 'logs/'
# data_root = '/scratch/qbi/uqdlucha/datasets/'
data_root = 'D:/Lucha_Data/datasets/'
# data_root for home: 'D:/Lucha_Data/datasets/'
# save_training_results = ''
TRAINED_NET_ROOT = ''

# GOD Data
# For Pretrain
god_pretrain_imgs = 'GOD/images/pretrain_30k/'
# For Stage 1
god_s1_train_imgs = 'GOD/images/train/'
god_s1_valid_imgs = 'GOD/images/valid/'
# For Stage 2 and 3
god_train_data = 'GOD/GOD_SubjectZ_train_normed.pickle' #
god_valid_data = 'GOD/GOD_SubjectZ_valid_normed_avg.pickle' # Average
# god_valid_data = 'GOD/GOD_all_subjects_valid.pickle' # Not average (50img*35presentations)

# NSD Data
# Change this for NSD tests
# nsd_mode = '1.8mm/'  # 3mm | 5S_Small etc.

# TODO: We will have to change imgs path depending on test/how we run sample size comparison
# For Stage 1
nsd_s1_train_imgs = 'NSD/images/train/'  # add + nsd_mode
nsd_s1_valid_imgs = 'NSD/images/valid/'
# TODO: Rename these to reflect output
# For Stage 2 and 3
# nsd_train_data = 'NSD_SubjectZ_train.pickle'
# nsd_valid_data = 'NSD_SubjectZ_train.pickle'



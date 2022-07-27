import os
import time
import numpy
import torch
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys

import torchvision
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch_lr_finder import LRFinder

import training_config
from model_2 import VaeGan
from utils_2 import ImageNetDataloader, GreyToColor, evaluate, PearsonCorrelation, \
    StructuralSimilarity, objective_assessment, parse_args, NLLNormal, potentiation


if __name__ == "__main__":
    try:
        numpy.random.seed(2010)
        torch.manual_seed(2010)
        torch.cuda.manual_seed(2010)
        logging.info('set up random seeds')

        torch.autograd.set_detect_anomaly(True)
        timestep = time.strftime("%Y%m%d-%H%M%S")
        # print('timestep is ',timestep)

        stage = 1

        """
        ARGS PARSER
        """
        arguments = True  # Set to False while testing

        if arguments:
            # args = parse_args(sys.argv[1:])
            parser = argparse.ArgumentParser()
            # parser.add_argument('--input', help="user path where the datasets are located", type=str)

            parser.add_argument('--run_name', default=timestep, help='sets the run name to the time shell script run',
                                type=str)
            parser.add_argument('--data_root', default=training_config.data_root,
                                help='sets directory of /datasets folder. Default set to scratch.'
                                     'If on HPC, use /scratch/qbi/uqdlucha/datasets/,'
                                     'If on home PC, us D:/Lucha_Data/datasets/', type=str)
            # Optimizing parameters | also, see lambda and margin in training_config.py
            parser.add_argument('--batch_size', default=training_config.batch_size, help='batch size for dataloader',
                                type=int)
            parser.add_argument('--epochs', default=training_config.n_epochs_pt, help='number of epochs', type=int)
            parser.add_argument('--num_workers', '-nw', default=training_config.num_workers,
                                help='number of workers for dataloader', type=int)
            parser.add_argument('--loss_method', default='Maria',
                                help='defines loss calculations. Maria, David, Orig.', type=str)
            parser.add_argument('--optim_method', default='RMS',
                                help='defines method for optimizer. Options: RMS or Adam.', type=str)
            parser.add_argument('--lr', default=training_config.learning_rate_pt, type=float)
            parser.add_argument('--decay_lr', default=training_config.decay_lr,
                                help='.98 in Maria, .75 in original VAE/GAN', type=float)
            parser.add_argument('--adam_beta', default=0.9,
                                help='sets the first value of the adam optimizer', type=float)
            parser.add_argument('--lambda_loss', default=1e-6,
                                help='sets the weighting of the reconstruction loss', type=float)
            parser.add_argument('--equilibrium_game', default='y', type=str,
                                 help='Sets whether to engage the equilibrium game for decoder/disc updates (y/n)')
            parser.add_argument('--d_scale', default=0.25,
                                help='sets the d value of scale for Ren loss', type=float)
            parser.add_argument('--g_scale', default=0.625,
                                help='sets the g value of scale for Ren loss', type=float)
            parser.add_argument('--gamma', default=1.0,
                                help='sets the weighting of KL divergence in encoder loss (Ren) or'
                                     'the weight of MSE_1 in encoder loss (David: 1 vs 5)', type=float)
            parser.add_argument('--backprop_method', default='trad', help='trad sets three diff loss functions,'
                                                                          'but new, means enc and dec are updated'
                                                                          ' using the same loss function', type=str)
            parser.add_argument('--klw', default=1.0, help='sets weighting for KL divergence', type=float)



            # Pretrained/checkpoint network components
            parser.add_argument('--network_checkpoint', default=None, help='loads checkpoint in the format '
                                                                           'vaegan_20220613-014326', type=str)
            parser.add_argument('--checkpoint_epoch', default=90, help='epoch of checkpoint network', type=int)
            parser.add_argument('--pretrained_net', '-pretrain', default=training_config.pretrained_net,
                                help='pretrained network', type=str)
            parser.add_argument('--load_epoch', '-pretrain_epoch', default=training_config.load_epoch,
                                help='epoch of the pretrained model', type=int)
            parser.add_argument('--dataset', default='both', help='GOD, NSD, both', type=str)
            # Only need vox_res arg from stage 2 and 3
            parser.add_argument('--vox_res', default='1.8mm', help='1.8mm, 3mm', type=str)
            # Probably only needed stage 2/3 (though do we want to change stage 1 - depends on whether we do the full Ren...
            # Thing where we do a smaller set for stage 1. But I think I might do a Maria and just have stage 1 with no...
            # pretraining phase...
            parser.add_argument('--set_size', default='max', help='max:max available, large:7500, med:4000, small:1200')
            parser.add_argument('--message', default='', help='Any notes or other information')
            args = parser.parse_args()

        if not arguments:
            import args

        """
        PATHS
        """
        # Get current working directory
        CWD = os.getcwd()
        OUTPUT_PATH = os.path.join(args.data_root, '../output/')

        TRAIN_DATA_PATH = os.path.join(args.data_root, training_config.god_pretrain_imgs)
        VALID_DATA_PATH = os.path.join(args.data_root, 'both/images/valid/')

        # Create directory for results
        stage_num = 'pretrain'
        SAVE_PATH = os.path.join(OUTPUT_PATH, args.dataset, stage_num, args.run_name)
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        SAVE_SUB_PATH = os.path.join(SAVE_PATH, 'pretrained_vaegan_{}.pth'.format(args.run_name))
        if not os.path.exists(SAVE_SUB_PATH):
            os.makedirs(SAVE_SUB_PATH)

        LOG_PATH = os.path.join(SAVE_PATH, training_config.LOGS_PATH)
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
            # os.chmod(LOG_PATH, 0o777)

        """
        LOGGING SETUP
        """
        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Info logging
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        logger = logging.getLogger()
        file_handler = logging.FileHandler(os.path.join(LOG_PATH, 'log.txt'))
        handler_formatting = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(handler_formatting)
        # logger = logging.getLogger()
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Check available gpu
        # import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(device)
        # conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.2 -c pytorch
        logger.info("Used device: %s" % device)
        if device == 'cpu':
            raise Exception()

        # Save arguments
        with open(os.path.join(SAVE_PATH, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)


        """
        DATASET LOADING
        """
        # Load image-only training data
        train_data = TRAIN_DATA_PATH
        valid_data = VALID_DATA_PATH

        # Load data
        # For pretraining, images are of different aspect ratios. We apply a resize to shorter side of the image \
        # To maintain the aspect ratio of the original image. Then apply a random crop and horizontal flip.
        training_data = ImageNetDataloader(train_data, pickle=False,
                                       transform=transforms.Compose([transforms.Resize(training_config.image_size),
                                                                     transforms.RandomCrop((training_config.image_size,
                                                                                            training_config.image_size)),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor(),
                                                                     GreyToColor(training_config.image_size),
                                                                     transforms.Normalize(training_config.mean,
                                                                                          training_config.std)
                                                                     ]))

        # We then apply CenterCrop and don't random flip to ensure that validation reconstructions match the \
        # ground truth image grid.
        validation_data = ImageNetDataloader(valid_data, pickle=False,
                                       transform=transforms.Compose([transforms.Resize(training_config.image_size),
                                                                     transforms.CenterCrop((training_config.image_size,
                                                                                            training_config.image_size)),
                                                                     transforms.ToTensor(),
                                                                     GreyToColor(training_config.image_size),
                                                                     transforms.Normalize(training_config.mean,
                                                                                          training_config.std)
                                                                     ]))

        dataloader_train = DataLoader(training_data, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)
        dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.num_workers)

        model = VaeGan(device=device, z_size=training_config.latent_dim, recon_level=training_config.recon_level).to(device)

        # Variables for equilibrium to improve GAN stability
        lr = 1e-7

        # An optimizer and schedulers for each of the sub-networks, so we can selectively backprop
        optim_method = args.optim_method  # RMS or Adam or SGD (Momentum)

        if optim_method == 'Combined':
            beta_1 = args.adam_beta
            # todo: make adam_beta arg
            eps = 1e-8
            encdec_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
            optimizer_encdec = torch.optim.Adam(params=encdec_params, lr=lr, eps=eps,
                                                 betas=(beta_1, 0.999), weight_decay=training_config.weight_decay)
            optimizer_discriminator = torch.optim.Adam(params=model.discriminator.parameters(), lr=lr, eps=eps,
                                                       betas=(beta_1, 0.999),  weight_decay=training_config.weight_decay)

        criterion_enc = nn.MSELoss()
        criterion_dis = nn.BCELoss()
        lr_enc_finder = LRFinder(model, optimizer_encdec, criterion_enc, device=device)
        lr_dis_finder = LRFinder(model.discriminator, optimizer_discriminator, criterion_dis, device=device)
        lr_enc_finder.range_test(dataloader_train, end_lr=100, num_iter=100)
        lr_enc_finder.plot()
        lr_enc_finder.reset()
        lr_dis_finder.range_test(dataloader_train, end_lr=100, num_iter=100)
        lr_dis_finder.plot()
        lr_dis_finder.reset()


        plt.show()
        # plt.close('all')

        exit(0)
    except Exception:
        logger.error("Fatal error", exc_info=True)
        sys.exit(1)

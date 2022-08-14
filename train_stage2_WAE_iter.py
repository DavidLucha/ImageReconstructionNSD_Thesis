import os
import time
import numpy
import json
import torch
import sys
import pickle
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

import torchvision
from torch import nn, no_grad
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, StepLR

import training_config
from model_2 import VaeGan, Encoder, Decoder, VaeGanCognitive, Discriminator, CognitiveEncoder, WaeGan, WaeGanCognitive
from utils_2 import GreyToColor, evaluate, PearsonCorrelation, \
    StructuralSimilarity, objective_assessment, parse_args, FmriDataloader, potentiation

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def main():
    try:
        timestep = time.strftime("%Y%m%d-%H%M%S")

        """
        ARGS PARSER
        """
        arguments = True  # Set to False while testing

        if arguments:
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
            parser.add_argument('--epochs', default=training_config.n_epochs, help='number of epochs', type=int)
            parser.add_argument('--iters', default=30000, help='sets max number of forward passes. 30k for stage 2'
                                                               ', 15k for stage 3.', type=int)
            parser.add_argument('--num_workers', '-nw', default=training_config.num_workers,
                                help='number of workers for dataloader', type=int)
            parser.add_argument('--lr_enc', default=.001, type=float)
            parser.add_argument('--lr_disc', default=.0005, type=float)
            parser.add_argument('--decay_lr', default=0.5,
                                help='.98 in Maria, .75 in original VAE/GAN', type=float)
            parser.add_argument('--seed', default=277603, help='sets seed, 0 makes a random int', type=int)
            parser.add_argument('--valid_shuffle', '-shuffle', default='False', type=str, help='defines whether'
                                                                                               'eval dataloader shuffles')
            parser.add_argument('--latent_dims', default=1024, type=int)
            parser.add_argument('--beta', default=0.5, type=float)
            parser.add_argument('--recon_loss', default='trad', type=str, help='sets whether to use pytroch mse'
                                                                               'or manual like in pretrain (manual)')
            parser.add_argument('--lin_size', default=1024, type=int, help='sets the number of nuerons in cog lin layer')
            parser.add_argument('--lin_layers', default=1, type=int, help='sets how many layers of cog network '
                                                                          'before the mu var layers.')
            parser.add_argument('--optim_method', default='Adam',
                                help='defines method for optimizer. Options: RMS or Adam.', type=str)
            parser.add_argument('--standardize', default='z',
                                help='determines whether the dataloader uses standardize.', type=str)
            parser.add_argument('--disc_loss', default='Maria',
                                help='determines whether we use Marias loss or the paper based one for disc', type=str)
            parser.add_argument('--WAE_loss', default='Maria',
                                help='determines whether we use Marias loss or the paper based one for WAE', type=str)
            parser.add_argument('--lambda_WAE', default=1, help='sets the multiplier for paper WAE loss', type=int)
            parser.add_argument('--lambda_GAN', default=10, help='sets the multiplier for individual GAN losses',
                                type=int)
            parser.add_argument('--lambda_recon', default=1, help='weight of recon loss', type=int)
            parser.add_argument('--clip_gradients', default='False',
                                help='determines whether to clip gradients or not', type=str)

            # Pretrained/checkpoint network components
            parser.add_argument('--network_checkpoint', default=None, help='loads checkpoint in the format '
                                                                           'vaegan_20220613-014326', type=str)
            parser.add_argument('--checkpoint_epoch', default=90, help='epoch of checkpoint network', type=int)
            parser.add_argument('--pretrained_net', '-pretrain', default=training_config.pretrained_net,
                                help='pretrained network', type=str)
            parser.add_argument('--load_from',default='stage_1', help='sets whether pretrained net is from pretrain'
                                                                      'or from stage_1 output', type=str)
            parser.add_argument('--load_epoch', '-pretrain_epoch', default='final',
                                help='epoch of the pretrained model', type=str)
            parser.add_argument('--dataset', default='NSD', help='GOD, NSD', type=str)
            # Only need vox_res arg from stage 2 and 3
            parser.add_argument('--vox_res', default='1pt8mm', help='1pt8mm, 3mm', type=str)
            # ROIs: V1, V2, V3, V1_to_V3, V4, HVC (faces + place + body areas), V1_to_V3_n_HVC, V1_to_V3_n_rand
            parser.add_argument('--ROI', default='VC', help='selects roi, only relevant for ROI analysis'
                                                            'otherwise default is VC.', type=str)
            parser.add_argument('--set_size', default='max', help='max:max available (including repeats), '
                                                                  'single_pres: max available single presentations,'
                                                                  '7500, 4000, 1200', type=str)
            parser.add_argument('--subject', default=1, help='Select subject number. GOD(1-5) and NSD(1-8)', type=int)
            parser.add_argument('--message', default='', help='Any notes or other information')
            args = parser.parse_args()

        if not arguments:
            import args

        """
        PATHS
        """
        # Get current working directory
        CWD = os.getcwd()
        OUTPUT_PATH = os.path.join(args.data_root, 'output/')
        SUBJECT_PATH = 'Subj_0{}/'.format(str(args.subject))

        # Load training data for GOD and NSD, default is NSD
        if args.dataset == 'NSD':
            if args.set_size == 'max':
                # To pull for study 1 and study 3 (ROIs) - default is VC (all ROIs)
                TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'train', args.set_size, args.ROI,
                                               'Subj_0{}_NSD_{}_train.pickle'.format(args.subject, args.set_size))
                if args.vox_res == "3mm":
                    # If needing to load max of 3mm (not in any studies for thesis) use this
                    TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'train', args.set_size,
                                                   'Subj_0{}_NSD_{}_train.pickle'.format(args.subject, args.set_size))
            elif args.set_size == 'single_pres':
                TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'train', args.set_size,
                                           'Subj_0{}_NSD_{}_train.pickle'.format(args.subject, args.set_size))
            else:
                # For loading 1200, 4000, 7500
                TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'train/single_pres', args.set_size,
                                           'Subj_0{}_{}_NSD_single_pres_train.pickle'.format(args.subject, args.set_size))

            # Currently valid data is set to 'max' meaning validation data contains multiple image presentations
            # If you only want to evaluate a single presentation of images replace both 'max' in strings below ...
            # with 'single_pres' and remove args.ROI
            VALID_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'valid', 'max', args.ROI,
                                           'Subj_0{}_NSD_max_valid.pickle'.format(args.subject))
        else:
            # Used to test on Generic Object Decoding Dataset (not used in Thesis)
            TRAIN_DATA_PATH = os.path.join(args.data_root, 'GOD',
                                           'GOD_Subject{}_train_normed.pickle'.format(args.subject))
            VALID_DATA_PATH = os.path.join(args.data_root, 'GOD',
                                           'GOD_Subject{}_valid_normed.pickle'.format(args.subject))

        # Create directory for results
        stage_num = 'stage_2'
        stage = 2

        SAVE_PATH = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, args.ROI, SUBJECT_PATH,
                                 stage_num, args.run_name)
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        SAVE_SUB_PATH = os.path.join(SAVE_PATH, '{}.pth'.format(args.run_name))
        if not os.path.exists(SAVE_SUB_PATH):
            os.makedirs(SAVE_SUB_PATH)

        LOG_PATH = os.path.join(SAVE_PATH, training_config.LOGS_PATH)
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)

        # Save arguments
        with open(os.path.join(SAVE_PATH, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

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
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        """
        SETTING SEEDS
        """
        seed = args.seed
        if seed == 0:
            seed = random.randint(1, 999999)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        logging.info('Set up random seeds...\nSeed is: {}'.format(seed))

        torch.autograd.set_detect_anomaly(True)
        # print('timestep is ',timestep)

        """
        DEVICE SETTING
        """
        # Check available gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info("Used device: %s" % device)
        if device == 'cpu':
            raise Exception()

        # Load data pickles for subject
        with open(TRAIN_DATA_PATH, "rb") as input_file:
            train_data = pickle.load(input_file)
        logger.info("Loading training pickles for subject {} from {}".format(args.subject, TRAIN_DATA_PATH))
        with open(VALID_DATA_PATH, "rb") as input_file:
            valid_data = pickle.load(input_file)
        logger.info("Loading validation pickles for subject {} from {}".format(args.subject, VALID_DATA_PATH))

        root_path = os.path.join(args.data_root, args.dataset + '/')

        """
        DATASET LOADING
        """
        if args.valid_shuffle == 'True':
            shuf = True
        else:
            shuf = False

        # standardize = False

        # if args.standardize == "True":
        #     standardize = True

        logging.info('Are we standardize the fMRI inputs into the dataloader?', args.standardize)

        # Load data
        training_data = FmriDataloader(dataset=train_data, root_path=root_path, standardizer=args.standardize,
                                           transform=transforms.Compose([transforms.Resize((training_config.image_size,
                                                                                            training_config.image_size)),
                                                                         transforms.CenterCrop(
                                                                             (training_config.image_size,
                                                                              training_config.image_size)),
                                                                         # RandomShift(),
                                                                         # SampleToTensor(),
                                                                         transforms.ToTensor(),
                                                                         GreyToColor(training_config.image_size),
                                                                         transforms.Normalize(training_config.mean,
                                                                                              training_config.std)
                                                                         ]))

        validation_data = FmriDataloader(dataset=valid_data, root_path=root_path, standardizer=args.standardize,
                                           transform=transforms.Compose([transforms.Resize((training_config.image_size,
                                                                                            training_config.image_size)),
                                                                         transforms.CenterCrop(
                                                                             (training_config.image_size,
                                                                              training_config.image_size)),
                                                                         # RandomShift(),
                                                                         # SampleToTensor(),
                                                                         transforms.ToTensor(),
                                                                         GreyToColor(training_config.image_size),
                                                                         transforms.Normalize(training_config.mean,
                                                                                              training_config.std)
                                                                         ]))

        dataloader_train = DataLoader(training_data, batch_size=args.batch_size,  # collate_fn=collate_fn,
                                      shuffle=True, num_workers=args.num_workers)
        dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size,  # collate_fn=collate_fn,
                                      shuffle=shuf, num_workers=args.num_workers)

        NUM_VOXELS = len(train_data[0]['fmri'])

        # Load Stage 1 network weights
        model_dir = os.path.join(OUTPUT_PATH, 'NSD', 'stage_1', args.pretrained_net,
                                   'stage_1_WAE_' + args.pretrained_net + '_{}.pth'.format(args.load_epoch))

        if args.load_from == 'pretrain':
            # Used if we didn't do pretrain -> stage 1, but just pretrain and use that.
            model_dir = os.path.join(OUTPUT_PATH, 'NSD', 'pretrain', args.pretrained_net,
                                     'pretrained_WAE_' + args.pretrained_net + '_{}.pth'.format(args.load_epoch))

        logging.info('Loaded network is:', model_dir)

        trained_model = WaeGan(device=device, z_size=args.latent_dims).to(device)
        trained_model.load_state_dict(torch.load(model_dir, map_location=device))

        # Fix decoder weights
        for param in trained_model.decoder.parameters():
            param.requires_grad = False

        # --- New fixed net for training --- #

        trained_model_fixed = WaeGan(device=device, z_size=args.latent_dims).to(device)
        trained_model_fixed.load_state_dict(torch.load(model_dir, map_location=device))

        # Fix decoder weights
        for param in trained_model_fixed.parameters():
            param.requires_grad = False

        trained_model_fixed.eval()

        # --- END fixed net for training --- #

        logging.info(f'Number of voxels: {NUM_VOXELS}')
        logging.info(f'Train data length: {len(train_data)}')
        logging.info(f'Validation data length: {len(valid_data)}')

        # Define model
        cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=args.latent_dims, lin_size=args.lin_size,
                                             lin_layers=args.lin_layers).to(device)
        # TODO: Check - aren't we supposed to be loading a trained discriminator too?
        # no, here we would want a fresh one. saying 1 to vis enc disc, and 0 to cog enc and then move cog towards vis
        model = WaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=trained_model.decoder,
                                z_size=args.latent_dims).to(device)


        logging.info('Using loaded network')
        stp = 1

        # Create empty results containers
        results = dict(
            epochs=[],
            loss_reconstruction=[],
            loss_penalty=[],
            loss_discriminator=[],
            loss_reconstruction_eval=[],
            loss_penalty_eval=[],
            loss_discriminator_eval=[]
        )

        # Variables for equilibrium to improve GAN stability
        lr_enc = args.lr_enc  # 0.001 - Maria | 0.0001 in st 1
        lr_disc = args.lr_disc  # 0.0005 - Maria | 0.00005 in st1
        # 0.0001 * 0.5
        beta = args.beta

        if args.optim_method == 'RMS':
            optimizer_encoder = torch.optim.RMSprop(params=model.encoder.parameters(), lr=lr_enc, alpha=0.9,)
            # optimizer_decoder = torch.optim.RMSprop(params=model.decoder.parameters(), lr=lr,
            #                                         alpha=0.9,
            #                                         eps=1e-8, weight_decay=training_config.weight_decay, momentum=0,
            #                                         centered=False)
            optimizer_discriminator = torch.optim.RMSprop(params=model.discriminator.parameters(), lr=lr_disc, alpha=0.9)
            lr_encoder = ExponentialLR(optimizer_encoder, gamma=args.decay_lr)
            # lr_decoder = ExponentialLR(optimizer_decoder, gamma=args.decay_lr)
            lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=args.decay_lr)

        else:
            # Optimizers
            optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lr_enc, betas=(beta, 0.999))
            # optimizer_decoder = torch.optim.Adam(model.decoder.parameters(), lr=0.001, betas=(0.5, 0.999))
            optimizer_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=lr_disc, betas=(beta, 0.999))

            lr_encoder = StepLR(optimizer_encoder, step_size=30, gamma=0.5)
            # lr_decoder = StepLR(optimizer_decoder, step_size=30, gamma=0.5)
            lr_discriminator = StepLR(optimizer_discriminator, step_size=30, gamma=0.5)

        # Define criterion
        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss(reduction='none')

        # Metrics
        pearson_correlation = PearsonCorrelation()
        structural_similarity = StructuralSimilarity(mean=training_config.mean, std=training_config.std)

        result_metrics_train = {}
        result_metrics_valid = {}
        metrics_train = {'train_PCC': pearson_correlation, 'train_SSIM': structural_similarity, 'train_MSE': mse_loss}
        metrics_valid = {'valid_PCC': pearson_correlation, 'valid_SSIM': structural_similarity, 'valid_MSE': mse_loss}

        # Resets the metrics_train and metrics_valid to empty
        if metrics_valid is not None:
            for key in metrics_valid.keys():
                results.update({key: []})
            for key, value in metrics_valid.items():
                result_metrics_valid.update({key: 0.0})

        if metrics_train is not None:
            for key in metrics_train.keys():
                results.update({key: []})
            for key, value in metrics_train.items():
                result_metrics_train.update({key: 0.0})


        show_img = True
        if show_img:
            trained_model.eval()
            real_batch = next(iter(dataloader_valid))

            # x_fmri = Variable(data_batch['fmri'], requires_grad=False).float().to(device)
            data_img = Variable(real_batch['image'], requires_grad=False).float().to(device)

            z_vis_enc, _ = trained_model.encoder(data_img)
            data_target = trained_model.decoder(z_vis_enc)

            images_dir = os.path.join(SAVE_PATH, 'images', 'pretest')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Validation Ground Truth')
            ax.imshow(
                make_grid(data_target.cpu().detach(), nrow=4, normalize=True).permute(1, 2, 0))
            gt_dir = os.path.join(images_dir, 'valid_recon_pretrained_' + 'grid')
            plt.savefig(gt_dir)

        batch_number = len(dataloader_train)
        step_index = 0
        epochs_n = args.epochs
        # iters = 0
        max_iters = args.iters

        while step_index < max_iters:
            for idx_epoch in range(args.epochs):
                try:
                    # For each batch
                    for batch_idx, data_batch in enumerate(dataloader_train):
                        if step_index < max_iters:
                            model.train()

                            # Fix decoder weights
                            for param in model.decoder.parameters():
                                param.requires_grad = False
                            # Fix the extra net
                            for param in trained_model_fixed.parameters():
                                param.requires_grad = False

                            trained_model_fixed.eval()

                            # frozen_params(model.decoder)
                            batch_size = len(data_batch)
                            logging.info(batch_size)
                            model.encoder.zero_grad()
                            model.discriminator.zero_grad()

                            x_fmri = Variable(data_batch['fmri'], requires_grad=False).float().to(device)
                            x_image = Variable(data_batch['image'], requires_grad=False).float().to(device)

                            # ----------Train discriminator-------------

                            # frozen_params(model.encoder)
                            # free_params(model.discriminator)

                            z_cog_enc, var = model.encoder(x_fmri)
                            # TODO: Return or leave this
                            z_vis_enc, var = trained_model_fixed.encoder(x_image)

                            # WAE | Trying to make Qz more like Pz
                            # disc output for encoded latent (cog) | Qz
                            logits_cog_enc = model.discriminator(z_cog_enc)
                            # disc output for visual encoder | Pz
                            logits_vis_enc = model.discriminator(z_vis_enc)

                            if args.disc_loss == "Maria":
                                sig_cog_enc = torch.sigmoid(logits_cog_enc)
                                sig_vis_enc = torch.sigmoid(logits_vis_enc)
                                # THEY HAVE FUCKED UP.
                                # here taking the BCE (1, cogenc) INCORRECT
                                # under the assumption we want the cog enc latent to be more like the VIS ENC
                                # the effect of this is a network which classifies a virgin cogenc as real
                                loss_discriminator_fake = - args.lambda_GAN * torch.sum(torch.log(sig_cog_enc + 1e-3))
                                # here taking the BCE of (0, vis enc)
                                # minimizes likelihood of discriminator identifying vis enc as 0
                                # is that the network incorrectly classifies the trained network encs as false
                                loss_discriminator_real = - args.lambda_GAN * torch.sum(torch.log(1 - sig_vis_enc + 1e-3))

                                loss_discriminator = loss_discriminator_fake + loss_discriminator_real

                                loss_discriminator_fake.backward(retain_graph=True)
                                loss_discriminator_real.backward(retain_graph=True)
                                mean_mult = batch_size * args.lambda_GAN
                            elif args.disc_loss == "Maria_Flip":
                                # Corrects the flip of cog and vis enc for real and fake
                                sig_cog_enc = torch.sigmoid(logits_cog_enc)
                                sig_vis_enc = torch.sigmoid(logits_vis_enc)
                                # THEY HAVE FUCKED UP.
                                # here taking the BCE (1, cogenc) INCORRECT
                                # under the assumption we want the cog enc latent to be more like the VIS ENC
                                # the effect of this is a network which classifies a virgin cogenc as real
                                loss_discriminator_fake = - args.lambda_GAN * torch.sum(torch.log(sig_vis_enc + 1e-3))
                                # here taking the BCE of (0, vis enc)
                                # minimizes likelihood of discriminator identifying vis enc as 0
                                # is that the network incorrectly classifies the trained network encs as false
                                loss_discriminator_real = - args.lambda_GAN * torch.sum(torch.log(1 - sig_cog_enc + 1e-3))

                                loss_discriminator = loss_discriminator_fake + loss_discriminator_real

                                loss_discriminator_fake.backward(retain_graph=True)
                                                                 # inputs=list(model.discriminator.parameters()))
                                loss_discriminator_real.backward(retain_graph=True)
                                                                 # inputs=list(model.discriminator.parameters()))
                                mean_mult = batch_size * args.lambda_GAN
                            elif args.disc_loss == "Both":
                                # Using Maria's but with modern Pytorch BCE loss + addition of loss terms before back pass
                                bce_loss = nn.BCEWithLogitsLoss(reduction='none')

                                # set up labels
                                labels_real = Variable(torch.ones_like(logits_vis_enc)).to(device)
                                labels_fake = Variable(torch.zeros_like(logits_cog_enc)).to(device)

                                # Qz is distribution of encoded latent space
                                loss_Qz = args.lambda_GAN * torch.sum(bce_loss(logits_cog_enc, labels_fake))
                                # Pz is distribution of prior (sampled)
                                loss_Pz = args.lambda_GAN * torch.sum(bce_loss(logits_vis_enc, labels_real))

                                loss_discriminator = args.lambda_WAE * (loss_Qz + loss_Pz)
                                loss_discriminator.backward(retain_graph=True)
                                                            # inputs=list(model.discriminator.parameters()))
                                mean_mult = batch_size * args.lambda_GAN
                            else:
                                # set up labels
                                labels_real = Variable(torch.ones_like(logits_vis_enc)).to(device)
                                labels_fake = Variable(torch.zeros_like(logits_cog_enc)).to(device)

                                # Qz is distribution of encoded latent space | here, the learning latent
                                loss_Qz = bce_loss(logits_cog_enc, labels_fake)
                                # Pz is distribution of prior (sampled) | or here, the teacher latent
                                loss_Pz = bce_loss(logits_vis_enc, labels_real)

                                loss_discriminator = args.lambda_WAE * (loss_Qz + loss_Pz)
                                loss_discriminator.backward(retain_graph=True)
                                                            # inputs=list(model.discriminator.parameters()))
                                mean_mult = 1

                            if args.clip_gradients == "True":
                                [p.grad.data.clamp_(-1, 1) for p in model.discriminator.parameters()]
                            optimizer_discriminator.step()

                            # ----------Train generator----------------
                            # free_params(model.encoder)
                            # (model.discriminator)

                            z_cog_enc, var = model.encoder(x_fmri)
                            x_recon = model.decoder(z_cog_enc)
                            logits_cog_enc = model.discriminator(z_cog_enc)

                            z_vis_enc, _ = trained_model_fixed.encoder(x_image)
                            x_gt = trained_model_fixed.decoder(z_vis_enc)

                            model.encoder.zero_grad()

                            if args.WAE_loss == 'Maria':
                                # Get sigmoid of logits
                                sig_cog_enc = torch.sigmoid(logits_cog_enc)
                                # loss_reconstruction = torch.sum(torch.sum(0.5 * (x_recon - x_image) ** 2, 1))
                                mse_loss = nn.MSELoss()
                                loss_reconstruction = mse_loss(x_recon, x_image)
                                # This is equivalent of BCELoss(real, 1)
                                # so we are saying the loss is the distance of the latent space of cog enc
                                # to the real 1 (of the visual enc)
                                loss_penalty = - args.lambda_GAN * torch.mean(torch.log(sig_cog_enc + 1e-3))

                                loss_reconstruction.backward(retain_graph=True)  # inputs=list(model.encoder.parameters()))
                                loss_penalty.backward()  # inputs=list(model.encoder.parameters()))
                                mean_mult_pen = batch_size * args.lambda_GAN
                                mean_mult_rec = batch_size
                            elif args.WAE_loss == "Both":
                                # As with Maria, but with BCELogits and feature loss (using vis recon as real)
                                # But also Maria does a MSELoss in this stage and not the manual squared diff
                                bce_loss = nn.BCEWithLogitsLoss(reduction='none')
                                # Like Maria's but with MSE and BCE from pytorch
                                # label for non-saturating loss
                                labels_saturated = Variable(torch.ones_like(logits_cog_enc)).to(device)
                                # Only thing, is in here stage 2 she uses mse mean rather than the weighted sum below
                                loss_reconstruction = torch.sum(torch.sum(0.5 * (x_recon - x_gt) ** 2, 1))
                                # loss_reconstruction = mse_loss(x_recon, x_gt)
                                loss_penalty = args.lambda_GAN * torch.sum(bce_loss(logits_cog_enc, labels_saturated))
                                loss_WAE = loss_reconstruction + loss_penalty * args.lambda_WAE
                                loss_WAE.backward()  # inputs=list(model.encoder.parameters()))
                                mean_mult_pen = batch_size * args.lambda_GAN
                                mean_mult_rec = batch_size
                            elif args.WAE_loss == "MSE":
                                # As with Maria, but with BCELogits and feature loss (using vis recon as real)
                                # But also Maria does a MSELoss in this stage and not the manual squared diff
                                bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
                                mse_loss = nn.MSELoss(reduction='mean')
                                # Like Maria's but with MSE and BCE from pytorch
                                # label for non-saturating loss
                                labels_saturated = Variable(torch.ones_like(logits_cog_enc)).to(device)
                                # Only thing, is in here stage 2 she uses mse mean rather than the weighted sum below
                                # loss_reconstruction = torch.sum(torch.sum(0.5 * (x_recon - x_gt) ** 2, 1))
                                loss_reconstruction = mse_loss(x_recon, x_gt) * args.lambda_recon
                                loss_penalty = args.lambda_GAN * bce_loss(logits_cog_enc, labels_saturated)
                                loss_WAE = loss_reconstruction + loss_penalty * args.lambda_WAE
                                loss_WAE.backward()  # inputs=list(model.encoder.parameters()))
                                mean_mult_pen = 1  # * 10?
                                mean_mult_rec = 1 * args.lambda_recon
                            else:
                                # Adapted from original WAE paper code
                                # label for non-saturating loss
                                labels_saturated = Variable(torch.ones_like(logits_cog_enc)).to(device)
                                loss_reconstruction = torch.mean(torch.sum(mse_loss(x_recon, x_gt), [1, 2, 3])) * 0.05
                                # loss_reconstruction = mse_loss(x_recon, x_gt)
                                loss_penalty = bce_loss(logits_cog_enc, labels_saturated)
                                loss_WAE = loss_reconstruction + loss_penalty * args.lambda_WAE
                                loss_WAE.backward()  # inputs=list(model.encoder.parameters()))
                                mean_mult_pen = 1
                                mean_mult_rec = 1
                                # loss_reconstruction = torch.sum(torch.sum(0.5 * (x_recon - x_gt) ** 2, 1))

                            for name, param in model.encoder.named_parameters():
                                if param.requires_grad:
                                    print('Encoder Parameters:', name, param.data)

                            if args.clip_gradients == "True":
                                # This isn't working.
                                [p.grad.data.clamp_(-1, 1) for p in model.encoder.parameters()]
                            optimizer_encoder.step()

                            model.zero_grad()

                            # register mean values of the losses for logging
                            loss_reconstruction_mean = loss_reconstruction.data.cpu().numpy() / mean_mult_rec
                            loss_penalty_mean = loss_penalty.data.cpu().numpy() / mean_mult_pen
                            loss_discriminator_mean = loss_discriminator.data.cpu().numpy() / mean_mult

                            logging.info(
                                f'Epoch  {idx_epoch} {batch_idx + 1:3.0f} / {100 * (batch_idx + 1) / len(dataloader_train):2.3f}%, '
                                f'---- recon loss: {loss_reconstruction_mean:.5f} ---- | '
                                f'---- penalty loss: {loss_penalty_mean:.5f} ---- | '
                                f'---- discrim loss: {loss_discriminator_mean:.5f}')

                            step_index += 1
                        else:
                            break

                    # EPOCH END
                    lr_encoder.step()
                    # lr_decoder.step()
                    lr_discriminator.step()

                    # Record losses & scores
                    results['epochs'].append(idx_epoch + stp)
                    results['loss_reconstruction'].append(loss_reconstruction_mean)
                    results['loss_penalty'].append(loss_penalty_mean)
                    results['loss_discriminator'].append(loss_discriminator_mean)

                    # plot arrangements
                    # grid_count = 25
                    nrow = int(math.sqrt(batch_size))
                    # if batch_size == 64:
                    #     grid_count = 25
                    # nrow = int(math.sqrt(grid_count))

                    if args.set_size in ("1200", "4000", "7500"):
                        train_recon_freq = 10
                    else:
                        train_recon_freq = 2

                    if not idx_epoch % train_recon_freq:
                        # Save train examples
                        images_dir = os.path.join(SAVE_PATH, 'images', 'train')
                        if not os.path.exists(images_dir):
                            os.makedirs(images_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Training Ground Truth at Epoch {}'.format(idx_epoch))
                        ax.imshow(make_grid(x_image.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                        gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                        plt.savefig(gt_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Training Vis Enc Reconstruction (Real) at Epoch {}'.format(idx_epoch))
                        ax.imshow(
                            make_grid(x_gt.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                        gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_vis_output_' + 'grid')
                        plt.savefig(gt_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Training Reconstruction at Epoch {}'.format(idx_epoch))
                        ax.imshow(make_grid(x_recon.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                        output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                        plt.savefig(output_dir)

                    logging.info('Evaluation')

                    for batch_idx, data_batch in enumerate(dataloader_valid):
                        # model.eval()

                        with no_grad():

                            model.eval()
                            # trained_model.eval()

                            data_in = Variable(data_batch['fmri'], requires_grad=False).float().to(device)
                            data_img = Variable(data_batch['image'], requires_grad=False).float().to(device)
                            # out, logits_out = model(data_in)
                            # data_target, z_target = trained_model(data_img)
                            batch_size = len(data_batch)
                            logging.info(batch_size)

                            z_cog_enc, _ = model.encoder(data_in)
                            out = model.decoder(z_cog_enc)
                            # z_vis_enc, _ = trained_model.encoder(x_image)

                            z_vis_enc, _ = trained_model_fixed.encoder(data_img)
                            data_target = trained_model_fixed.decoder(z_vis_enc)

                            logits_out = model.discriminator(z_cog_enc)
                            logits_target = model.discriminator(z_vis_enc)

                            bce_loss = nn.BCEWithLogitsLoss(reduction='none')

                            if args.disc_loss == "Maria":
                                sig_out = torch.sigmoid(logits_out)
                                sig_target = torch.sigmoid(logits_target)
                                loss_out_fake = - args.lambda_GAN * torch.sum(torch.log(sig_out + 1e-3))
                                loss_out_real = - args.lambda_GAN * torch.sum(torch.log(1 - sig_target + 1e-3))

                                loss_discriminator_eval = loss_out_fake + loss_out_real
                                loss_discriminator_mean_eval = loss_discriminator_eval / (batch_size * args.lambda_GAN)
                            elif args.disc_loss == "Maria_Flip":
                                sig_out = torch.sigmoid(logits_out)
                                sig_target = torch.sigmoid(logits_target)
                                loss_out_fake = - args.lambda_GAN * torch.sum(torch.log(sig_target + 1e-3))
                                loss_out_real = - args.lambda_GAN * torch.sum(torch.log(1 - sig_out + 1e-3))

                                loss_discriminator_eval = loss_out_fake + loss_out_real
                                loss_discriminator_mean_eval = loss_discriminator_eval / (batch_size * args.lambda_GAN)
                            else:
                                # Note the below is only accurate if using 'both' for WAE and disc loss
                                # Discriminator loss
                                labels_real_eval = Variable(torch.ones_like(logits_target, requires_grad=False)).to(device)
                                labels_fake_eval = Variable(torch.zeros_like(logits_out, requires_grad=False)).to(device)
                                loss_out_fake = args.lambda_GAN * torch.sum(bce_loss(logits_out, labels_fake_eval))
                                loss_target_real = args.lambda_GAN * torch.sum(bce_loss(logits_target, labels_real_eval))
                                mean_mult = batch_size * args.lambda_GAN
                                loss_discriminator_mean_eval = (loss_out_fake + loss_target_real) / mean_mult

                            # Recon and penalty loss
                            labels_saturated_eval = Variable(torch.ones_like(logits_out, requires_grad=False)).to(device)
                            if args.WAE_loss == "MSE":
                                bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
                                mse_loss = nn.MSELoss(reduction='mean')

                                loss_reconstruction_eval = mse_loss(out, data_target) * args.lambda_recon
                                loss_reconstruction_mean_eval = loss_reconstruction_eval / args.lambda_recon

                                loss_penalty_eval = args.lambda_GAN * bce_loss(logits_out, labels_saturated_eval)
                                loss_penalty_mean_eval = loss_penalty_eval
                            if args.WAE_loss == "Maria":
                                sig_out = torch.sigmoid(logits_out)
                                mse_loss = nn.MSELoss()
                                loss_reconstruction_mean_eval = mse_loss(out, data_img)
                                loss_penalty_mean_eval = - args.lambda_GAN * torch.mean(torch.log(sig_out + 1e-3))

                                # mean_mult_pen = batch_size * args.lambda_GAN
                                # mean_mult_rec = 1
                            else:
                                loss_reconstruction_eval = torch.sum(torch.sum(0.5 * (out - data_target) ** 2, 1))
                                loss_reconstruction_mean_eval = loss_reconstruction_eval / batch_size
                                loss_penalty_eval = args.lambda_GAN * torch.sum(bce_loss(logits_out, labels_saturated_eval))
                                # mean_mult = batch_size  # * 10?
                                loss_penalty_mean_eval = loss_penalty_eval / batch_size * args.lambda_GAN

                            # Validation metrics for the first validation batch
                            if metrics_valid is not None:
                                for key, metric in metrics_valid.items():
                                    if key == 'cosine_similarity':
                                        result_metrics_valid[key] = metric(out, data_target).mean()
                                    else:
                                        result_metrics_valid[key] = metric(out, data_target)

                            # Training metrics for the last training batch
                            if metrics_train is not None:
                                for key, metric in metrics_train.items():
                                    if key == 'cosine_similarity':
                                        result_metrics_train[key] = metric(x_recon, x_gt).mean()
                                    else:
                                        result_metrics_train[key] = metric(x_recon, x_gt)

                            out = out.data.cpu()

                        # Save validation examples
                        images_dir = os.path.join(SAVE_PATH, 'images', 'valid')
                        if not os.path.exists(images_dir):
                            os.makedirs(images_dir)
                            os.makedirs(os.path.join(images_dir, 'random'))

                        out = out.data.cpu()

                        if args.set_size in ("1200", "4000"):
                            eval_recon_freq = 20
                        elif args.set_size == "7500":
                            eval_recon_freq = 2
                        else:
                            eval_recon_freq = 1

                        if shuf:
                            if not idx_epoch % eval_recon_freq:
                                fig, ax = plt.subplots(figsize=(10, 10))
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.set_title('Validation Ground Truth')
                                ax.imshow(
                                    make_grid(data_target.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                                gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                                plt.savefig(gt_dir)

                                # fig, ax = plt.subplots(figsize=(10, 10))
                                # ax.set_xticks([])
                                # ax.set_yticks([])
                                # ax.set_title('Validation Vis Enc Reconstruction at Epoch {}'.format(idx_epoch))
                                # ax.imshow(make_grid(vis_out.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                                # output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_vis_output_' + 'grid')
                                # plt.savefig(output_dir)

                        else:  # valid_shuffle is false, so same images are shown
                            if idx_epoch == 0:
                                fig, ax = plt.subplots(figsize=(10, 10))
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.set_title('Validation Ground Truth')
                                ax.imshow(make_grid(data_target.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                                gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                                plt.savefig(gt_dir)

                                # fig, ax = plt.subplots(figsize=(10, 10))
                                # ax.set_xticks([])
                                # ax.set_yticks([])
                                # ax.set_title('Validation Vis Enc Reconstruction at Epoch {}'.format(idx_epoch))
                                # ax.imshow(
                                #     make_grid(vis_out.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                                # output_dir = os.path.join(images_dir,
                                #                           'epoch_' + str(idx_epoch) + '_vis_output_' + 'grid')
                                # plt.savefig(output_dir)

                        if not idx_epoch % eval_recon_freq:
                            fig, ax = plt.subplots(figsize=(10, 10))
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_title('Validation Reconstruction at Epoch {}'.format(idx_epoch))
                            ax.imshow(make_grid(out.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                            output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                            plt.savefig(output_dir)

                        # out = (out + 1) / 2
                        # out = make_grid(out, nrow=8)

                        # out = model(None, 100)
                        # out = out.data.cpu()
                        # out = (out + 1) / 2
                        # out = make_grid(out, nrow=8)
                        # writer.add_image("generated", out, step_index)

                        # out = data_target.data.cpu()
                        # out = (out + 1) / 2
                        # out = make_grid(out, nrow=8)

                        if metrics_valid is not None:
                            for key, values in result_metrics_valid.items():
                                result_metrics_valid[key] = torch.mean(values)

                        if metrics_train is not None:
                            for key, values in result_metrics_train.items():
                                result_metrics_train[key] = torch.mean(values)

                        logging.info(
                            f'Epoch  {idx_epoch} ---- train PCC:  {result_metrics_train["train_PCC"].item():.5f} ---- | '
                            f'---- train SSIM: {result_metrics_train["train_SSIM"].item():.5f} ---- '
                            f'---- train MSE: {result_metrics_train["train_MSE"].item():.5f} ---- ')

                        logging.info(
                            f'Epoch  {idx_epoch} ---- valid PCC:  {result_metrics_valid["valid_PCC"].item():.5f} ---- | '
                            f'---- valid SSIM: {result_metrics_valid["valid_SSIM"].item():.5f} ---- '
                            f'---- valid MSE: {result_metrics_valid["valid_MSE"].item():.5f} ---- ')

                        # only for one batch due to memory issue
                        break

                    # Record losses & scores for eval
                    results['loss_reconstruction_eval'].append(loss_reconstruction_mean_eval)
                    results['loss_penalty_eval'].append(loss_penalty_mean_eval)
                    results['loss_discriminator_eval'].append(loss_discriminator_mean_eval)

                    if args.set_size == "1200":
                        save_div = 50
                    elif args.set_size == "4000":
                        save_div = 20
                    elif args.set_size == "7500":
                        save_div = 10
                    else:
                        save_div = 5

                    if not idx_epoch % save_div or idx_epoch == epochs_n-1:
                        torch.save(model.state_dict(), SAVE_SUB_PATH.replace('.pth', '_' + str(idx_epoch) + '.pth'))
                        logging.info('Saving model')

                    if metrics_valid is not None:
                        for key, value in result_metrics_valid.items():
                            metric_value = value.detach().clone().item()
                            results[key].append(metric_value)

                    if metrics_train is not None:
                        for key, value in result_metrics_train.items():
                            metric_value = value.detach().clone().item()
                            results[key].append(metric_value)

                    results_to_save = pd.DataFrame(results)
                    results_to_save.to_csv(SAVE_SUB_PATH.replace(".pth", "_results.csv"), index=False)

                except KeyboardInterrupt as e:
                    logging.info(e, 'Saving plots')

                finally:

                    plt.figure(figsize=(10, 5))
                    plt.title("Discriminator Loss | Training & Eval")
                    plt.plot(results['loss_discriminator'], label="Disc Training")
                    plt.plot(results['loss_discriminator_eval'], label="Disc Eval")
                    plt.xlabel("iterations")
                    plt.ylabel("Loss")
                    plt.legend()
                    plots_dir = os.path.join(SAVE_PATH, 'plots')
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    plot_dir = os.path.join(plots_dir, 'Disc_loss')
                    plt.savefig(plot_dir)

                    plt.figure(figsize=(10, 5))
                    plt.title("Reconstruction Loss | Training & Eval")
                    plt.plot(results['loss_reconstruction'], label="Recon Training")
                    plt.plot(results['loss_reconstruction_eval'], label="Recon Eval")
                    plt.xlabel("iterations")
                    plt.ylabel("Loss")
                    plt.legend()
                    plots_dir = os.path.join(SAVE_PATH, 'plots')
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    plot_dir = os.path.join(plots_dir, 'Recon_loss')
                    plt.savefig(plot_dir)

                    plt.figure(figsize=(10, 5))
                    plt.title("Penalty Loss | Training & Eval")
                    plt.plot(results['loss_penalty'], label="Penalty Training")
                    plt.plot(results['loss_penalty_eval'], label="Penalty Eval")
                    # plt.plot(results['loss_reconstruction'], label="LR")
                    plt.xlabel("iterations")
                    plt.ylabel("Loss")
                    plt.legend()
                    plots_dir = os.path.join(SAVE_PATH, 'plots')
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    plot_dir = os.path.join(plots_dir, 'Penalty_loss')
                    plt.savefig(plot_dir)

                    for key, value in result_metrics_valid.items():
                        # metric_value = value.detach().clone().item()
                        # results[key].append(metric_value)
                        plt.figure(figsize=(10, 5))
                        plt.title("{} During Validation".format(key))
                        plt.plot(results[key], label=key)
                        # plt.plot(results['loss_discriminator_fake'], label="DF")
                        plt.xlabel("iterations")
                        plt.ylabel(key)
                        plt.legend()
                        plots_dir = os.path.join(SAVE_PATH, 'plots/valid_metrics/')
                        if not os.path.exists(plots_dir):
                            os.makedirs(plots_dir)
                        plot_dir = os.path.join(plots_dir, '{}_plot_valid'.format(key))
                        plt.savefig(plot_dir)

                    for key, value in result_metrics_train.items():
                        # metric_value = value.detach().clone().item()
                        # results[key].append(metric_value)
                        plt.figure(figsize=(10, 5))
                        plt.title("{} During Training".format(key))
                        plt.plot(results[key], label=key)
                        # plt.plot(results['loss_discriminator_fake'], label="DF")
                        plt.xlabel("iterations")
                        plt.ylabel(key)
                        plt.legend()
                        plots_dir = os.path.join(SAVE_PATH, 'plots/train_metrics/')
                        if not os.path.exists(plots_dir):
                            os.makedirs(plots_dir)
                        plot_dir = os.path.join(plots_dir, '{}_plot_train'.format(key))
                        plt.savefig(plot_dir)

                    logging.info("Plots are saved")
                    # plt.show()
                    plt.close('all')

        # Save final model
        torch.save(model.state_dict(), SAVE_SUB_PATH.replace('.pth', '_final.pth'))
        logging.info('Saving model at max iteration')

        exit(0)

    except Exception:
        logger.error("Fatal error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

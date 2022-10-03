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

from torch import nn, no_grad
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, StepLR

import training_config
from model_2 import Decoder, CognitiveEncoder, WaeGan, WaeGanCognitive
from utils_2 import GreyToColor, evaluate, PearsonCorrelation, \
    StructuralSimilarity, FmriDataloader, potentiation

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
            parser.add_argument('--lr_dec', default=.001, type=float)
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
            parser.add_argument('--lin_size', default=1024, type=int,
                                help='sets the number of nuerons in cog lin layer')
            parser.add_argument('--lin_layers', default=1, type=int, help='sets how many layers of cog network ')
            parser.add_argument('--optim_method', default='Adam',
                                help='defines method for optimizer. Options: RMS or Adam.', type=str)
            parser.add_argument('--standardize', default='True',
                                help='determines whether the dataloader uses standardize.', type=str)
            parser.add_argument('--disc_loss', default='Maria',
                                help='determines whether we use Marias loss or the paper based one for disc', type=str)
            parser.add_argument('--WAE_loss', default='Maria',
                                help='determines whether we use Marias loss or the paper based one for WAE', type=str)
            parser.add_argument('--lambda_WAE', default=1, help='sets the multiplier for paper GAN loss', type=int)
            parser.add_argument('--lambda_GAN', default=10, help='sets the multiplier for individual GAN losses',
                                type=int)
            parser.add_argument('--lambda_recon', default=1, help='weight of recon loss', type=int)
            parser.add_argument('--clip_gradients', default='False',
                                help='determines whether to clip gradients or not', type=str)
            parser.add_argument('--weight_decay', default=0.0, type=float, help='sets the weight decay for Adam')
            parser.add_argument('--momentum', default=0.9, type=float, help='sets the momentum for cog enc')

            # Pretrained/checkpoint network components
            parser.add_argument('--network_checkpoint', default=None, help='loads checkpoint in the format '
                                                                           'vaegan_20220613-014326', type=str)
            parser.add_argument('--checkpoint_epoch', default=90, help='epoch of checkpoint network', type=int)
            parser.add_argument('--load_from',default='pretrain', help='sets whether pretrained net is from pretrain'
                                                                      'or from stage_1 output', type=str)
            parser.add_argument('--st1_net', default=training_config.pretrained_net,
                                help='pretrained network from stage 1', type=str)
            parser.add_argument('--st1_load_epoch', default='final',
                                help='epoch of the pretrained model', type=str)
            parser.add_argument('--st2_net', default=training_config.pretrained_net,
                                help='pretrained network from stage 1', type=str)
            parser.add_argument('--st2_load_epoch', default='final',
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
            from hidden import args

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
            elif args.set_size == 'single_pres':
                TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'train', args.set_size,
                                               'Subj_0{}_NSD_{}_train.pickle'.format(args.subject, args.set_size))
            else:
                # For loading 1200, 4000, 7500
                TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'train/single_pres', args.set_size,
                                               'Subj_0{}_{}_NSD_single_pres_train.pickle'.format(args.subject,
                                                                                                 args.set_size))

            # Currently valid data is set to 'max' meaning validation data contains multiple image presentations
            # If you only want to evaluate a single presentation of images replace both 'max' in strings below ...
            # with 'single_pres'
            # VALID_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'valid', 'max', args.ROI,
            #                                'Subj_0{}_NSD_max_valid.pickle'.format(args.subject))
            VALID_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'valid/single_pres', args.ROI,
                                           'Subj_0{}_NSD_single_pres_valid.pickle'.format(args.subject))

        else:
            TRAIN_DATA_PATH = os.path.join(args.data_root, 'GOD',
                                           'GOD_Subject{}_train_normed.pickle'.format(args.subject))
            VALID_DATA_PATH = os.path.join(args.data_root, 'GOD',
                                           'GOD_Subject{}_valid_normed.pickle'.format(args.subject))

        # Create directory for results
        stage_num = 'stage_3'
        stage = 3

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
        st1_model_dir = os.path.join(OUTPUT_PATH, 'NSD', 'stage_1', args.st1_net,
                                 'stage_1_WAE_' + args.st1_net + '_{}.pth'.format(args.st1_load_epoch))

        if args.load_from == 'pretrain':
            # Used if we didn't do pretrain -> stage 1, but just pretrain and use that.
            st1_model_dir = os.path.join(OUTPUT_PATH, 'NSD', 'pretrain', args.st1_net,
                                     'pretrained_WAE_' + args.st1_net + '_{}.pth'.format(args.st1_load_epoch))

        teacher_model = WaeGan(device=device, z_size=args.latent_dims).to(device)
        teacher_model.load_state_dict(torch.load(st1_model_dir, map_location=device))
        # this loads teacher model with model from stage 1 - TODO: insert here
        for param in teacher_model.encoder.parameters():
            param.requires_grad = False

        # Load Stage 2 network weights
        st2_model_dir = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, args.ROI, SUBJECT_PATH,
                                     'stage_2', args.st2_net, args.st2_net + '_{}.pth'.format(args.st2_load_epoch))

        decoder = Decoder(z_size=args.latent_dims, size=256).to(device)
        cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=args.latent_dims, lin_size=args.lin_size,
                                             lin_layers=args.lin_layers, momentum=args.momentum).to(device)
        trained_model = WaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=decoder,
                                        z_size=args.latent_dims).to(device)
        # This then loads the network from stage 2 (cog enc) to fix encoder in stage 3
        # It uses the components from stage 2 and builds from here for main model
        trained_model.load_state_dict(torch.load(st2_model_dir, map_location=device))

        # Here we want to load the stage 1 discriminator that is good at moving the enc towards sampled
        # Maria uses a fresh discriminator but that doesn't make sense
        if args.disc_loss == "David":
            # Here since we're moving the latent to the sampled, we borrow the stage 1 disc which is good at classifying
            model = WaeGanCognitive(device=device, encoder=trained_model.encoder, decoder=trained_model.decoder,
                                    discriminator=teacher_model.discriminator, z_size=args.latent_dims).to(device)
        elif args.disc_loss == "Both":
            # Here since we want the latent to be more like the visual encoder, we use the stage 2 that has been trained
            # ... on the vis enc latent classification
            model = WaeGanCognitive(device=device, encoder=trained_model.encoder, decoder=trained_model.decoder,
                                    discriminator=trained_model.discriminator, z_size=args.latent_dims).to(device)
        else:
            # Maria's
            model = WaeGanCognitive(device=device, encoder=trained_model.encoder, decoder=trained_model.decoder,
                                    z_size=args.latent_dims).to(device)

        # Fix encoder weights
        for param in model.encoder.parameters():
            param.requires_grad = False

        logging.info(f'Number of voxels: {NUM_VOXELS}')
        logging.info(f'Train data length: {len(train_data)}')
        logging.info(f'Validation data length: {len(valid_data)}')

        # Loading Checkpoint | If you want to continue training for existing checkpoint
        # Set checkpoint path
        # CHECKPOINT CODE NOT WORKIGN RIGHT NOW
        if args.network_checkpoint is not None:
            net_checkpoint_path = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, SUBJECT_PATH,
                                               'stage_2', args.network_checkpoint,
                                               'stage_2_vaegan_' + args.network_checkpoint + '.pth')
            print(net_checkpoint_path)

        # Load and show results for checkpoint
        if args.network_checkpoint is not None and os.path.exists(net_checkpoint_path.replace(".pth", "_results.csv")):
            logging.info('Load pretrained model')
            checkpoint_dir = net_checkpoint_path.replace(".pth", '_{}.pth'.format(args.checkpoint_epoch))
            model.load_state_dict(torch.load(checkpoint_dir))
            model.eval()
            results = pd.read_csv(net_checkpoint_path.replace(".pth", "_results.csv"))
            results = {col_name: list(results[col_name].values) for col_name in results.columns}
            stp = 1 + len(results['epochs'])
            # Calculates the lr at time of checkpoint
            # TODO: potentiation breaks here because two lrs.
            lr = potentiation(lr, args.decay_lr, args.checkpoint_epoch)
            if training_config.evaluate:
                images_dir = os.path.join(SAVE_PATH, 'images')
                if not os.path.exists(images_dir):
                    os.makedirs(images_dir)
                pcc, ssim, mse, is_mean = evaluate(model, dataloader_valid, norm=True, mean=training_config.mean,
                                                   std=training_config.std,
                                                   path=images_dir)
                print("Mean PCC:", pcc)
                print("Mean SSIM:", ssim)
                print("Mean MSE:", mse)
                print("IS mean", is_mean)
                exit(0)
        else:
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
        lr_dec = args.lr_dec  # 0.001 - Maria | drop this here - 0.0001/3
        lr_disc = args.lr_disc  # 0.0005 - Maria | drop this here I imagine, 0.0001 or 0.00005
        beta = args.beta

        if args.optim_method == 'RMS':
            # optimizer_encoder = torch.optim.RMSprop(params=model.encoder.parameters(), lr=lr_enc, alpha=0.9,)
            optimizer_decoder = torch.optim.RMSprop(params=model.decoder.parameters(), lr=lr_dec,alpha=0.9)
            optimizer_discriminator = torch.optim.RMSprop(params=model.discriminator.parameters(), lr=lr_disc, alpha=0.9)
            # lr_encoder = ExponentialLR(optimizer_encoder, gamma=args.decay_lr)
            lr_decoder = ExponentialLR(optimizer_decoder, gamma=args.decay_lr)
            lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=args.decay_lr)

        else:
            # Optimizers
            # optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lr_enc, betas=(beta, 0.999))
            optimizer_decoder = torch.optim.Adam(model.decoder.parameters(), lr=lr_dec, betas=(beta, 0.999),
                                                 weight_decay=args.weight_decay)
            optimizer_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=lr_disc, betas=(beta, 0.999),
                                                       weight_decay=args.weight_decay)

            # lr_encoder = StepLR(optimizer_encoder, step_size=30, gamma=0.5)
            lr_decoder = StepLR(optimizer_decoder, step_size=30, gamma=0.5)
            lr_discriminator = StepLR(optimizer_discriminator, step_size=30, gamma=0.5)

        # Define criterion
        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss(reduction='none')

        # Metrics
        pearson_correlation = PearsonCorrelation()
        structural_similarity = StructuralSimilarity(mean=training_config.mean, std=training_config.std)
        # mse_loss = nn.MSELoss()

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
                            frozen_params(model.encoder)
                            batch_size = len(data_batch)
                            model.decoder.zero_grad()
                            model.discriminator.zero_grad()

                            x_fmri = Variable(data_batch['fmri'], requires_grad=False).float().to(device)
                            x_image = Variable(data_batch['image'], requires_grad=False).float().to(device)
                            # z, _ = trained_model.encoder(x_image)
                            # x_gt = trained_model.decoder(z)

                            # ----------Train discriminator-------------

                            frozen_params(model.decoder)
                            free_params(model.discriminator)

                            if args.disc_loss == "Maria":
                                z_cog_enc, var = model.encoder(x_fmri)
                                z_vis_enc, var = teacher_model.encoder(x_image)
                                # z_cog_enc, var = model.encoder(x_fmri)
                                # z_cog_enc = Variable(torch.randn_like(z_vis_enc) * 0.5).to(device)

                                # WAE | Trying to make Qz more like Pz
                                # disc output for encoded latent (cog) | Qz
                                logits_cog_enc = model.discriminator(z_cog_enc)
                                # disc output for visual encoder | Pz
                                logits_vis_enc = model.discriminator(z_vis_enc)

                                sig_cog_enc = torch.sigmoid(logits_cog_enc)
                                sig_vis_enc = torch.sigmoid(logits_vis_enc)

                                loss_discriminator_fake = - args.lambda_GAN * torch.sum(torch.log(sig_cog_enc + 1e-3))
                                loss_discriminator_real = - args.lambda_GAN * torch.sum(torch.log(1 - sig_vis_enc + 1e-3))

                                loss_discriminator = loss_discriminator_real + loss_discriminator_real

                                loss_discriminator_fake.backward(retain_graph=True, inputs=list(model.discriminator.parameters()))
                                loss_discriminator_real.backward(retain_graph=True, inputs=list(model.discriminator.parameters()))
                                mean_mult = batch_size * args.lambda_GAN
                            elif args.disc_loss == "Maria_Flip":
                                z_cog_enc, var = model.encoder(x_fmri)
                                z_vis_enc, var = teacher_model.encoder(x_image)
                                # z_cog_enc, var = model.encoder(x_fmri)
                                # z_cog_enc = Variable(torch.randn_like(z_vis_enc) * 0.5).to(device)

                                # WAE | Trying to make Qz more like Pz
                                # disc output for encoded latent (cog) | Qz
                                logits_cog_enc = model.discriminator(z_cog_enc)
                                # disc output for visual encoder | Pz
                                logits_vis_enc = model.discriminator(z_vis_enc)

                                sig_cog_enc = torch.sigmoid(logits_cog_enc)
                                sig_vis_enc = torch.sigmoid(logits_vis_enc)

                                loss_discriminator_fake = - args.lambda_GAN * torch.sum(torch.log(sig_vis_enc + 1e-3))
                                loss_discriminator_real = - args.lambda_GAN * torch.sum(torch.log(1 - sig_cog_enc + 1e-3))

                                loss_discriminator = loss_discriminator_real + loss_discriminator_real

                                loss_discriminator_fake.backward(retain_graph=True, inputs=list(model.discriminator.parameters()))
                                loss_discriminator_real.backward(retain_graph=True, inputs=list(model.discriminator.parameters()))
                                mean_mult = batch_size * args.lambda_GAN
                            elif args.disc_loss == "Both": # We used this one
                                # SINCE WE DON'T UPDATE THE ENCODER HERE, THIS HAS NO ROLE
                                # IT TRAINS THE DISCRIMINATOR BUT THAT DOESN'T INFORM ANYTHING.
                                # This will be as Maria's but with the discriminator fixed
                                z_cog_enc, var = model.encoder(x_fmri)
                                z_vis_enc, var = teacher_model.encoder(x_image)
                                # z_cog_enc, var = model.encoder(x_fmri)
                                # z_cog_enc = Variable(torch.randn_like(z_vis_enc) * 0.5).to(device)

                                # WAE | Trying to make Qz more like Pz
                                # disc output for encoded latent (cog) | Qz
                                logits_cog_enc = model.discriminator(z_cog_enc)
                                # disc output for visual encoder | Pz
                                logits_vis_enc = model.discriminator(z_vis_enc)

                                bce_loss = nn.BCEWithLogitsLoss(reduction='none')

                                # set up labels
                                labels_real = Variable(torch.ones_like(logits_vis_enc)).to(device)
                                labels_fake = Variable(torch.zeros_like(logits_cog_enc)).to(device)

                                # Qz is distribution of encoded latent space
                                loss_Qz = args.lambda_GAN * torch.sum(bce_loss(logits_cog_enc, labels_fake))
                                # Pz is distribution of prior (sampled)
                                loss_Pz = args.lambda_GAN * torch.sum(bce_loss(logits_vis_enc, labels_real))

                                loss_discriminator = args.lambda_WAE * (loss_Qz + loss_Pz)
                                loss_discriminator.backward(retain_graph=True,
                                                            inputs=list(model.discriminator.parameters()))
                                mean_mult = batch_size * args.lambda_GAN
                            elif args.disc_loss == "David":
                                # My logic here is that like Ren's work, here we want to use the reals to refine
                                # Similarly, we want to get the latent space closer to that inital prior now
                                # Rather than that of the vis_enc - I'm loosely attached to this idea
                                # Though it's possible it's still better just to get it closer to the vis_enc latent
                                z_cog_enc, var = model.encoder(x_fmri)
                                z_samp = Variable(torch.randn_like(z_cog_enc) * 0.5).to(device)
                                # z_vis_enc, var = teacher_model.encoder(x_image)
                                # z_cog_enc, var = model.encoder(x_fmri)
                                # z_cog_enc = Variable(torch.randn_like(z_vis_enc) * 0.5).to(device)

                                # WAE | Trying to make Qz more like Pz
                                # disc output for encoded latent (cog) | Qz
                                logits_cog_enc = model.discriminator(z_cog_enc)
                                # disc output for visual encoder | Pz
                                logits_samp = model.discriminator(z_samp)

                                bce_loss = nn.BCEWithLogitsLoss(reduction='none')

                                # set up labels
                                labels_real = Variable(torch.ones_like(logits_samp)).to(device)
                                labels_fake = Variable(torch.zeros_like(logits_cog_enc)).to(device)

                                # Qz is distribution of encoded latent space
                                loss_Qz = args.lambda_GAN * torch.sum(bce_loss(logits_cog_enc, labels_fake))
                                # Pz is distribution of prior (sampled)
                                loss_Pz = args.lambda_GAN * torch.sum(bce_loss(logits_samp, labels_real))

                                loss_discriminator = args.lambda_WAE * (loss_Qz + loss_Pz)
                                loss_discriminator.backward(retain_graph=True,
                                                            inputs=list(model.discriminator.parameters()))
                                mean_mult = batch_size * args.lambda_GAN

                            # loss_discriminator.backward(retain_graph=True)
                            # [p.grad.data.clamp_(-1, 1) for p in model.discriminator.parameters()]
                            optimizer_discriminator.step()

                            # ----------Train generator----------------
                            model.decoder.zero_grad()

                            free_params(model.decoder)
                            frozen_params(model.discriminator)

                            z_cog_enc, var = model.encoder(x_fmri)
                            x_recon = model.decoder(z_cog_enc)
                            logits_cog_enc = model.discriminator(z_cog_enc)

                            if args.WAE_loss == "Maria": # We used this one
                                mse_loss = nn.MSELoss()
                                loss_reconstruction = mse_loss(x_recon, x_image) * args.lambda_recon
                                mean_mult = 1 * args.lambda_recon
                            else:
                                loss_reconstruction = torch.sum(torch.sum(0.5 * (x_recon - x_image) ** 2, 1))
                                mean_mult = batch_size

                            loss_reconstruction.backward(inputs=list(model.decoder.parameters()))
                            # loss_penalty.backward()
                            # [p.grad.data.clamp_(-1, 1) for p in model.encoder.parameters()]
                            optimizer_decoder.step()

                            # loss_penalty just for reporting
                            bce_loss = nn.BCEWithLogitsLoss(reduction='none')
                            labels_saturated = Variable(torch.ones_like(logits_cog_enc)).to(device)
                            loss_penalty = args.lambda_GAN * torch.sum(bce_loss(logits_cog_enc, labels_saturated))

                            # register mean values of the losses for logging
                            loss_reconstruction_mean = loss_reconstruction.data.cpu().numpy() / mean_mult
                            loss_penalty_mean = loss_penalty.data.cpu().numpy() / mean_mult * args.lambda_GAN
                            loss_discriminator_mean = loss_discriminator.data.cpu().numpy() / mean_mult

                            # Turning this off just to get logs clutter free
                            # logging.info(
                            #     f'Epoch  {idx_epoch} {batch_idx + 1:3.0f} / {100 * (batch_idx + 1) / len(dataloader_train):2.3f}%, '
                            #     f'---- recon loss: {loss_reconstruction_mean:.5f} ---- | '
                            #     f'---- penalty loss: {loss_penalty_mean:.5f} ---- | '
                            #     f'---- discrim fake loss: {loss_discriminator_mean:.5f}')

                            step_index += 1
                        else:
                            break

                    # EPOCH END
                    # lr_encoder.step()
                    lr_decoder.step()
                    lr_discriminator.step()

                    # Record losses & scores
                    results['epochs'].append(idx_epoch + stp)
                    results['loss_reconstruction'].append(loss_reconstruction_mean)
                    results['loss_penalty'].append(loss_penalty_mean)
                    results['loss_discriminator'].append(loss_discriminator_mean)

                    # plot arrangements
                    # grid_count = 25
                    nrow = 5
                    # if batch_size == 64:
                    #     grid_count = 25
                    # nrow = int(math.sqrt(grid_count))

                    if args.set_size in ("1200", "4000", "7500"):
                        train_recon_freq = 150
                    else:
                        train_recon_freq = 20

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
                        ax.set_title('Training Reconstruction at Epoch {}'.format(idx_epoch))
                        ax.imshow(make_grid(x_recon.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                        output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                        plt.savefig(output_dir)

                    logging.info('Evaluation')

                    for batch_idx, data_batch in enumerate(dataloader_valid):
                        # model.eval()

                        with no_grad():

                            model.eval()
                            trained_model.eval()
                            teacher_model.eval()

                            data_in = Variable(data_batch['fmri'], requires_grad=False).float().to(device)
                            data_target = Variable(data_batch['image'], requires_grad=False).float().to(device)
                            out, logits_out = model(data_in)

                            bce_loss = nn.BCEWithLogitsLoss(reduction='none')

                            labels_saturated_eval = Variable(torch.ones_like(logits_out, requires_grad=False)).to(
                                device)

                            # Note: below will not match training if using Maria. Assumes, David + Both | or both + both
                            # Discriminator loss
                            if args.disc_loss == "Maria":
                                _, z_target = teacher_model(data_target)
                                logits_target = model.discriminator(z_target)

                                sig_out = torch.sigmoid(logits_out)
                                sig_target = torch.sigmoid(logits_target)
                                loss_out_fake = - args.lambda_GAN * torch.sum(torch.log(sig_out + 1e-3))
                                loss_out_real = - args.lambda_GAN * torch.sum(torch.log(1 - sig_target + 1e-3))

                                loss_discriminator_eval = loss_out_fake + loss_out_real
                                loss_discriminator_mean_eval = loss_discriminator_eval / (batch_size * args.lambda_GAN)
                            elif args.disc_loss == "Maria_Flip":
                                _, z_target = teacher_model(data_target)
                                logits_target = model.discriminator(z_target)

                                sig_out = torch.sigmoid(logits_out)
                                sig_target = torch.sigmoid(logits_target)
                                loss_out_fake = - args.lambda_GAN * torch.sum(torch.log(sig_target + 1e-3))
                                loss_out_real = - args.lambda_GAN * torch.sum(torch.log(1 - sig_out + 1e-3))

                                loss_discriminator_eval = loss_out_fake + loss_out_real
                                loss_discriminator_mean_eval = loss_discriminator_eval / (batch_size * args.lambda_GAN)
                            elif args.disc_loss == "David":
                                z_target = Variable(torch.randn_like(z_cog_enc, requires_grad=False) * 0.5).to(device)
                                logits_target = model.discriminator(z_target)

                                labels_real_eval = Variable(torch.ones_like(logits_target, requires_grad=False)).to(device)
                                labels_fake_eval = Variable(torch.zeros_like(logits_out, requires_grad=False)).to(device)

                                loss_out_fake = args.lambda_GAN * torch.sum(bce_loss(logits_out, labels_fake_eval))
                                loss_target_real = args.lambda_GAN * torch.sum(bce_loss(logits_target, labels_real_eval))
                                mean_mult = batch_size * args.lambda_GAN
                                loss_discriminator_mean_eval = (loss_out_fake + loss_target_real) / mean_mult
                            else:
                                _, z_target = teacher_model(data_target)
                                logits_target = model.discriminator(z_target)
                                labels_real_eval = Variable(torch.ones_like(logits_target, requires_grad=False)).to(
                                    device)
                                labels_fake_eval = Variable(torch.zeros_like(logits_out, requires_grad=False)).to(
                                    device)
                                loss_out_fake = args.lambda_GAN * torch.sum(bce_loss(logits_out, labels_fake_eval))
                                loss_target_real = args.lambda_GAN * torch.sum(bce_loss(logits_target, labels_real_eval))
                                mean_mult = batch_size * args.lambda_GAN
                                loss_discriminator_mean_eval = (loss_out_fake + loss_target_real) / mean_mult

                            # Recon loss
                            if args.WAE_loss == "Maria": # and MSE
                                bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
                                mse_loss = nn.MSELoss(reduction='mean')

                                loss_reconstruction_eval = mse_loss(out, data_target) * args.lambda_recon
                                loss_reconstruction_mean_eval = loss_reconstruction_eval / args.lambda_recon

                                loss_penalty_eval = args.lambda_GAN * bce_loss(logits_out, labels_saturated_eval)
                                loss_penalty_mean_eval = loss_penalty_eval
                            else: # both
                                bce_loss = nn.BCEWithLogitsLoss(reduction='none')
                                loss_reconstruction_eval = torch.sum(torch.sum(0.5 * (out - data_target) ** 2, 1))
                                loss_reconstruction_mean_eval = loss_reconstruction_eval / batch_size

                                loss_penalty_eval = args.lambda_GAN * torch.sum(bce_loss(logits_out, labels_saturated_eval))

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
                                        result_metrics_train[key] = metric(x_recon, x_image).mean()
                                    else:
                                        result_metrics_train[key] = metric(x_recon, x_image)

                            out = out.data.cpu()

                        # Save validation examples
                        images_dir = os.path.join(SAVE_PATH, 'images', 'valid')
                        if not os.path.exists(images_dir):
                            os.makedirs(images_dir)
                            os.makedirs(os.path.join(images_dir, 'random'))

                        out = out.data.cpu()

                        if args.set_size in ("1200", "4000"):
                            eval_recon_freq = 400
                        elif args.set_size == "7500":
                            eval_recon_freq = 40
                        else:
                            eval_recon_freq = 20

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

                        else:  # valid_shuffle is false, so same images are shown
                            if idx_epoch == 0:
                                fig, ax = plt.subplots(figsize=(10, 10))
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.set_title('Validation Ground Truth')
                                ax.imshow(make_grid(data_target.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                                gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                                plt.savefig(gt_dir)

                        if not idx_epoch % eval_recon_freq:
                            fig, ax = plt.subplots(figsize=(10, 10))
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_title('Validation Reconstruction at Epoch {}'.format(idx_epoch))
                            ax.imshow(make_grid(out.cpu().detach(), nrow=nrow, normalize=True).permute(1, 2, 0))
                            output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                            plt.savefig(output_dir)

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

                    if args.set_size == "1200":
                        save_div = 1000
                    elif args.set_size == "4000":
                        save_div = 400
                    elif args.set_size == "7500":
                        save_div = 200
                    else:
                        save_div = 50

                    if not idx_epoch % save_div or idx_epoch == epochs_n-1:
                        torch.save(model.state_dict(), SAVE_SUB_PATH.replace('.pth', '_' + str(idx_epoch) + '.pth'))
                        logging.info('Saving model')

                    # Record losses & scores
                    results['loss_reconstruction_eval'].append(loss_reconstruction_mean_eval)
                    results['loss_penalty_eval'].append(loss_penalty_mean_eval)
                    results['loss_discriminator_eval'].append(loss_discriminator_mean_eval)

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

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
            parser.add_argument('--vox_res', default='1.8mm', help='1.8mm, 3mm', type=str)
            # Probably only needed stage 2/3 (though do we want to change stage 1 - depends on whether we do the full Ren...
            # Thing where we do a smaller set for stage 1. But I think I might do a Maria and just have stage 1 with no...
            # pretraining phase...
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
            if args.set_size == 'max' or args.set_size == 'single_pres':
                TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'train', args.set_size,
                                           'Subj_0{}_NSD_{}_train.pickle'.format(args.subject, args.set_size))
            else:
                TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'train', args.set_size,
                                           'Subj_0{}_{}_NSD_single_pres_train.pickle'.format(args.subject, args.set_size))

            # Currently valid data is set to 'max' meaning validation data contains multiple image presentations
            # If you only want to evaluate a single presentation of images replace both 'max' in strings below ...
            # with 'single_pres'
            VALID_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'valid', 'max',
                                           'Subj_0{}_NSD_max_valid.pickle'.format(args.subject))
        else:
            TRAIN_DATA_PATH = os.path.join(args.data_root, 'GOD',
                                           'GOD_Subject{}_train_normed.pickle'.format(args.subject))
            VALID_DATA_PATH = os.path.join(args.data_root, 'GOD',
                                           'GOD_Subject{}_valid_normed.pickle'.format(args.subject))

        # Create directory for results
        stage_num = 'stage_2'
        stage = 2

        SAVE_PATH = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, SUBJECT_PATH, stage_num,
                                 args.run_name)
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        SAVE_SUB_PATH = os.path.join(SAVE_PATH, 'stage_2_WAE_{}.pth'.format(args.run_name))
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

        # Load data
        training_data = FmriDataloader(dataset=train_data, root_path=root_path,
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

        validation_data = FmriDataloader(dataset=valid_data, root_path=root_path,
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
        st2_model_dir = os.path.join(OUTPUT_PATH, 'NSD', 'stage_2', args.st2_net,
                                     'stage_2_WAE_' + args.st2_net + '_{}.pth'.format(args.st2_load_epoch))

        decoder = Decoder(z_size=args.latent_dims, size=256).to(device)
        cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=args.latent_dims).to(device)
        trained_model = WaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=decoder,
                                        z_size=args.latent_dims).to(device)
        # This then loads the network from stage 2 (cog enc) to fix encoder in stage 3
        # It uses the components from stage 2 and builds from here for main model
        trained_model.load_state_dict(torch.load(st2_model_dir, map_location=device))
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
            loss_discriminator_fake=[],
            loss_discriminator_real=[]
        )

        # Variables for equilibrium to improve GAN stability
        lr_dec = args.lr_dec  # 0.001 - Maria
        lr_disc = args.lr_disc  # 0.0005 - Maria
        beta = args.beta

        # Optimizers
        # optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lr_enc, betas=(beta, 0.999))
        optimizer_decoder = torch.optim.Adam(model.decoder.parameters(), lr=lr_dec, betas=(beta, 0.999))
        optimizer_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=lr_disc, betas=(beta, 0.999))

        # lr_encoder = StepLR(optimizer_encoder, step_size=30, gamma=0.5)
        lr_decoder = StepLR(optimizer_decoder, step_size=30, gamma=0.5)
        lr_discriminator = StepLR(optimizer_discriminator, step_size=30, gamma=0.5)

        # Metrics
        pearson_correlation = PearsonCorrelation()
        structural_similarity = StructuralSimilarity(mean=training_config.mean, std=training_config.std)
        mse_loss = nn.MSELoss()

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

                            z_fake, var = model.encoder(x_fmri)
                            z_real, var = teacher_model.encoder(x_image)
                            # z_fake, var = model.encoder(x_fmri)
                            # z_fake = Variable(torch.randn_like(z_real) * 0.5).to(device)

                            d_real = model.discriminator(z_real)
                            d_fake = model.discriminator(z_fake)

                            loss_discriminator_fake = - 10 * torch.sum(torch.log(d_fake + 1e-3))
                            loss_discriminator_real = - 10 * torch.sum(torch.log(1 - d_real + 1e-3))
                            loss_discriminator_fake.backward(retain_graph=True, inputs=list(model.discriminator.parameters()))
                            loss_discriminator_real.backward(retain_graph=True, inputs=list(model.discriminator.parameters()))

                            # loss_discriminator.backward(retain_graph=True)
                            # [p.grad.data.clamp_(-1, 1) for p in model.discriminator.parameters()]
                            optimizer_discriminator.step()

                            # ----------Train generator----------------
                            # TODO: not sure if this helps - check the back prop method in old stage 2
                            # it should be fine because they're repushing through the vars through model
                            # model.decoder.zero_grad()

                            free_params(model.decoder)
                            frozen_params(model.discriminator)

                            z_real, var = model.encoder(x_fmri)
                            x_recon = model.decoder(z_real)
                            d_real = model.discriminator(z_real)

                            if args.recon_loss == 'manual':
                                loss_reconstruction = torch.sum(torch.sum(0.5 * (x_recon - x_image) ** 2, 1))
                            else:  # trad
                                mse_loss = nn.MSELoss()
                                loss_reconstruction = mse_loss(x_recon, x_image)

                            loss_penalty = - 10 * torch.mean(torch.log(d_real + 1e-3))

                            loss_reconstruction.backward(retain_graph=True, inputs=list(model.decoder.parameters()))
                            # loss_penalty.backward()
                            # [p.grad.data.clamp_(-1, 1) for p in model.encoder.parameters()]
                            optimizer_decoder.step()

                            # register mean values of the losses for logging
                            loss_reconstruction_mean = loss_reconstruction.data.cpu().numpy() / batch_size
                            loss_penalty_mean = loss_penalty.data.cpu().numpy() / batch_size
                            loss_discriminator_fake_mean = loss_discriminator_fake.data.cpu().numpy() / batch_size
                            loss_discriminator_real_mean = loss_discriminator_real.data.cpu().numpy() / batch_size

                            logging.info(
                                f'Epoch  {idx_epoch} {batch_idx + 1:3.0f} / {100 * (batch_idx + 1) / len(dataloader_train):2.3f}%, '
                                f'---- recon loss: {loss_reconstruction_mean:.5f} ---- | '
                                f'---- penalty loss: {loss_penalty_mean:.5f} ---- | '
                                f'---- discrim fake loss: {loss_discriminator_fake:.5f} ---- | '
                                f'---- discrim real loss: {loss_discriminator_real:.5f}')

                            step_index += 1
                        else:
                            break

                    # EPOCH END
                    # lr_encoder.step()
                    lr_decoder.step()
                    lr_discriminator.step()

                    if not idx_epoch % 2:
                        # Save train examples
                        images_dir = os.path.join(SAVE_PATH, 'images', 'train')
                        if not os.path.exists(images_dir):
                            os.makedirs(images_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Training Ground Truth at Epoch {}'.format(idx_epoch))
                        ax.imshow(make_grid(x_image[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                        gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                        plt.savefig(gt_dir)

                        # fig, ax = plt.subplots(figsize=(10, 10))
                        # ax.set_xticks([])
                        # ax.set_yticks([])
                        # ax.set_title('Training Vis Enc Reconstruction (Real) at Epoch {}'.format(idx_epoch))
                        # ax.imshow(
                        #     make_grid(x_gt[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                        # gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_vis_output_' + 'grid')
                        # plt.savefig(gt_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Training Reconstruction at Epoch {}'.format(idx_epoch))
                        ax.imshow(make_grid(x_recon[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                        output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                        plt.savefig(output_dir)

                    logging.info('Evaluation')

                    for batch_idx, data_batch in enumerate(dataloader_valid):
                        # model.eval()

                        with no_grad():

                            model.eval()
                            data_in = Variable(data_batch['fmri'], requires_grad=False).float().to(device)
                            data_target = Variable(data_batch['image'], requires_grad=False).float().to(device)
                            out = model(data_in)

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

                        if shuf:
                            fig, ax = plt.subplots(figsize=(10, 10))
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_title('Validation Ground Truth')
                            ax.imshow(
                                make_grid(data_target[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                            gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                            plt.savefig(gt_dir)

                            # fig, ax = plt.subplots(figsize=(10, 10))
                            # ax.set_xticks([])
                            # ax.set_yticks([])
                            # ax.set_title('Validation Vis Enc Reconstruction at Epoch {}'.format(idx_epoch))
                            # ax.imshow(make_grid(vis_out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                            # output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_vis_output_' + 'grid')
                            # plt.savefig(output_dir)

                        else:  # valid_shuffle is false, so same images are shown
                            if idx_epoch == 0:
                                fig, ax = plt.subplots(figsize=(10, 10))
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.set_title('Validation Ground Truth')
                                ax.imshow(make_grid(data_target[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                                gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                                plt.savefig(gt_dir)

                                # fig, ax = plt.subplots(figsize=(10, 10))
                                # ax.set_xticks([])
                                # ax.set_yticks([])
                                # ax.set_title('Validation Vis Enc Reconstruction at Epoch {}'.format(idx_epoch))
                                # ax.imshow(
                                #     make_grid(vis_out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                                # output_dir = os.path.join(images_dir,
                                #                           'epoch_' + str(idx_epoch) + '_vis_output_' + 'grid')
                                # plt.savefig(output_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Validation Reconstruction at Epoch {}'.format(idx_epoch))
                        ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                        output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                        plt.savefig(output_dir)

                        out = (out + 1) / 2
                        out = make_grid(out, nrow=8)

                        # out = model(None, 100)
                        # out = out.data.cpu()
                        # out = (out + 1) / 2
                        # out = make_grid(out, nrow=8)
                        # writer.add_image("generated", out, step_index)

                        out = data_target.data.cpu()
                        out = (out + 1) / 2
                        out = make_grid(out, nrow=8)

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

                    if not idx_epoch % 10 or idx_epoch == epochs_n-1:
                        torch.save(model.state_dict(), SAVE_SUB_PATH.replace('.pth', '_' + str(idx_epoch) + '.pth'))
                        logging.info('Saving model')

                    # Record losses & scores
                    results['epochs'].append(idx_epoch + stp)
                    results['loss_reconstruction'].append(loss_reconstruction_mean)
                    results['loss_penalty'].append(loss_penalty_mean)
                    results['loss_discriminator_fake'].append(loss_discriminator_fake_mean)
                    results['loss_discriminator_real'].append(loss_discriminator_real_mean)

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
                    plt.title("Discriminator Loss During Training")
                    plt.plot(results['loss_discriminator_real'], label="DR")
                    plt.plot(results['loss_discriminator_fake'], label="DF")
                    plt.xlabel("iterations")
                    plt.ylabel("Loss")
                    plt.legend()
                    plots_dir = os.path.join(SAVE_PATH, 'plots')
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    plot_dir = os.path.join(plots_dir, 'Disc_loss')
                    plt.savefig(plot_dir)

                    plt.figure(figsize=(10, 5))
                    plt.title("Reconstruction Loss During Training")
                    plt.plot(results['loss_penalty'], label="Penalty")
                    plt.plot(results['loss_reconstruction'], label="Reconstruction")
                    plt.xlabel("iterations")
                    plt.ylabel("Loss")
                    plt.legend()
                    plots_dir = os.path.join(SAVE_PATH, 'plots')
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    plot_dir = os.path.join(plots_dir, 'Recon_loss')
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

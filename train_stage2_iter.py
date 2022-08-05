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
from torch.optim.lr_scheduler import ExponentialLR

import training_config
from model_2 import VaeGan, Encoder, Decoder, VaeGanCognitive, Discriminator, CognitiveEncoder
from utils_2 import GreyToColor, evaluate, PearsonCorrelation, \
    StructuralSimilarity, objective_assessment, parse_args, FmriDataloader, potentiation

if __name__ == "__main__":
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
            parser.add_argument('--lr_enc', default=.0001, type=float)
            parser.add_argument('--lr_disc', default=.00001, type=float)
            parser.add_argument('--decay_lr', default=training_config.decay_lr,
                                help='.98 in Maria, .75 in original VAE/GAN', type=float)
            parser.add_argument('--equilibrium_game', default='n', type=str,
                                help='Sets whether to engage the equilibrium game for decoder/disc updates (y/n)')
            parser.add_argument('--backprop_method', default='clip', help='trad sets three diff loss functions,'
                                                                          'but clip, clips the gradients to help'
                                                                          'avoid the late spikes in loss', type=str)
            parser.add_argument('--seed', default=277603, help='sets seed, 0 makes a random int', type=int)
            parser.add_argument('--valid_shuffle', '-shuffle', default='False', type=str, help='defines whether'
                                                                                               'eval dataloader shuffles')

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

        SAVE_SUB_PATH = os.path.join(SAVE_PATH, 'stage_2_vaegan_{}.pth'.format(args.run_name))
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

        # This writer function is for torch.tensorboard
        writer = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name)
        writer_encoder = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/encoder')
        writer_decoder = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/decoder')
        writer_discriminator = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/discriminator')

        # Load Stage 1 network weights
        model_dir = os.path.join(OUTPUT_PATH, 'NSD', 'stage_1', args.pretrained_net,
                                   'stage_1_vaegan_' + args.pretrained_net + '_{}.pth'.format(args.load_epoch))

        if args.load_from == 'pretrain':
            # Used if we didn't do pretrain -> stage 1, but just pretrain and use that.
            model_dir = os.path.join(OUTPUT_PATH, 'NSD', 'pretrain', args.pretrained_net,
                                     'pretrained_vaegan_' + args.pretrained_net + '_{}.pth'.format(args.load_epoch))

        logging.info('Loaded network is:', model_dir)

        # Initialize network and load Stage 1 weights
        teacher_model = VaeGan(device=device, z_size=training_config.latent_dim).to(device)
        logging.info('Loading model from Stage 1')
        teacher_model.load_state_dict(torch.load(model_dir, map_location=device))
        discriminator = teacher_model.discriminator
        teacher_model.discriminator.train() # Not discriminator.train()?

        # Fix decoder weights
        for param in teacher_model.decoder.parameters():
            param.requires_grad = False

        logging.info(f'Validation data length: {NUM_VOXELS}')
        logging.info(f'Validation data length: {len(train_data)}')
        logging.info(f'Validation data length: {len(valid_data)}')

        # Define model
        cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=training_config.latent_dim).to(device)
        model = VaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=teacher_model.decoder,
                                discriminator=discriminator, teacher_net=teacher_model, stage=2,
                                z_size=training_config.latent_dim).to(device)

        # Variables for equilibrium to improve GAN stability
        margin = training_config.margin
        equilibrium = training_config.equilibrium
        lambda_mse = training_config.lambda_mse
        decay_mse = training_config.decay_mse
        lr_enc = args.lr_enc
        lr_disc = args.lr_disc

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
            loss_encoder=[],
            loss_decoder=[],
            loss_discriminator=[],
            loss_reconstruction=[]
        )

        # An optimizer and schedulers for each of the sub-networks, so we can selectively backprop
        optimizer_encoder = torch.optim.RMSprop(params=model.encoder.parameters(), lr=lr_enc,
                                                alpha=0.9,
                                                eps=1e-8, weight_decay=training_config.weight_decay, momentum=0,
                                                centered=False)
        lr_encoder = ExponentialLR(optimizer_encoder, gamma=args.decay_lr)

        # optimizer_decoder = torch.optim.RMSprop(params=model.decoder.parameters(), lr=lr,
        #                                         alpha=0.9,
        #                                         eps=1e-8, weight_decay=training_config.weight_decay, momentum=0,
        #                                         centered=False)
        # lr_decoder = ExponentialLR(optimizer_decoder, gamma=args.decay_lr)

        optimizer_discriminator = torch.optim.RMSprop(params=model.discriminator.parameters(),
                                                      lr=lr_disc,
                                                      alpha=0.9, eps=1e-8, weight_decay=training_config.weight_decay,
                                                      momentum=0, centered=False)
        lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=args.decay_lr)

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
                            batch_size = len(data_batch['image'])
                            # x = Variable(data_batch, requires_grad=False).float().to(device)

                            # Fix decoder weights
                            for param in model.decoder.parameters():
                                param.requires_grad = False

                            x_gt, x_tilde, disc_class, disc_layer, mus, log_variances = model(data_batch)
                            # x_gt = reconstruction by teacher (vis enc)
                            # x_tilde = reconstruction from cog
                            # x_tilde, disc_class, disc_layer, mus, log_variances = model(x) # OLD STAGE 1

                            # Split so we can get the different parts
                            # hid_ refers to hidden discriminator layer
                            # dis_ refers to final sigmoid output from discriminator
                            # real = vis reconstruction (teacher)
                            # pred = cog reconstruction
                            hid_dis_real = disc_layer[:batch_size]
                            hid_dis_pred = disc_layer[batch_size:-batch_size]
                            hid_dis_sampled = disc_layer[-batch_size:]

                            # disc_class = fin_dis_
                            fin_dis_real = disc_class[:batch_size]
                            fin_dis_pred = disc_class[batch_size:-batch_size]
                            fin_dis_sampled = disc_class[-batch_size:]

                            # VAE/GAN loss
                            nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled = \
                                VaeGanCognitive.loss(x_gt, x_tilde, hid_dis_real, hid_dis_pred, fin_dis_real,
                                                     fin_dis_pred, fin_dis_sampled, mus, log_variances)

                            loss_encoder = torch.sum(kl) + torch.sum(mse)
                            loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(
                                bce_dis_sampled)
                            loss_decoder = torch.sum(training_config.lambda_mse * mse) - (1.0 - training_config.lambda_mse) * loss_discriminator

                            logging.info('Encoder loss: {} \nDecoder loss: {} \nDiscriminator loss: {}'.format(loss_encoder,
                                                                                                               loss_decoder,
                                                                                                               loss_discriminator))

                            # Register mean values for logging
                            loss_encoder_mean = loss_encoder.data.cpu().numpy() / batch_size
                            loss_discriminator_mean = loss_discriminator.data.cpu().numpy() / batch_size
                            loss_decoder_mean = loss_decoder.data.cpu().numpy() / batch_size
                            loss_nle_mean = torch.sum(nle).data.cpu().numpy() / batch_size

                            # Selectively disable the decoder of the discriminator if they are unbalanced
                            train_dis = True
                            train_dec = False

                            # Initially try training without equilibrium
                            equilibrium_game = args.equilibrium_game

                            if equilibrium_game == 'y':
                                if torch.mean(bce_dis_original).item() < equilibrium - margin or torch.mean(
                                        bce_dis_predicted).item() < equilibrium - margin:
                                    train_dis = False
                                if torch.mean(bce_dis_original).item() > equilibrium + margin or torch.mean(
                                        bce_dis_predicted).item() > equilibrium + margin:
                                    train_dec = False
                                if train_dec is False and train_dis is False:
                                    train_dis = True
                                    train_dec = True

                            if args.backprop_method == 'trad':
                                # BACKPROP
                                loss_encoder.backward(retain_graph=True, inputs=list(model.encoder.parameters()))
                                optimizer_encoder.step()

                                # if train_dec:
                                #     model.decoder.zero_grad()
                                #     loss_decoder.backward(retain_graph=True, inputs=list(model.decoder.parameters()))
                                #     optimizer_decoder.step()

                                if train_dis:
                                    model.discriminator.zero_grad()
                                    loss_discriminator.backward(inputs=list(model.discriminator.parameters()))
                                    optimizer_discriminator.step()

                                model.zero_grad()

                            if args.backprop_method == 'clip':
                                # BACKPROP
                                loss_encoder.backward(retain_graph=True, inputs=list(model.encoder.parameters()))
                                [p.grad.data.clamp_(-1, 1) for p in model.encoder.parameters()]
                                optimizer_encoder.step()

                                # if train_dec:
                                #     model.decoder.zero_grad()
                                #     loss_decoder.backward(retain_graph=True, inputs=list(model.decoder.parameters()))
                                #     optimizer_decoder.step()

                                if train_dis:
                                    model.discriminator.zero_grad()
                                    loss_discriminator.backward(inputs=list(model.discriminator.parameters()))
                                    [p.grad.data.clamp_(-1, 1) for p in model.discriminator.parameters()]
                                    optimizer_discriminator.step()

                                model.zero_grad()

                            logging.info(
                                f'Epoch  {idx_epoch} {batch_idx + 1:3.0f} / {100 * (batch_idx + 1) / len(dataloader_train):2.3f}%, '
                                f'---- encoder loss: {loss_encoder_mean:.5f} ---- | '
                                f'---- decoder loss: {loss_decoder_mean:.5f} ---- | '
                                f'---- discriminator loss: {loss_discriminator_mean:.5f} ---- | '
                                f'---- network status (dec, dis): {train_dec}, {train_dis}')

                            writer.add_scalar('loss_reconstruction_batch', loss_nle_mean, step_index)
                            writer_encoder.add_scalar('loss_encoder_batch', loss_encoder_mean, step_index)
                            writer_decoder.add_scalar('loss_decoder_discriminator_batch', loss_decoder_mean, step_index)
                            writer_discriminator.add_scalar('loss_decoder_discriminator_batch', loss_discriminator_mean, step_index)

                            step_index += 1
                        else:
                            break

                    # EPOCH END
                    lr_encoder.step()
                    # lr_decoder.step()
                    lr_discriminator.step()
                    margin *= training_config.decay_margin
                    equilibrium *= training_config.decay_equilibrium

                    if margin > equilibrium:
                        equilibrium = margin
                    lambda_mse *= decay_mse
                    if lambda_mse > 1:
                        lambda_mse = 1

                    writer.add_scalar('loss_reconstruction', loss_nle_mean, idx_epoch)
                    writer_encoder.add_scalar('loss_encoder', loss_encoder_mean, idx_epoch)
                    writer_decoder.add_scalar('loss_decoder_discriminator', loss_decoder_mean, idx_epoch)
                    writer_discriminator.add_scalar('loss_decoder_discriminator', loss_discriminator_mean, idx_epoch)

                    if not idx_epoch % 2:
                        # Save train examples
                        images_dir = os.path.join(SAVE_PATH, 'images', 'train')
                        if not os.path.exists(images_dir):
                            os.makedirs(images_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Training Ground Truth at Epoch {}'.format(idx_epoch))
                        ax.imshow(make_grid(data_batch['image'][: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                        gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                        plt.savefig(gt_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Training Vis Enc Reconstruction (Real) at Epoch {}'.format(idx_epoch))
                        ax.imshow(
                            make_grid(x_gt[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                        gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_vis_output_' + 'grid')
                        plt.savefig(gt_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Training Cog Enc Reconstruction at Epoch {}'.format(idx_epoch))
                        ax.imshow(make_grid(x_tilde[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                        output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_cog_output_' + 'grid')
                        plt.savefig(output_dir)

                    logging.info('Evaluation')

                    for batch_idx, data_batch in enumerate(dataloader_valid):
                        model.eval()

                        with no_grad():

                            data_target = Variable(data_batch['image'], requires_grad=False).float().to(device)
                            out, vis_out = model(data_batch)

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
                                        result_metrics_train[key] = metric(x_tilde, x_gt).mean()
                                    else:
                                        result_metrics_train[key] = metric(x_tilde, x_gt)

                            out = out.data.cpu()

                        # Save validation examples
                        images_dir = os.path.join(SAVE_PATH, 'images', 'valid')
                        if not os.path.exists(images_dir):
                            os.makedirs(images_dir)
                            os.makedirs(os.path.join(images_dir, 'random'))

                        out = out.data.cpu()
                        vis_out = vis_out.data.cpu()

                        if shuf:
                            fig, ax = plt.subplots(figsize=(10, 10))
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_title('Validation Ground Truth')
                            ax.imshow(
                                make_grid(data_target[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                            gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                            plt.savefig(gt_dir)

                            fig, ax = plt.subplots(figsize=(10, 10))
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_title('Validation Vis Enc Reconstruction at Epoch {}'.format(idx_epoch))
                            ax.imshow(make_grid(vis_out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                            output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_vis_output_' + 'grid')
                            plt.savefig(output_dir)

                        else:  # valid_shuffle is false, so same images are shown
                            if idx_epoch == 0:
                                fig, ax = plt.subplots(figsize=(10, 10))
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.set_title('Validation Ground Truth')
                                ax.imshow(make_grid(data_target[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                                gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                                plt.savefig(gt_dir)

                                fig, ax = plt.subplots(figsize=(10, 10))
                                ax.set_xticks([])
                                ax.set_yticks([])
                                ax.set_title('Validation Vis Enc Reconstruction at Epoch {}'.format(idx_epoch))
                                ax.imshow(
                                    make_grid(vis_out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                                output_dir = os.path.join(images_dir,
                                                          'epoch_' + str(idx_epoch) + '_vis_output_' + 'grid')
                                plt.savefig(output_dir)

                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Validation Cog Enc Reconstruction at Epoch {}'.format(idx_epoch))
                        ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                        output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_cog_output_' + 'grid')
                        plt.savefig(output_dir)

                        out = (out + 1) / 2
                        out = make_grid(out, nrow=8)
                        writer.add_image("reconstructed", out, step_index)

                        # out = model(None, 100)
                        # out = out.data.cpu()
                        # out = (out + 1) / 2
                        # out = make_grid(out, nrow=8)
                        # writer.add_image("generated", out, step_index)

                        out = data_target.data.cpu()
                        out = (out + 1) / 2
                        out = make_grid(out, nrow=8)
                        writer.add_image("original", out, step_index)

                        if metrics_valid is not None:
                            for key, values in result_metrics_valid.items():
                                result_metrics_valid[key] = torch.mean(values)
                            # Logging metrics
                            if writer is not None:
                                for key, values in result_metrics_valid.items():
                                    writer.add_scalar(key, values, stp + idx_epoch)

                        if metrics_train is not None:
                            for key, values in result_metrics_train.items():
                                result_metrics_train[key] = torch.mean(values)
                            # Logging metrics
                            if writer is not None:
                                for key, values in result_metrics_train.items():
                                    writer.add_scalar(key, values, stp + idx_epoch)

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
                    results['loss_encoder'].append(loss_encoder_mean)
                    results['loss_decoder'].append(loss_decoder_mean)
                    results['loss_discriminator'].append(loss_discriminator_mean)
                    results['loss_reconstruction'].append(loss_nle_mean)

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
                    plt.title("Generator and Discriminator Loss During Training")
                    plt.plot(results['loss_decoder'], label="G")
                    plt.plot(results['loss_discriminator'], label="D")
                    plt.xlabel("iterations")
                    plt.ylabel("Loss")
                    plt.legend()
                    plots_dir = os.path.join(SAVE_PATH, 'plots')
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    plot_dir = os.path.join(plots_dir, 'GD_loss')
                    plt.savefig(plot_dir)

                    plt.figure(figsize=(10, 5))
                    plt.title("Encoder and Reconstruction Loss During Training")
                    plt.plot(results['loss_encoder'], label="E")
                    plt.plot(results['loss_reconstruction'], label="R")
                    plt.xlabel("iterations")
                    plt.ylabel("Loss")
                    plt.legend()
                    plots_dir = os.path.join(SAVE_PATH, 'plots')
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    plot_dir = os.path.join(plots_dir, 'ER_loss')
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

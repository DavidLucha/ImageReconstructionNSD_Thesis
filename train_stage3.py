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
    StructuralSimilarity, objective_assessment, parse_args, NLLNormal, FmriDataloader, potentiation

if __name__ == "__main__":
    try:
        numpy.random.seed(753)
        torch.manual_seed(753)
        torch.cuda.manual_seed(753)

        logging.info('set up random seeds')

        torch.autograd.set_detect_anomaly(True)
        timestep = time.strftime("%Y%m%d-%H%M%S")

        stage = 3

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
            parser.add_argument('--epochs', default=training_config.n_epochs_s3, help='number of epochs', type=int)
            parser.add_argument('--num_workers', '-nw', default=training_config.num_workers,
                                help='number of workers for dataloader', type=int)
            parser.add_argument('--loss_method', default='Maria',
                                help='defines loss calculations. Maria, David, Orig.', type=str)
            # We don't currently have a change for optim_method in stage 2/3
            parser.add_argument('--optim_method', default='RMS',
                                help='defines method for optimizer. Options: RMS or Adam.', type=str)
            parser.add_argument('--lr', default=training_config.learning_rate, type=float)
            parser.add_argument('--decay_lr', default=training_config.decay_lr,
                                help='.98 in Maria, .75 in original VAE/GAN', type=float)

            # Pretrained/checkpoint network components
            parser.add_argument('--network_checkpoint', default=None, help='loads checkpoint in the format '
                                                                           'vaegan_20220613-014326', type=str)
            parser.add_argument('--checkpoint_epoch', default=400, help='epoch of checkpoint network', type=int)
            parser.add_argument('--stage_2_trained', default=training_config.stage_2_trained,
                                help='pretrained network from stage 2', type=str)
            parser.add_argument('--load_epoch', '-pretrain_epoch', default=400,
                                help='epoch of the pretrained model from stage 2', type=int)
            parser.add_argument('--dataset', default='GOD', help='GOD, NSD', type=str)
            # Only need vox_res arg from stage 2 and 3
            parser.add_argument('--vox_res', default='1.8mm', help='1.8mm, 3mm', type=str)
            # Probably only needed stage 2/3 (though do we want to change stage 1 - depends on whether we do the full Ren...
            # Thing where we do a smaller set for stage 1. But I think I might do a Maria and just have stage 1 with no...
            # pretraining phase...
            parser.add_argument('--set_size', default='max', help='max:max available (including repeats), '
                                                                  'single_pres: max available single presentations, '
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
                                               'Subj_0{}_{}_NSD_single_pres_train.pickle'.format(args.subject,
                                                                                                 args.set_size))

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
        stage_num = 'stage_3'
        SAVE_PATH = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, SUBJECT_PATH, stage_num,
                                 'vaegan_{}'.format(args.run_name))
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        SAVE_SUB_PATH = os.path.join(SAVE_PATH, 'stage_3_vaegan_{}.pth'.format(args.run_name))
        if not os.path.exists(SAVE_SUB_PATH):
            os.makedirs(SAVE_SUB_PATH)

        LOG_PATH = os.path.join(SAVE_PATH, training_config.LOGS_PATH)
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)

        """
        LOGGING SETUP
        """
        # Info logging
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
        logger = logging.getLogger()
        file_handler = logging.FileHandler(os.path.join(LOG_PATH, 'log.txt'))
        handler_formatting = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(handler_formatting)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

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

        # Save arguments
        with open(os.path.join(SAVE_PATH, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        root_path = os.path.join(args.data_root, args.dataset + '/')

        """
        DATASET LOADING
        """
        # image_crop = training_config.image_crop

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

        dataloader_train = DataLoader(training_data, batch_size=args.batch_size,  # collate_fn=PadCollate(dim=0),
                                      shuffle=True, num_workers=args.num_workers)
        dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size,  # collate_fn=PadCollate(dim=0),
                                      shuffle=False, num_workers=args.num_workers)

        NUM_VOXELS = len(train_data[0]['fmri'])

        # This writer function is for torch.tensorboard
        writer = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name)
        writer_encoder = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/encoder')
        writer_decoder = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/decoder')
        writer_discriminator = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/discriminator')

        logging.info(f'Validation data length: {NUM_VOXELS}')
        logging.info(f'Validation data length: {len(train_data)}')
        logging.info(f'Validation data length: {len(valid_data)}')

        """
        What model components do we need?
            Generator from Stage 1/Stage 2 works because it's the same
            Cognitive encoder from Stage 2
            Discriminator from Stage 2
            
        We don't need teacher model. Though Maria, loads it, but doesn't load any weights into it.
        """

        # Initialize network and load Stage 2 weights
        # decoder = Decoder(z_size=training_config.latent_dim, size=encoder.size).to(device)
        # teacher_model = VaeGan(device=device, z_size=training_config.latent_dim).to(device)
        # teacher_model.load_state_dict(torch.load(model_dir, map_location=device))
        # discriminator = teacher_model.discriminator
        # teacher_model.discriminator.train() # Not discriminator.train()?
        # teacher_model = None  # Don't need for stage 3 (use images as real)

        # If error with loading model then:
        teacher_model = VaeGan(device=device, z_size=training_config.latent_dim).to(device)
        # David: they don't load model params here (teacher_model specifically)
        for param in teacher_model.parameters():
            param.requires_grad = False

        vis_encoder = Encoder(z_size=training_config.latent_dim).to(device)  # Only used to grab size for decoder

        # Define model for stage II
        cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=training_config.latent_dim).to(device)
        decoder = Decoder(z_size=training_config.latent_dim, size=vis_encoder.size).to(device)  # TODO: Check size is okay
        # Though it should be because every one instance uses decoder.size=encoder.size
        # Given that we use the same decoder, this should be the same
        discriminator = Discriminator().to(device)  # Default: channels=3, recon_level=3

        # Initialize VaeGanCognitive
        model = VaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=decoder,
                                discriminator=discriminator, teacher_net=teacher_model,
                                z_size=training_config.latent_dim, stage=3).to(device)

        # Variables for equilibrium to improve GAN stability
        margin = training_config.margin
        equilibrium = training_config.equilibrium
        lambda_mse = training_config.lambda_mse
        decay_mse = training_config.decay_mse
        lr = args.lr

        # Load Stage 2 network weights
        model_dir = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, SUBJECT_PATH,
                                 'stage_2', args.stage_2_trained,
                                 'stage_2_' + args.stage_2_trained + '_{}.pth'.format(args.load_epoch))
        logging.info('Loaded network is:', model_dir)

        # Load weights from stage 2
        logging.info('Loading model from Stage 2')
        model.load_state_dict(torch.load(model_dir, map_location=device))
        # I wonder if this gives us an issue, because we don't have the teacher model but
        # Stage 2 would include a teacher_network. Might need it after all.
        model.discriminator.train()
        model.decoder.train()

        # Fix Encoder (as per Ren's stages)
        for param in model.encoder.parameters():
            param.requires_grad = False

        # Loading Checkpoint | If you want to continue training for existing checkpoint
        # Set checkpoint path
        if args.network_checkpoint is not None:
            net_checkpoint_path = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, SUBJECT_PATH,
                                               'stage_3', args.network_checkpoint,
                                               'stage_3_' + args.network_checkpoint + '.pth')
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
        # optimizer_encoder = torch.optim.RMSprop(params=model.encoder.parameters(), lr=lr,
        #                                         alpha=0.9,
        #                                         eps=1e-8, weight_decay=training_config.weight_decay, momentum=0,
        #                                         centered=False)
        # lr_encoder = ExponentialLR(optimizer_encoder, gamma=args.decay_lr)

        optimizer_decoder = torch.optim.RMSprop(params=model.decoder.parameters(), lr=lr,
                                                alpha=0.9, eps=1e-8, weight_decay=training_config.weight_decay,
                                                momentum=0, centered=False)
        lr_decoder = ExponentialLR(optimizer_decoder, gamma=args.decay_lr)

        optimizer_discriminator = torch.optim.RMSprop(params=model.discriminator.parameters(),
                                                      lr=lr,
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

        for idx_epoch in range(args.epochs):

            try:

                # For each batch
                for batch_idx, data_batch in enumerate(dataloader_train):

                    model.train()
                    batch_size = len(data_batch['image'])
                    # x = Variable(data_batch, requires_grad=False).float().to(device)

                    # Fix encoder weights
                    for param in teacher_model.parameters():
                        param.requires_grad = False
                    for param in model.encoder.parameters():
                        param.requires_grad = False
                    for param in model.decoder.parameters():
                        param.requires_grad = True
                    for param in model.discriminator.parameters():
                        param.requires_grad = True

                    x_gt, x_tilde, disc_class, disc_layer, mus, log_variances = model(data_batch)

                    # Split so we can get the different parts
                    hid_dis_real = disc_layer[:batch_size]
                    hid_dis_pred = disc_layer[batch_size:-batch_size]
                    hid_dis_sampled = disc_layer[-batch_size:]

                    # disc_class = fin_dis_
                    fin_dis_real = disc_class[:batch_size]
                    fin_dis_pred = disc_class[batch_size:-batch_size]
                    fin_dis_sampled = disc_class[-batch_size:]

                    # VAE/GAN Loss
                    loss_method = args.loss_method  # 'Maria', 'Ren'

                    # VAE/GAN loss
                    if loss_method == 'Maria':
                        nle, kl, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled = \
                            VaeGanCognitive.loss(x_gt, x_tilde, hid_dis_real, hid_dis_pred, fin_dis_real,
                                                 fin_dis_pred, fin_dis_sampled, mus, log_variances)

                        loss_encoder = torch.sum(kl) + torch.sum(mse)
                        loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(
                            bce_dis_sampled)
                        loss_decoder = torch.sum(training_config.lambda_mse * mse) - (1.0 - training_config.lambda_mse) * loss_discriminator

                        # Register mean values for logging
                        loss_encoder_mean = loss_encoder.data.cpu().numpy() / batch_size
                        loss_discriminator_mean = loss_discriminator.data.cpu().numpy() / batch_size
                        loss_decoder_mean = loss_decoder.data.cpu().numpy() / batch_size
                        loss_nle_mean = torch.sum(nle).data.cpu().numpy() / batch_size

                    if loss_method == 'Ren':
                        # Using Ren's Loss Function
                        # TODO: Calculate loss functions here. Or do in function.
                        # TODO: ADD EXTRA PARAMS (Only for COG)
                        """nle, loss_encoder, loss_decoder, loss_discriminator = VaeGanCognitive.ren_loss(x_gt, x_tilde, mus,
                                                                                              log_variances, hid_dis_real,
                                                                                              hid_dis_pred, fin_dis_real,
                                                                                              fin_dis_pred, stage=stage,
                                                                                              device=device)"""
                        # Register mean values for logging
                        loss_encoder_mean = torch.mean(loss_encoder).data.cpu().numpy()
                        loss_discriminator_mean = loss_discriminator.data.cpu().numpy()  # / batch_size
                        loss_decoder_mean = loss_decoder.item()  # NOT USED, JUST FOR REF

                        loss_encoder_mean_old = loss_encoder.data.cpu().numpy()  # / batch_size
                        loss_discriminator_mean_old = loss_discriminator.data.cpu().numpy()  # / batch_size
                        loss_decoder_mean_old = loss_decoder.data.cpu().numpy()  # NOT USED, JUST FOR REF
                        loss_nle_mean = torch.sum(nle).data.cpu().numpy() / batch_size

                    # Selectively disable the decoder of the discriminator if they are unbalanced
                    train_dis = True
                    train_dec = True

                    # Initially try training without equilibrium
                    train_equilibrium = False # Leave off probably
                    if train_equilibrium:
                        if torch.mean(bce_dis_original).item() < equilibrium - margin or torch.mean(
                                bce_dis_predicted).item() < equilibrium - margin:
                            train_dis = False
                        if torch.mean(bce_dis_original).item() > equilibrium + margin or torch.mean(
                                bce_dis_predicted).item() > equilibrium + margin:
                            train_dec = False
                        if train_dec is False and train_dis is False:
                            train_dis = True
                            train_dec = True

                    # BACKPROP
                    model.zero_grad() # TODO: Don't I need this for Stage 1 and 2? Currently missing. Test.
                    # loss_encoder.backward(retain_graph=True, inputs=list(model.encoder.parameters()))
                    # optimizer_encoder.step()

                    if train_dec:
                        model.decoder.zero_grad()
                        loss_decoder.backward(retain_graph=True, inputs=list(model.decoder.parameters()))
                        optimizer_decoder.step()

                    if train_dis:
                        model.discriminator.zero_grad()
                        loss_discriminator.backward(inputs=list(model.discriminator.parameters()))
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

                # EPOCH END
                # lr_encoder.step()
                lr_decoder.step()
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
                    ax.set_title('Training Reconstruction at Epoch {}'.format(idx_epoch))
                    ax.imshow(make_grid(x_tilde[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                    output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                    plt.savefig(output_dir)

                logging.info('Evaluation')

                for batch_idx, data_batch in enumerate(dataloader_valid):
                    model.eval()

                    with no_grad():

                        data_target = Variable(data_batch['image'], requires_grad=False).float().to(device)
                        out = model(data_batch)

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
                    ax.set_title('Validation Reconstruction at Epoch {}'.format(idx_epoch))
                    ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                    output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
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

                    break

                if not idx_epoch % 20 or idx_epoch == epochs_n-1:
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
        exit(0)
    except Exception:
        logger.error("Fatal error", exc_info=True)
        sys.exit(1)

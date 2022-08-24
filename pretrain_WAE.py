import os
import random
import time
import numpy
import torch
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import json
import random
import sys

import torchvision
from torch import nn
from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, StepLR

import training_config
from model_2 import WaeGan
from utils_2 import ImageNetDataloader, GreyToColor, evaluate, PearsonCorrelation, \
    StructuralSimilarity, objective_assessment, parse_args, NLLNormal, potentiation, FmriDataloader

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
            parser.add_argument('--lr_enc', default=0.0001, type=float)
            parser.add_argument('--lr_dec', default=0.0001, type=float)
            parser.add_argument('--lr_disc', default=0.0001 * 0.5, type=float)
            parser.add_argument('--decay_lr', default=0.5,
                                help='.98 in Maria, .75 in original VAE/GAN', type=float)
            parser.add_argument('--backprop_method', default='clip', help='trad sets three diff loss functions,'
                                                                          'but clip, clips the gradients to help'
                                                                          'avoid the late spikes in loss', type=str)
            parser.add_argument('--disc_loss', default='Maria',
                                help='determines whether we use Marias loss or the paper based one for disc', type=str)
            parser.add_argument('--WAE_loss', default='Maria',
                                help='determines whether we use Marias loss or the paper based one for WAE', type=str)
            parser.add_argument('--lambda_WAE', default=1, help='sets the multiplier for paper GAN loss', type=int)

            parser.add_argument('--seed', default=277603, help='sets seed, 0 makes a random int', type=int)
            parser.add_argument('--gpus', default=1, help='number of gpus but just testing this', type=int)
            parser.add_argument('--latent_dims', default=128, type=int)
            parser.add_argument('--beta', default=0.5, type=float)
            parser.add_argument('--dataloader', default='image', help='if anything but pickle, it will just'
                                                                      'grab all 72k images across 8 subjs. If pickle,'
                                                                      ' then only the single pres for train and valid',
                                type=str)
            parser.add_argument('--subject', default=0, help='only used if dataloader is pickle', type=int)


            # Pretrained/checkpoint network components
            parser.add_argument('--network_checkpoint', default=None, help='loads checkpoint in the format '
                                                                           'WAE_20220613-014326', type=str)
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
        OUTPUT_PATH = os.path.join(args.data_root, 'output/')

        TRAIN_DATA_PATH = os.path.join(args.data_root, training_config.god_pretrain_imgs)
        VALID_DATA_PATH = os.path.join(args.data_root, 'NSD/images/valid/')

        if args.dataset == 'NSD':
            TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD/images/train/')

        # Create directory for results
        stage_num = 'pretrain'
        stage = 1

        # TODO: If trying subject specific pretraining - then change save paths.
        SAVE_PATH = os.path.join(OUTPUT_PATH, args.dataset, stage_num, args.run_name)
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        SAVE_SUB_PATH = os.path.join(SAVE_PATH, 'pretrained_WAE_{}.pth'.format(args.run_name))
        if not os.path.exists(SAVE_SUB_PATH):
            os.makedirs(SAVE_SUB_PATH)

        LOG_PATH = os.path.join(SAVE_PATH, training_config.LOGS_PATH)
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
            # os.chmod(LOG_PATH, 0o777)

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
        # logger = logging.getLogger()
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
        # import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(device)
        # conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.2 -c pytorch
        logger.info("Used device: %s" % device)
        if device == 'cpu':
            raise Exception()


        """
        DATASET LOADING
        """
        # Load image-only training data
        train_data = TRAIN_DATA_PATH
        valid_data = VALID_DATA_PATH

        if args.dataloader == 'pickle':
            TRAIN_DATA_PATH = os.path.join(args.data_root, 'NSD/3mm/train/single_pres',
                                           'Subj_0{}_NSD_single_pres_train.pickle'.format(args.subject))
            VALID_DATA_PATH = os.path.join(args.data_root, 'NSD/3mm/valid/single_pres',
                                           'Subj_0{}_NSD_single_pres_valid.pickle'.format(args.subject))

            train_data = TRAIN_DATA_PATH
            valid_data = VALID_DATA_PATH

            root_path = os.path.join(args.data_root, args.dataset + '/')
            # Used to test whether loading on an individuals subject would be better
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
                                             transform=transforms.Compose(
                                                 [transforms.Resize((training_config.image_size,
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

        else:
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

        # This writer function is for torch.tensorboard - might be worth
        writer = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name)
        writer_encoder = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/encoder')
        writer_decoder = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/decoder')
        writer_discriminator = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/discriminator')

        model = WaeGan(device=device, z_size=args.latent_dims).to(device)

        # Variables for equilibrium to improve GAN stability
        margin = training_config.margin
        equilibrium = training_config.equilibrium
        lambda_mse = training_config.lambda_mse
        decay_mse = training_config.decay_mse
        # lr = args.lr

        # Loading Checkpoint | If you want to continue previous training
        # Set checkpoint path
        if args.network_checkpoint is not None:
            net_checkpoint_path = os.path.join(OUTPUT_PATH, args.dataset, 'pretrain', args.network_checkpoint,
                                       'pretrained_WAE_' + args.network_checkpoint + '.pth')
            print(net_checkpoint_path)

        # Load and show results
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
            logging.info('Initialize')
            stp = 1

        results = dict(
            epochs=[],
            loss_reconstruction=[],
            loss_penalty=[],
            loss_discriminator=[]
        )

        beta = args.beta
        lr_enc = args.lr_enc  # 0.0001 in Maria | Paper = 3e-4 (0.0003) | 0.003 probably though
        lr_dec = args.lr_dec  # 0.0001 in Maria | Paper = 3e-4 (0.0003)
        lr_disc = args.lr_disc  # 0.5 * 0.0001 in Maria | Paper = 1e-3 (0.001) | might be good

        # Optimizers
        optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lr_enc, betas=(beta, 0.999))
        optimizer_decoder = torch.optim.Adam(model.decoder.parameters(), lr=lr_dec, betas=(beta, 0.999))
        optimizer_discriminator = torch.optim.Adam(model.discriminator.parameters(), lr=lr_disc, betas=(beta, 0.999))

        lr_encoder = StepLR(optimizer_encoder, step_size=30, gamma=0.5)
        lr_decoder = StepLR(optimizer_decoder, step_size=30, gamma=0.5)
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

        batch_number = len(dataloader_train)
        step_index = 0
        epochs_n = args.epochs
        epoch_times = []

        for idx_epoch in range(epochs_n):

            try:
                time_1 = time.time()

                # For each batch
                for batch_idx, data_batch in enumerate(dataloader_train):
                    model.train()

                    batch_size = len(data_batch)
                    model.encoder.zero_grad()
                    model.decoder.zero_grad()
                    model.discriminator.zero_grad()

                    x = Variable(data_batch, requires_grad=False).float().to(device)

                    # ----------Train discriminator-------------

                    frozen_params(model.decoder)
                    frozen_params(model.encoder)
                    free_params(model.discriminator)

                    z_enc, var = model.encoder(x)
                    z_samp = Variable(torch.randn_like(z_enc) * 0.5).to(device)

                    # disc output for encoded latent | Qz
                    logits_enc = model.discriminator(z_enc)
                    # disc output for sampled (noise) latent | Pz
                    logits_samp = model.discriminator(z_samp)

                    if args.disc_loss == "Maria":
                        sig_enc = torch.sigmoid(logits_enc)
                        sig_samp = torch.sigmoid(logits_samp)

                        # this is taking the BCE (1 - sampled) of the sampled CORRECT
                        loss_discriminator_fake = - 10 * torch.sum(torch.log(sig_samp + 1e-3))
                        # this takes the BCE (0 - encoded) of the encoded latent CORRECT
                        loss_discriminator_real = - 10 * torch.sum(torch.log(1 - sig_enc + 1e-3))

                        loss_discriminator = loss_discriminator_real + loss_discriminator_real

                        loss_discriminator_fake.backward(retain_graph=True,
                                                         inputs=list(model.discriminator.parameters()))
                        loss_discriminator_real.backward(retain_graph=True,
                                                         inputs=list(model.discriminator.parameters()))
                        mean_mult = batch_size
                    elif args.disc_loss == "Both":
                        # Using Maria's but with modern Pytorch BCE loss + addition of loss terms before back pass
                        bce_loss = nn.BCEWithLogitsLoss(reduction='none')

                        # set up labels
                        labels_real = Variable(torch.ones_like(logits_samp)).to(device)
                        labels_fake = Variable(torch.zeros_like(logits_enc)).to(device)

                        # Qz is distribution of encoded latent space
                        loss_Qz = 10 * torch.sum(bce_loss(logits_enc, labels_fake))
                        # Pz is distribution of prior (sampled)
                        loss_Pz = 10 * torch.sum(bce_loss(logits_samp, labels_real))

                        loss_discriminator = args.lambda_WAE * (loss_Qz + loss_Pz)
                        loss_discriminator.backward(retain_graph=True, inputs=list(model.discriminator.parameters()))
                        mean_mult = batch_size * 10
                    else:
                        # set up labels
                        labels_real = Variable(torch.ones_like(logits_samp)).to(device)
                        labels_fake = Variable(torch.zeros_like(logits_enc)).to(device)

                        # Qz is distribution of encoded latent space
                        loss_Qz = bce_loss(logits_enc, labels_fake)
                        # Pz is distribution of prior (sampled)
                        loss_Pz = bce_loss(logits_samp, labels_real)

                        loss_discriminator = args.lambda_WAE * (loss_Qz + loss_Pz)
                        loss_discriminator.backward(retain_graph=True, inputs=list(model.discriminator.parameters()))
                        mean_mult = 1

                    # [p.grad.data.clamp_(-1, 1) for p in model.discriminator.parameters()]
                    optimizer_discriminator.step()

                    # ----------Train generator----------------
                    model.encoder.zero_grad()
                    model.decoder.zero_grad()

                    free_params(model.encoder)
                    free_params(model.decoder)
                    frozen_params(model.discriminator)

                    z_enc, var = model.encoder(x)
                    x_recon = model.decoder(z_enc)
                    logits_enc = model.discriminator(z_enc)

                    encdec_params = list(model.encoder.parameters()) + list(model.decoder.parameters())

                    if args.WAE_loss == 'Maria':
                        sig_enc = torch.sigmoid(logits_enc)
                        loss_reconstruction = torch.sum(torch.sum(0.5 * (x_recon - x) ** 2, 1))
                        # loss_reconstruction = mse_loss(x_recon, x)
                        # non-saturating loss | taking the BCE (1, real) CORRECT
                        loss_penalty = - 10 * torch.sum(torch.log(sig_enc + 1e-3))

                        loss_reconstruction.backward(retain_graph=True, inputs=encdec_params)
                        loss_penalty.backward(inputs=encdec_params)
                        mean_mult = batch_size
                    elif args.WAE_loss == "Both":
                        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
                        # Like Maria's but with MSE and BCE from pytorch
                        # label for non-saturating loss
                        labels_saturated = Variable(torch.ones_like(logits_enc)).to(device)
                        loss_reconstruction = torch.sum(torch.sum(0.5 * (x_recon - x) ** 2, 1))
                        # loss_reconstruction = mse_loss(x_recon, x)
                        loss_penalty = 10 * torch.sum(bce_loss(logits_enc, labels_saturated))
                        loss_WAE = loss_reconstruction + loss_penalty * args.lambda_WAE
                        loss_WAE.backward(inputs=encdec_params)
                        mean_mult = batch_size * 10
                    else:
                        # Adapted from original WAE paper code
                        # label for non-saturating loss
                        labels_saturated = Variable(torch.ones_like(logits_enc)).to(device)
                        loss_reconstruction = torch.mean(torch.sum(mse_loss(x_recon, x), [1, 2, 3])) * 0.05
                        # loss_reconstruction = mse_loss(x_recon, x)
                        loss_penalty = bce_loss(logits_enc, labels_saturated)
                        loss_WAE = loss_reconstruction + loss_penalty * args.lambda_WAE
                        loss_WAE.backward(inputs=encdec_params)
                        mean_mult = 1

                    # [p.grad.data.clamp_(-1, 1) for p in model.encoder.parameters()]
                    optimizer_encoder.step()
                    optimizer_decoder.step()

                    # register mean values of the losses for logging
                    loss_reconstruction_mean = loss_reconstruction.data.cpu().numpy() / mean_mult
                    loss_penalty_mean = loss_penalty.data.cpu().numpy() / mean_mult
                    loss_discriminator_mean = loss_discriminator.data.cpu().numpy() / mean_mult
                    # loss_discriminator_real_mean = loss_discriminator_real.data.cpu().numpy() / batch_size

                    # logging.info(
                    #     f'Epoch  {idx_epoch} {batch_idx + 1:3.0f} / {100 * (batch_idx + 1) / len(dataloader_train):2.3f}%, '
                    #     f'---- recon loss: {loss_reconstruction_mean:.5f} ---- | '
                    #     f'---- penalty loss: {loss_penalty_mean:.5f} ---- | '
                    #     f'---- discrim loss: {loss_discriminator_mean:.5f}')

                    step_index += 1

                # EPOCH END
                lr_encoder.step()
                lr_decoder.step()
                lr_discriminator.step()

                if not idx_epoch % 2:
                    # Save train examples
                    images_dir = os.path.join(SAVE_PATH, 'images', 'train')
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)

                    # Ground truth
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title('Training Ground Truth at Epoch {}'.format(idx_epoch))
                    ax.imshow(make_grid(x[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                    gt_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_ground_truth_' + 'grid')
                    plt.savefig(gt_dir)

                    # Reconstructed
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title('Training Reconstruction at Epoch {}'.format(idx_epoch))
                    ax.imshow(make_grid(x_recon[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                    output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                    plt.savefig(output_dir)

                logging.info('Evaluation')

                for batch_idx, data_batch in enumerate(dataloader_valid):

                    batch_size = len(data_batch)
                    data_batch = Variable(data_batch, requires_grad=False).float().to(device)

                    model.eval()
                    data_in = Variable(data_batch, requires_grad=False).float().to(device)
                    data_target = Variable(data_batch, requires_grad=False).float().to(device)
                    out, _ = model(data_in)

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
                                result_metrics_train[key] = metric(x_recon, x).mean()
                            else:
                                result_metrics_train[key] = metric(x_recon, x)

                    # Save validation examples
                    images_dir = os.path.join(SAVE_PATH, 'images', 'valid')
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)
                        os.makedirs(os.path.join(images_dir, 'random'))

                    out = out.data.cpu()

                    if idx_epoch == 0:
                        fig, ax = plt.subplots(figsize=(10, 10))
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title('Validation Ground Truth')
                        ax.imshow(make_grid(data_in[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
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

                    """out = model(None, 100)
                    out = out.data.cpu()
                    out = (out + 1) / 2
                    out = make_grid(out, nrow=8)
                    writer.add_image("generated", out, step_index)
    
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title('Random Generation at Epoch {}'.format(idx_epoch))
                    ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                    output_dir = os.path.join(images_dir, 'random', 'epoch_' + str(idx_epoch) + '_output_' + 'rand')
                    plt.savefig(output_dir)"""

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

                if not idx_epoch % 30 or idx_epoch == epochs_n-1:
                    torch.save(model.state_dict(), SAVE_SUB_PATH.replace('.pth', '_' + str(idx_epoch) + '.pth'))
                    logging.info('Saving model')

                # Record losses & scores
                results['epochs'].append(idx_epoch + stp)
                results['loss_reconstruction'].append(loss_reconstruction_mean)
                results['loss_penalty'].append(loss_penalty_mean)
                results['loss_discriminator'].append(loss_discriminator_mean)
                # results['loss_discriminator_real'].append(loss_discriminator_real_mean)

                if metrics_valid is not None:
                    for key, value in result_metrics_valid.items():
                        metric_value = value.detach().clone().item()
                        # Changed from: torch.tensor(value, dtype=torch.float64).item()
                        results[key].append(metric_value)

                if metrics_train is not None:
                    for key, value in result_metrics_train.items():
                        metric_value = value.detach().clone().item()
                        results[key].append(metric_value)

                results_to_save = pd.DataFrame(results)
                results_to_save.to_csv(SAVE_SUB_PATH.replace(".pth", "_results.csv"), index=False)

                # Calculates average epoch duration
                time_2 = time.time()
                time_diff = time_2 - time_1  # in seconds
                epoch_times.append(time_diff)
                avg_time = (sum(epoch_times) / len(epoch_times)) / 60
                logging.info('Duration of epoch {} was: {:.2f} minutes.'
                             '\nAverage epoch duration is {:.2f} minutes.'.format(idx_epoch,
                                                                              epoch_times[idx_epoch] / 60, avg_time))

            except KeyboardInterrupt as e:
                 logging.info(e, 'Saving plots')

            finally:

                plt.figure(figsize=(10, 5))
                plt.title("Discriminator Loss During Training")
                plt.plot(results['loss_discriminator'], label="Discrim")
                # plt.plot(results['loss_discriminator_fake'], label="DF")
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
                plt.plot(results['loss_penalty'], label="LP")
                plt.plot(results['loss_reconstruction'], label="LR")
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
        exit(0)
    except Exception:
        logger.error("Fatal error", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

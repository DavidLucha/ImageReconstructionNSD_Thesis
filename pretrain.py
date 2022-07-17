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

import training_config
from model_2 import VaeGan
from utils_2 import ImageNetDataloader, GreyToColor, evaluate, PearsonCorrelation, \
    StructuralSimilarity, objective_assessment, parse_args, NLLNormal, potentiation


if __name__ == "__main__":
    try:
        numpy.random.seed(2010)
        torch.manual_seed(2010)
        torch.cuda.manual_seed(2010)

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

            # Optimizing parameters | also, see lambda and margin in training_config.py
            parser.add_argument('--batch_size', default=training_config.batch_size, help='batch size for dataloader',
                                type=int)
            parser.add_argument('--epochs', default=training_config.n_epochs_pt, help='number of epochs', type=int)
            parser.add_argument('--num_workers', '-nw', default=training_config.num_workers,
                                help='number of workers for dataloader', type=int)
            parser.add_argument('-loss_method', default='maria', help='defines loss calculations', type=str)
            parser.add_argument('--lr', default=training_config.learning_rate_pt, type=float)
            parser.add_argument('--decay_lr', default=training_config.decay_lr,
                                help='.98 in Maria, .75 in original VAE/GAN', type=float)

            # Pretrained/checkpoint network components
            parser.add_argument('--network_checkpoint', default=None, help='loads checkpoint in the format '
                                                                           'vaegan_20220613-014326', type=str)
            parser.add_argument('--checkpoint_epoch', default=90, help='epoch of checkpoint network', type=int)
            parser.add_argument('--pretrained_net', '-pretrain', default=training_config.pretrained_net,
                                help='pretrained network', type=str)
            parser.add_argument('--load_epoch', '-pretrain_epoch', default=training_config.load_epoch,
                                help='epoch of the pretrained model', type=int)
            parser.add_argument('--dataset', default='GOD', help='GOD, NSD', type=str)
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
        OUTPUT_PATH = os.path.join(training_config.data_root, 'output/')

        # Load training data for GOD and NSD, default is NSD
        TRAIN_DATA_PATH = os.path.join(training_config.data_root, training_config.nsd_s1_train_imgs)
        VALID_DATA_PATH = os.path.join(training_config.data_root, training_config.nsd_s1_valid_imgs)

        if args.dataset == 'GOD':
            TRAIN_DATA_PATH = os.path.join(training_config.data_root, training_config.god_pretrain_imgs)
            VALID_DATA_PATH = os.path.join(training_config.data_root, training_config.god_s1_valid_imgs)
            # TODO: Change this. Let's make this a subset of a larger pre-training batch.

        # Create directory for results
        stage_num = 'pretrain'
        SAVE_PATH = os.path.join(OUTPUT_PATH, args.dataset, stage_num, 'vaegan_{}'.format(args.run_name))
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
        # Info logging
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
        logger = logging.getLogger()
        file_handler = logging.FileHandler(os.path.join(LOG_PATH, 'log.txt'))
        handler_formatting = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(handler_formatting)
        # logger = logging.getLogger()
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        # Check available gpu
        # import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(device)
        # conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.2 -c pytorch
        logger.info("Used device: %s" % device)
        if device == 'cpu':
            raise Exception()

        logging.info('set up random seeds')

        torch.manual_seed(2010)

        # Save arguments
        # CURRENTLY NOT WORKING
        with open(os.path.join(SAVE_PATH, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)


        """
        DATASET LOADING
        """
        # Load image-only training data
        train_data = TRAIN_DATA_PATH
        valid_data = VALID_DATA_PATH

        if args.dataset == 'GOD':
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

        # TODO: NSD Dataloader
        elif args.dataset == 'NSD':
            """
            Insert here, if needed.
            """

        else:
            logging.info('Wrong dataset')


        """# Test dataloader and show grid
        print(plt.style.available)
    
        font = {'family': 'Times New Roman',
                'weight': 'bold',
                'size': 14}
    
        def show_and_save(file_name, img):
            npimg = numpy.transpose(img.numpy(), (1, 2, 0))
            fig = plt.figure(dpi=200)
            # f = "./%s.png" % file_name
            fig.suptitle(file_name, fontsize=14, fontweight='bold')
            plt.imshow(npimg)
            # plt.imsave(f, npimg)
    
        real_batch = next(iter(dataloader_train))
    
        fig, ax = plt.subplots(figsize=(10, 10))
        # ax.text(family="Times New Roman", size=12)
        ax.set_xticks([]) # Sets to no ticks
        ax.set_yticks([])
        ax.set_title('Iteration of Real Batch', **font)
        ax.imshow(make_grid(real_batch[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
        fig.show()
    
        # output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
        # plt.savefig(output_dir)
    
        # show_and_save("Test Dataloader", make_grid((real_batch * 0.5 + 0.5).cpu(), 8))
        # plt.show()
    
        raise Exception('Stop after grid show.')"""

        # This writer function is for torch.tensorboard - might be worth
        writer = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name)
        writer_encoder = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/encoder')
        writer_decoder = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/decoder')
        writer_discriminator = SummaryWriter(SAVE_PATH + '/runs_' + args.run_name + '/discriminator')

        model = VaeGan(device=device, z_size=training_config.latent_dim, recon_level=training_config.recon_level).to(device)

        # Variables for equilibrium to improve GAN stability
        margin = training_config.margin
        equilibrium = training_config.equilibrium
        lambda_mse = training_config.lambda_mse
        decay_mse = training_config.decay_mse
        lr = args.lr

        # Loading Checkpoint | If you want to continue previous training
        # Set checkpoint path
        if args.network_checkpoint is not None:
            net_checkpoint_path = os.path.join(OUTPUT_PATH, args.dataset, 'pretrain', args.network_checkpoint,
                                       'pretrained_' + args.network_checkpoint + '.pth')
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

        # print(lr) # TODO: Remove if working

        results = dict(
            epochs=[],
            loss_encoder=[],
            loss_decoder=[],
            loss_discriminator=[],
            loss_reconstruction=[]
        )

        # An optimizer and schedulers for each of the sub-networks, so we can selectively backprop
        # TODO: Change LR to args for stages | test RMS vs Adam
        # TODO: Change LR to args for stages
        optim_method = 'RMS'  # RMS or Adam or SGD (Momentum)
        if optim_method == 'RMS':
            optimizer_encoder = torch.optim.RMSprop(params=model.encoder.parameters(), lr=lr,
                                                    alpha=0.9,
                                                    eps=1e-8, weight_decay=training_config.weight_decay, momentum=0,
                                                    centered=False)
            optimizer_decoder = torch.optim.RMSprop(params=model.decoder.parameters(), lr=lr,
                                                    alpha=0.9,
                                                    eps=1e-8, weight_decay=training_config.weight_decay, momentum=0,
                                                    centered=False)
            optimizer_discriminator = torch.optim.RMSprop(params=model.discriminator.parameters(),
                                                          lr=lr,
                                                          alpha=0.9, eps=1e-8, weight_decay=training_config.weight_decay,
                                                          momentum=0, centered=False)

        if optim_method == 'Adam':
            optimizer_encoder = torch.optim.Adam(params=model.encoder.parameters(), lr=lr,
                                                 eps=1e-8, weight_decay=training_config.weight_decay)
            optimizer_decoder = torch.optim.Adam(params=model.decoder.parameters(), lr=lr,
                                                 eps=1e-8, weight_decay=training_config.weight_decay)
            optimizer_discriminator = torch.optim.Adam(params=model.discriminator.parameters(),
                                                       lr=lr, eps=1e-8,
                                                       weight_decay=training_config.weight_decay)


        # Initialize schedulers for learning rate
        lr_encoder = ExponentialLR(optimizer_encoder, gamma=args.decay_lr)
        lr_decoder = ExponentialLR(optimizer_decoder, gamma=args.decay_lr)
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

        for idx_epoch in range(epochs_n):

            try:

                # For each batch
                for batch_idx, data_batch in enumerate(dataloader_train):
                    model.train()

                    if args.dataset == 'GOD':
                        batch_size = len(data_batch)
                        x = Variable(data_batch, requires_grad=False).float().to(device)

                    # elif args.dataset == 'NSD':
                    #     batch_size = len(data_batch[0])
                    #     x = Variable(data_batch[0], requires_grad=False).float().to(device)
                    # else:
                    #     logging.info('Wrong dataset') #ingo

                    x_tilde, disc_class, disc_layer, mus, log_variances = model(x)

                    # Split so we can get the different parts
                    # disc_layer = hid_dis_
                    hid_dis_real = disc_layer[:batch_size]
                    hid_dis_pred = disc_layer[batch_size:-batch_size]
                    hid_dis_sampled = disc_layer[-batch_size:]

                    # disc_class = fin_dis_
                    fin_dis_real = disc_class[:batch_size]
                    fin_dis_pred = disc_class[batch_size:-batch_size]
                    fin_dis_sampled = disc_class[-batch_size:]

                    # Selectively disable the decoder of the discriminator if they are unbalanced
                    train_dis = True
                    train_dec = True

                    loss_method = args.loss_method # 'Maria', 'Orig', 'Ren'

                    # VAE/GAN loss
                    if loss_method == 'Maria':
                        nle, kl, mse_1, mse_2, bce_dis_original, bce_dis_predicted, bce_dis_sampled, \
                        bce_gen_recon, bce_gen_sampled = VaeGan.loss(x, x_tilde, hid_dis_real,
                                                                     hid_dis_pred, hid_dis_sampled,
                                                                     fin_dis_real, fin_dis_pred,
                                                                     fin_dis_sampled, mus, log_variances)

                        loss_encoder = torch.sum(kl) + torch.sum(mse_1)
                        loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(
                            bce_dis_sampled)
                        loss_decoder = torch.sum(training_config.lambda_mse * mse_1) - (1.0 - training_config.lambda_mse) * loss_discriminator

                        # Register mean values for logging
                        loss_encoder_mean = loss_encoder.data.cpu().numpy() / batch_size
                        loss_discriminator_mean = loss_discriminator.data.cpu().numpy() / batch_size
                        loss_decoder_mean = loss_decoder.data.cpu().numpy() / batch_size
                        loss_nle_mean = torch.sum(nle).data.cpu().numpy() / batch_size

                    if loss_method == 'David':
                        nle, kl, mse_1, mse_2, bce_dis_original, bce_dis_predicted, bce_dis_sampled, \
                        bce_gen_recon, bce_gen_sampled = VaeGan.loss(x, x_tilde, hid_dis_real,
                                                                     hid_dis_pred, hid_dis_sampled,
                                                                     fin_dis_real, fin_dis_pred,
                                                                     fin_dis_sampled, mus, log_variances)

                        loss_encoder = torch.sum(kl) + torch.sum(mse_1)
                        loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(
                            bce_dis_sampled)
                        bce_g_tilde_loss = -torch.log(0.375 - fin_dis_pred + 1e-3)
                        loss_decoder = torch.sum(bce_g_tilde_loss) - torch.sum(training_config.lambda_mse * mse_1)
                        # loss_decoder = torch.sum(training_config.lambda_mse * mse_1) - (1.0 - training_config.lambda_mse) * loss_discriminator

                        # Register mean values for logging
                        loss_encoder_mean = loss_encoder.data.cpu().numpy() / batch_size
                        loss_discriminator_mean = loss_discriminator.data.cpu().numpy() / batch_size
                        loss_decoder_mean = loss_decoder.data.cpu().numpy() / batch_size
                        loss_nle_mean = torch.sum(nle).data.cpu().numpy() / batch_size

                    if loss_method == 'Orig':
                        nle, kl, mse_1, mse_2, bce_dis_original, bce_dis_predicted, bce_dis_sampled, \
                        bce_gen_recon, bce_gen_sampled = VaeGan.loss(x, x_tilde, hid_dis_real,
                                                                     hid_dis_pred, hid_dis_sampled,
                                                                     fin_dis_real, fin_dis_pred,
                                                                     fin_dis_sampled, mus, log_variances)

                        # Loss from torch vaegan loss
                        loss_encoder = torch.sum(mse_1) + torch.sum(mse_2) + torch.sum(kl)
                        loss_discriminator = torch.sum(bce_dis_original) + torch.sum(bce_dis_predicted) + torch.sum(bce_dis_sampled)
                        loss_decoder = torch.sum(bce_gen_sampled) + torch.sum(bce_gen_recon)
                        loss_decoder = torch.sum(lambda_mse / 2 * mse_1) + torch.sum(lambda_mse / 2 * mse_2) + (
                                    1.0 - lambda_mse) * loss_decoder

                        # Register mean values for logging
                        loss_encoder_mean = loss_encoder.data.cpu().numpy() / batch_size
                        loss_discriminator_mean = loss_discriminator.data.cpu().numpy() / batch_size
                        loss_decoder_mean = loss_decoder.data.cpu().numpy() / batch_size
                        loss_nle_mean = torch.sum(nle).data.cpu().numpy() / batch_size

                    if loss_method == 'Ren':
                        # Ren Loss Function
                        kl, feature_loss_pred, dis_fake_pred_loss, dis_real_loss, dec_fake_pred_loss = \
                            VaeGan.ren_loss(x, x_tilde, mus, log_variances, hid_dis_real, hid_dis_pred, fin_dis_real,
                                            fin_dis_pred, hid_dis_sampled, fin_dis_sampled, stage=stage, device=device)

                        print(kl, feature_loss_pred, dis_fake_pred_loss, dis_real_loss)

                        loss_encoder = kl + feature_loss_pred # GOOD | but Ren does some further division
                        loss_discriminator = dis_fake_pred_loss + dis_real_loss  # could add sampled term
                        loss_decoder = (1 - training_config.lambda_mse) * dec_fake_pred_loss - training_config.lambda_mse * feature_loss_pred


                        # Register mean values for logging
                        loss_encoder_mean = torch.mean(loss_encoder).data.cpu().numpy()
                        loss_discriminator_mean = loss_discriminator.data.cpu().numpy()  # / batch_size
                        loss_decoder_mean = loss_decoder.item()  # .cpu().numpy()/ batch_size


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
                    # Backpropagation below ensures same results as if running optimizers in isolation
                    # And works in PyTorch==1.10 (allows for other important functions)
                    loss_encoder.backward(retain_graph=True, inputs=list(model.encoder.parameters()))
                    optimizer_encoder.step()

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
                lr_encoder.step()
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
                    ax.imshow(make_grid(x_tilde[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
                    output_dir = os.path.join(images_dir, 'epoch_' + str(idx_epoch) + '_output_' + 'grid')
                    plt.savefig(output_dir)

                logging.info('Evaluation')

                for batch_idx, data_batch in enumerate(dataloader_valid):

                    if args.dataset == 'GOD':
                        batch_size = len(data_batch)
                        data_batch = Variable(data_batch, requires_grad=False).float().to(device)

                    elif args.dataset == 'NSD':
                        batch_size = len(data_batch[0])
                        data_batch = Variable(data_batch[0], requires_grad=False).float().to(device)
                    else:
                        logging.info('Wrong dataset') #ingo

                    model.eval()
                    data_in = Variable(data_batch, requires_grad=False).float().to(device)
                    data_target = Variable(data_batch, requires_grad=False).float().to(device)
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
                                result_metrics_train[key] = metric(x_tilde, x).mean()
                            else:
                                result_metrics_train[key] = metric(x_tilde, x)

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

                if not idx_epoch % 5 or idx_epoch == epochs_n-1:
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
                        # Changed from: torch.tensor(value, dtype=torch.float64).item()
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
        exit(0)
    except Exception:
        logger.error("Fatal error", exc_info=True)
        sys.exit(1)

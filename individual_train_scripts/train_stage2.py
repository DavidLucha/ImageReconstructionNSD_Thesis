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
    StructuralSimilarity, objective_assessment, parse_args, FmriDataloader, collate_fn

numpy.random.seed(8)
torch.manual_seed(8)
torch.cuda.manual_seed(8)

torch.autograd.set_detect_anomaly(True)
timestep = time.strftime("%Y%m%d-%H%M%S")

stage = 2

if __name__ == "__main__":

    """
    ARGS PARSER
    """
    arguments = False  # Set to False while testing

    if arguments:
        args = parse_args(sys.argv[1:])

    if not arguments:
        import args

    """
    PATHS
    """
    # Get current working directory
    CWD = os.getcwd()
    OUTPUT_PATH = os.path.join(training_config.data_root, 'output/')
    SUBJECT_PATH = 'Subject{}/'.format(str(args.subject_no))

    # Load training data for GOD and NSD, default is NSD
    TRAIN_DATA_PATH = os.path.join(training_config.data_root,
                                   training_config.nsd_train_data.replace("Z", str(args.subject_no)))
    VALID_DATA_PATH = os.path.join(training_config.data_root,
                                   training_config.nsd_valid_data.replace("Z", str(args.subject_no)))

    if args.dataset == 'GOD':
        TRAIN_DATA_PATH = os.path.join(training_config.data_root,
                                       training_config.god_train_data.replace("Z", str(args.subject_no)))
        VALID_DATA_PATH = os.path.join(training_config.data_root,
                                       training_config.god_valid_data.replace("Z", str(args.subject_no)))

    # Create directory for results
    stage_num = 'stage_2'
    SAVE_PATH = os.path.join(OUTPUT_PATH, args.dataset, SUBJECT_PATH, stage_num, 'vaegan_{}'.format(timestep))
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    SAVE_SUB_PATH = os.path.join(SAVE_PATH, 'stage_2_vaegan_{}.pth'.format(timestep))
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
    file_handler = logging.FileHandler(os.path.join(LOG_PATH, 'log.log'))
    handler_formatting = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(handler_formatting)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Check available gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Used device: %s" % device)

    logging.info('set up random seeds')
    torch.manual_seed(2022)

    # Load data which were concatenated for 4 subjects and split into training and validation sets
    with open(TRAIN_DATA_PATH, "rb") as input_file:
        train_data = pickle.load(input_file)
    with open(VALID_DATA_PATH, "rb") as input_file:
        valid_data = pickle.load(input_file)

    # Save arguments
    # CURRENTLY NOT WORKING
    # with open(os.path.join(SAVE_PATH, 'config.txt'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    root_path = os.path.join(training_config.data_root, args.dataset + '/')

    """
    DATASET LOADING
    """
    # TODO: Update dataloaders for cog/vis split
    if args.dataset == 'GOD':
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

        """training_data = FmriDataloader(dataset=train_data, root_path=root_path,
                                           transform=transforms.Compose([
                                                # Rescale(output_size=training_config.image_size),
                                                CenterCrop(output_size=training_config.image_size),
                                                Rescale(output_size=training_config.image_size),
                                                RandomShift(),
                                                SampleToTensor(), # Only one transforming both
                                                Normalization(mean=training_config.mean,
                                                              std=training_config.std)]))

        validation_data = FmriDataloader(dataset=valid_data, root_path=root_path,
                                           transform=transforms.Compose([
                                                                    CenterCrop(output_size=training_config.image_size),
                                                                    Rescale(output_size=training_config.image_size),
                                                                    RandomShift(),
                                                                    SampleToTensor(), # Only one transforming both
                                                                    Normalization(mean=training_config.mean,
                                                                                  std=training_config.std)]))"""

        dataloader_train = DataLoader(training_data, batch_size=args.batch_size,  # collate_fn=collate_fn,
                                      shuffle=True, num_workers=args.num_workers)
        dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size,  # collate_fn=collate_fn,
                                      shuffle=False, num_workers=args.num_workers)

        NUM_VOXELS = len(train_data[0]['fmri'])

    # TODO: NSD Dataloader
    elif args.dataset == 'NSD':
        image_crop = training_config.image_crop

        # Load data
        training_data = CocoDataloader(train_data, pickle=False,
                                       transform=transforms.Compose([transforms.CenterCrop((image_crop, image_crop)),
                                                                     transforms.Resize((training_config.image_size,
                                                                                        training_config.image_size)),
                                                                     transforms.RandomHorizontalFlip(),
                                                                     transforms.ToTensor(),
                                                                     GreyToColor(training_config.image_size),
                                                                     # converts greyscale to 3 channels
                                                                     transforms.Normalize(training_config.mean,
                                                                                          training_config.std)
                                                                     ]))

        dataloader_train = DataLoader(training_data, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.num_workers)

    else:
        logging.info('Wrong dataset')

    # This writer function is for torch.tensorboard
    writer = SummaryWriter(SAVE_PATH + '/runs_' + timestep)
    writer_encoder = SummaryWriter(SAVE_PATH + '/runs_' + timestep + '/encoder')
    writer_decoder = SummaryWriter(SAVE_PATH + '/runs_' + timestep + '/decoder')
    writer_discriminator = SummaryWriter(SAVE_PATH + '/runs_' + timestep + '/discriminator')

    # Load Stage 1 network weights
    model_dir = os.path.join(OUTPUT_PATH, args.dataset, 'pretrain', args.pretrained_net[0],
                               'pretrained_' + args.pretrained_net[0] + '_' + str(args.pretrained_net[1]) + '.pth')
    logging.info('Loaded network is:', model_dir)
    # TODO: Change 'pretrain' to stage 1 when it's working

    # Set model directory from Stage 1
    # model_dir = trained_net.replace(".pth", '_{}.pth'.format(args.load_epoch))

    # Initialize network and load Stage 1 weights
    # encoder = Encoder(z_size=args.latent_dim).to(device) # These aren't used anywhere TODO: Check delete?
    # decoder = Decoder(z_size=args.latent_dim, size=encoder.size).to(device)  # TODO: Check size var
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

    # Loading Checkpoint | If you want to continue training for existing checkpoint
    # Set checkpoint path
    if args.network_checkpoint is not None:
        net_checkpoint_path = os.path.join(OUTPUT_PATH, args.dataset, SUBJECT_PATH, 'stage_2', args.network_checkpoint,
                                   'stage_2_' + args.network_checkpoint + '.pth')
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

    # Variables for equilibrium to improve GAN stability
    margin = training_config.margin
    equilibrium = training_config.equilibrium
    lambda_mse = training_config.lambda_mse
    decay_mse = training_config.decay_mse

    # An optimizer and schedulers for each of the sub-networks, so we can selectively backprop
    optimizer_encoder = torch.optim.RMSprop(params=model.encoder.parameters(), lr=training_config.learning_rate,
                                            alpha=0.9,
                                            eps=1e-8, weight_decay=training_config.weight_decay, momentum=0,
                                            centered=False)
    lr_encoder = ExponentialLR(optimizer_encoder, gamma=training_config.decay_lr)

    # optimizer_decoder = torch.optim.RMSprop(params=model.decoder.parameters(), lr=training_config.learning_rate,
    #                                         alpha=0.9,
    #                                         eps=1e-8, weight_decay=training_config.weight_decay, momentum=0,
    #                                         centered=False)
    # lr_decoder = ExponentialLR(optimizer_decoder, gamma=training_config.decay_lr)

    optimizer_discriminator = torch.optim.RMSprop(params=model.discriminator.parameters(),
                                                  lr=training_config.learning_rate,
                                                  alpha=0.9, eps=1e-8, weight_decay=training_config.weight_decay,
                                                  momentum=0, centered=False)
    lr_discriminator = ExponentialLR(optimizer_discriminator, gamma=training_config.decay_lr)

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
    epochs_n = training_config.n_epochs

    for idx_epoch in range(args.epochs):

        try:

            # For each batch
            for batch_idx, data_batch in enumerate(dataloader_train):
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

                # VAE/GAN Loss
                loss_method = 'Maria' # 'Maria', 'Ren'

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
                train_dec = False

                # Initially try training without equilibrium
                train_equilibrium = False  # Leave off probably
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
                    os.makedirs(os.path.join(images_dir, 'random'))

                out = out.data.cpu()

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

                # only for one batch due to memory issue
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
            # plt.show() # TODO: Check this
    exit(0)
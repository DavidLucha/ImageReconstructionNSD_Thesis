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
import lpips

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
    StructuralSimilarity, objective_assessment, parse_args, FmriDataloader, potentiation, save_out

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def main():
    timestep = time.strftime("%Y%m%d-%H%M%S")

    """
    ARGS PARSER
    """
    arguments = True  # Set to False while testing

    if arguments:
        parser = argparse.ArgumentParser()
        # parser.add_argument('--input', help="user path where the datasets are located", type=str)

        # parser.add_argument('--run_name', default=timestep, help='sets the run name to the time shell script run',
        #                     type=str)
        parser.add_argument('--data_root', default=training_config.data_root,
                            help='sets directory of /datasets folder. Default set to scratch.'
                                 'If on HPC, use /scratch/qbi/uqdlucha/datasets/,'
                                 'If on home PC, us D:/Lucha_Data/datasets/', type=str)
        # Optimizing parameters | also, see lambda and margin in training_config.py
        parser.add_argument('--batch_size', default=training_config.batch_size, help='batch size for dataloader',
                            type=int)
        # parser.add_argument('--epochs', default=training_config.n_epochs, help='number of epochs', type=int)
        # parser.add_argument('--iters', default=30000, help='sets max number of forward passes. 30k for stage 2'
        #                                                    ', 15k for stage 3.', type=int)
        parser.add_argument('--num_workers', '-nw', default=training_config.num_workers,
                            help='number of workers for dataloader', type=int)
        # parser.add_argument('--lr_dec', default=.001, type=float)
        # parser.add_argument('--lr_disc', default=.0005, type=float)
        # parser.add_argument('--decay_lr', default=0.5,
        #                     help='.98 in Maria, .75 in original VAE/GAN', type=float)
        parser.add_argument('--seed', default=277603, help='sets seed, 0 makes a random int', type=int)
        parser.add_argument('--valid_shuffle', '-shuffle', default='False', type=str, help='defines whether'
                                                                                           'eval dataloader shuffles')
        parser.add_argument('--latent_dims', default=1024, type=int)
        # parser.add_argument('--beta', default=0.5, type=float)
        # parser.add_argument('--recon_loss', default='trad', type=str, help='sets whether to use pytroch mse'
        #                                                                    'or manual like in pretrain (manual)')
        parser.add_argument('--lin_size', default=1024, type=int,
                            help='sets the number of nuerons in cog lin layer')
        parser.add_argument('--lin_layers', default=1, type=int, help='sets how many layers of cog network ')
        # parser.add_argument('--optim_method', default='Adam',
        #                     help='defines method for optimizer. Options: RMS or Adam.', type=str)
        parser.add_argument('--standardize', default='False',
                            help='determines whether the dataloader uses standardize.', type=str)
        parser.add_argument('--save', default='True',
                            help='save individual reconstruction images?', type=str)
        # parser.add_argument('--disc_loss', default='Maria',
        #                     help='determines whether we use Marias loss or the paper based one for disc', type=str)
        # parser.add_argument('--WAE_loss', default='Maria',
        #                     help='determines whether we use Marias loss or the paper based one for WAE', type=str)
        # parser.add_argument('--lambda_WAE', default=1, help='sets the multiplier for paper GAN loss', type=int)
        # parser.add_argument('--lambda_GAN', default=10, help='sets the multiplier for individual GAN losses',
        #                     type=int)
        # parser.add_argument('--lambda_recon', default=1, help='weight of recon loss', type=int)
        # parser.add_argument('--clip_gradients', default='False',
        #                     help='determines whether to clip gradients or not', type=str)

        # Pretrained/checkpoint network components
        # parser.add_argument('--network_checkpoint', default=None, help='loads checkpoint in the format '
        #                                                                'vaegan_20220613-014326', type=str)
        # parser.add_argument('--checkpoint_epoch', default=90, help='epoch of checkpoint network', type=int)
        # parser.add_argument('--load_from',default='pretrain', help='sets whether pretrained net is from pretrain'
        #                                                           'or from stage_1 output', type=str)
        # parser.add_argument('--st1_net', default=training_config.pretrained_net,
        #                     help='pretrained network from stage 1', type=str)
        # parser.add_argument('--st1_load_epoch', default='final',
        #                     help='epoch of the pretrained model', type=str)
        parser.add_argument('--st3_net', default=training_config.pretrained_net,
                            help='pretrained network from stage 1', type=str)
        parser.add_argument('--st3_load_epoch', default='final',
                            help='epoch of the pretrained model', type=str)
        parser.add_argument('--load_from', default='root', help='loads from either networks folder or normal'
                                                                    ' stage 3 output', type=str)
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
                                           'Subj_0{}_{}_NSD_single_pres_train.pickle'.format(args.subject,
                                                                                             args.set_size))

        # Currently valid data is set to 'max' meaning validation data contains multiple image presentations
        # If you only want to evaluate a single presentation of images replace both 'max' in strings below ...
        # with 'single_pres'
        VALID_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'valid', 'max', args.ROI,
                                       'Subj_0{}_NSD_max_valid.pickle'.format(args.subject))
    else:
        TRAIN_DATA_PATH = os.path.join(args.data_root, 'GOD',
                                       'GOD_Subject{}_train_normed.pickle'.format(args.subject))
        VALID_DATA_PATH = os.path.join(args.data_root, 'GOD',
                                       'GOD_Subject{}_valid_normed.pickle'.format(args.subject))

    SAVE_PATH = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, args.ROI, SUBJECT_PATH,
                             "evaluation", args.st3_net + "_" + timestep)
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # saving_dir = os.path.join(SAVE_PATH, 'evaluation_{}_{}'.format(args.st3_net, timestep))

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
    # logging.info('timestep is ',timestep)

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

    dataloader_train = DataLoader(training_data, batch_size=args.batch_size, drop_last=False,  # collate_fn=collate_fn,
                                  shuffle=True, num_workers=args.num_workers)
    dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size, drop_last=False,  # collate_fn=collate_fn,
                                  shuffle=shuf, num_workers=args.num_workers)

    NUM_VOXELS = len(train_data[0]['fmri'])
    logging.info(f'Number of voxels: {NUM_VOXELS}')
    logging.info(f'Train data length: {len(train_data)}')
    logging.info(f'Validation data length: {len(valid_data)}')

    # Load Stage 3 network weights
    # model_dir = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, args.ROI, SUBJECT_PATH,
    #                              'stage_3', args.st3_net, args.st3_net + '_{}.pth'.format(args.st3_load_epoch))
    if args.load_from == "networks":
        model_dir = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, "networks",
                                 args.st3_net + '_{}.pth'.format(args.st3_load_epoch))
    else:
        model_dir = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, args.ROI, SUBJECT_PATH,
                                     'stage_3', args.st3_net, args.st3_net + '_{}.pth'.format(args.st3_load_epoch))

    decoder = Decoder(z_size=args.latent_dims, size=256).to(device)
    cognitive_encoder = CognitiveEncoder(input_size=NUM_VOXELS, z_size=args.latent_dims, lin_size=args.lin_size,
                                         lin_layers=args.lin_layers).to(device)
    model = WaeGanCognitive(device=device, encoder=cognitive_encoder, decoder=decoder,
                                    z_size=args.latent_dims).to(device)

    model.load_state_dict(torch.load(model_dir, map_location=device))

    # Fix encoder weights
    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    # Load and show results for checkpoint
    logging.info('Load pretrained model')

    # ----------- MAIN CODE ------------
    # Make directory
    images_dir = os.path.join(SAVE_PATH, 'images')

    if args.save == "True":
        save = True

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
    else:
        save = False

    # save_out(model, dataloader_valid, path=images_dir)
    pcc, ssim, mse = evaluate(model, dataloader_valid, norm=False, mean=training_config.mean,
                              std=training_config.std, path=images_dir, save=save, resize=200)
    # logging.info("Mean PCC: {:.2f}".format(pcc.item()))
    # logging.info("Mean SSIM: {:.2f}".format(ssim.item()))
    # logging.info("Mean MSE: {:.2f}".format(mse.item()))
    # logging.info("Mean LPIPS: {:.2f}".format(lpips.item()))
    # logging.info("Mean IS:", is_score)

    # Plot histogram for objective assessment
    obj_score = dict(pcc=[], ssim=[], lpips=[], mse=[])
    for top in [2, 5, 10]:
        obj_pcc, obj_ssim, obj_lpips, obj_mse, averages = objective_assessment(model, dataloader_valid, top=top,
                                                                               save_path=SAVE_PATH)
        obj_score['pcc'].append(obj_pcc.item())
        obj_score['ssim'].append(obj_ssim.item())
        obj_score['lpips'].append(obj_lpips.item())
        obj_score['mse'].append(obj_mse.item())

    logging.info("Mean PCC: {:.2f}".format(averages[0]))
    logging.info("Mean SSIM: {:.2f}".format(averages[1]))
    logging.info("Mean LPIPS: {:.2f}".format(averages[2]))
    logging.info("Mean MSE: {:.2f}".format(averages[3]))

    obj_results_to_save = pd.DataFrame(obj_score)
    results_to_save = pd.DataFrame(obj_results_to_save)
    results_to_save.to_csv(os.path.join(SAVE_PATH, "results.csv"), index=False)

    # Graphing for PCC
    for test in ('pcc', 'ssim', 'lpips', 'mse'):
        x_axis = ['2-way', '5-way', '10-way']
        y_axis = [obj_score[test][0], obj_score[test][1], obj_score[test][2]]
        bars = plt.bar(x_axis, y_axis, width=0.5)
        plt.axhline(y=0.5, xmin=0, xmax=0.33, linewidth=1, color='k')
        plt.axhline(y=0.2, xmin=0.33, xmax=0.66, linewidth=1, color='k')
        plt.axhline(y=0.1, xmin=0.66, xmax=1.0, linewidth=1, color='k')
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            plt.text(bar.get_x() + 0.10, yval + .005, f'{y_axis[i] * 100:.2f}')
        plt.ylabel(test)
        plt.title('Objective assessment')
        plot_sav = os.path.join(SAVE_PATH, "{}_objective_assessment".format(test))
        plt.savefig(plot_sav)

        # Clear plots
        plt.cla()

    logging.info("Objective score PCC: {:.2f}".format(obj_score['pcc'][0]))
    logging.info("Objective score SSIM: {:.2f}".format(obj_score['ssim'][0]))
    logging.info("Objective score LPIPS: {:.2f}".format(obj_score['lpips'][0]))
    logging.info("Objective score MSE: {:.2f}".format(obj_score['mse'][0]))
    plt.close()
    exit(0)


if __name__ == "__main__":
    main()

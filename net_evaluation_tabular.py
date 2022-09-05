# This takes the networks, and computes the metrics for each reconstruction and candidates

import os
import numpy
import json
import torch
import pickle
import logging
import argparse
import matplotlib.pyplot as plt
import random

from torchvision import transforms
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset

import training_config
from model_2 import VaeGan, Encoder, Decoder, VaeGanCognitive, Discriminator, CognitiveEncoder, WaeGan, WaeGanCognitive
from utils_2 import GreyToColor, evaluate, PearsonCorrelation, objective_assessment_table, \
    objective_assessment_table_batch,  FmriDataloader, save_network_out


def main():

    """
    ARGS PARSER
    """
    arguments = True  # Set to False while testing

    if arguments:
        parser = argparse.ArgumentParser()
        # parser.add_argument('--input', help="user path where the datasets are located", type=str)

        # parser.add_argument('--run_name', default=timestep, help='sets the run name to the time shell script run',
        #                     type=str)
        parser.add_argument('--data_root', default='D:/Lucha_Data/datasets/',
                            help='sets directory of /datasets folder. Default set to scratch.'
                                 'If on HPC, use /scratch/qbi/uqdlucha/datasets/,'
                                 'If on home PC, us D:/Lucha_Data/datasets/', type=str)
        parser.add_argument('--network_root', default='D:/Lucha_Data/final_networks/',
                            help='sets directory of /datasets folder. Default set to scratch.'
                                 'If on HPC, use /scratch/qbi/uqdlucha/final_networks/,'
                                 'If on home PC, us D:/Lucha_Data/final_networks/', type=str)
        # Optimizing parameters | also, see lambda and margin in training_config.py
        parser.add_argument('--batch_size', default=872, help='batch size for dataloader, set to full',
                            type=int)
        # parser.add_argument('--epochs', default=training_config.n_epochs, help='number of epochs', type=int)
        # parser.add_argument('--iters', default=30000, help='sets max number of forward passes. 30k for stage 2'
        #                                                    ', 15k for stage 3.', type=int)
        parser.add_argument('--num_workers', '-nw', default=training_config.num_workers,
                            help='number of workers for dataloader', type=int)
        parser.add_argument('--seed', default=277603, help='sets seed, 0 makes a random int', type=int)
        parser.add_argument('--valid_shuffle', '-shuffle', default='False', type=str, help='defines whether'
                                                                                           'eval dataloader shuffles')
        parser.add_argument('--latent_dims', default=1024, type=int)
        parser.add_argument('--lin_size', default=2048, type=int,
                            help='sets the number of nuerons in cog lin layer')
        parser.add_argument('--lin_layers', default=2, type=int, help='sets how many layers of cog network ')
        # parser.add_argument('--optim_method', default='Adam',
        #                     help='defines method for optimizer. Options: RMS or Adam.', type=str)
        parser.add_argument('--standardize', default='False',
                            help='determines whether the dataloader uses standardize.', type=str)
        parser.add_argument('--save', default='True',
                            help='save individual reconstruction images?', type=str)
        parser.add_argument('--st3_net', default=training_config.pretrained_net,
                            help='pretrained network from stage 1', type=str)
        parser.add_argument('--st3_load_epoch', default='final',
                            help='epoch of the pretrained model', type=str)
        parser.add_argument('--load_from', default='networks', help='loads from either networks folder or normal'
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
    OUTPUT_PATH = os.path.join(args.network_root, 'output/')
    SUBJECT_PATH = 'Subj_0{}/'.format(str(args.subject))

    # Load training data for GOD and NSD, default is NSD
    if args.dataset == 'NSD':
        VALID_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'valid', 'single_pres', args.ROI,
                                       'Subj_0{}_NSD_single_pres_valid.pickle'.format(args.subject))
        # VALID_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'old_valid', 'max', args.ROI,
        #                                'Subj_0{}_NSD_max_valid.pickle'.format(args.subject))

    SAVE_PATH = os.path.join(OUTPUT_PATH, args.vox_res, args.st3_net)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH, exist_ok=True)

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

    dataloader_valid = DataLoader(validation_data, batch_size=args.batch_size,  # drop_last=False,  # collate_fn=collate_fn,
                                  shuffle=shuf, num_workers=args.num_workers)

    NUM_VOXELS = len(valid_data[0]['fmri'])
    logging.info(f'Number of voxels: {NUM_VOXELS}')
    logging.info(f'Validation data length: {len(valid_data)}')

    # Load Stage 3 network weights
    # model_dir = os.path.join(OUTPUT_PATH, args.dataset, args.vox_res, args.set_size, args.ROI, SUBJECT_PATH,
    #                              'stage_3', args.st3_net, args.st3_net + '_{}.pth'.format(args.st3_load_epoch))

    if args.load_from == "networks":
        # You would need to change this.
        all_nets_root = args.network_root

        model_dir = os.path.join(all_nets_root, args.vox_res, 'all',
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

    # Test model load
    test_model = False

    if test_model:
        for batch_idx, data_batch in enumerate(dataloader_valid):
            logging.info('Testing model from pretraining')
            model.eval()
            data_in = Variable(data_batch['fmri'], requires_grad=False).float().to(device)
            data_target = Variable(data_batch['image'], requires_grad=False).float().to(device)
            out, _ = model(data_in)

            out = out.data.cpu()

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(out[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
            test_output_dir = os.path.join(SAVE_PATH, 'model_test_valid')
            plt.savefig(test_output_dir)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(data_target[: 25].cpu().detach(), nrow=5, normalize=True).permute(1, 2, 0))
            gt_dir = os.path.join(SAVE_PATH, 'model_test_ground_truth')
            plt.savefig(gt_dir)
            plt.show()
            exit(0)

    # raise Exception('Check output.')
    # ----------- MAIN CODE ------------
    # Make directory
    images_dir = os.path.join(SAVE_PATH, 'images')

    if args.save == "True":
        save = True

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
    else:
        save = False

    ## save_out(model, dataloader_valid, path=images_dir)

    # pcc, ssim, mse = evaluate(model, dataloader_valid, norm=False, mean=training_config.mean,
    #                           std=training_config.std, path=images_dir, save=save, resize=200)
    # logging.info("Mean PCC: {:.2f}".format(pcc.item()))
    # logging.info("Mean SSIM: {:.2f}".format(ssim.item()))
    # logging.info("Mean MSE: {:.2f}".format(mse.item()))

    ## logging.info("Mean LPIPS: {:.2f}".format(lpips.item()))
    ## logging.info("Mean IS:", is_score)

    # Plot histogram for objective assessment
    obj_score = dict(
        pcc=[],
        pcc_sd=[],
        ssim=[],
        ssim_sd=[],
        lpips=[],
        lpips_sd=[],
        mse=[],
        mse_sd=[]
    )

    if save:
        save_network_out(model, dataloader_valid, path=images_dir, save=save, resize=200)

    # gets all tables for each metric
    # TODO Get rid of this
    if args.batch_size == 872:
        table_pcc_pd, table_ssim_pd, table_lpips_pd = objective_assessment_table(model, dataloader_valid,
                                                                                 save_path=SAVE_PATH)
    else:
        table_pcc_pd, table_ssim_pd, table_lpips_pd = objective_assessment_table_batch(model, dataloader_valid,
                                                                                      save_path=SAVE_PATH)

    # obj_all['score']
    """obj_score['pcc'].append(statistics.mean(obj_all['pcc_score']))
    obj_score['pcc_sd'].append(statistics.stdev(obj_all['pcc_score']))
    obj_score['ssim'].append(statistics.mean(obj_all['ssim_score']))
    obj_score['ssim_sd'].append(statistics.stdev(obj_all['ssim_score']))
    obj_score['lpips'].append(statistics.mean(obj_all['lpips_score']))
    obj_score['lpips_sd'].append(statistics.stdev(obj_all['lpips_score']))
    obj_score['mse'].append(statistics.mean(obj_all['mse_score']))
    obj_score['mse_sd'].append(statistics.stdev(obj_all['mse_score']))

    # logging.info("Mean PCC (obj): {:.2f}".format(averages[0]))
    # logging.info("Mean SSIM (obj): {:.2f}".format(averages[1]))
    # logging.info("Mean LPIPS (obj): {:.2f}".format(averages[2]))
    # logging.info("Mean MSE (obj): {:.2f}".format(averages[3]))

    # obj_results_to_save = pd.DataFrame(obj_score)
    # results_to_save = pd.DataFrame(obj_results_to_save)
    # results_to_save.to_csv(os.path.join(SAVE_PATH, "results.csv"), index=False)

    # Graphing for PCC
    for test in ('pcc', 'ssim', 'lpips', 'mse'):
        x_axis = ['2-way', '5-way', '10-way']
        y_axis = [obj_score[test][0], obj_score[test][1], obj_score[test][2]]
        y_axis_err = [obj_score[test + '_sd'][0], obj_score[test + '_sd'][1], obj_score[test + '_sd'][2]]
        bars = plt.bar(x_axis, y_axis, width=0.5)
        plt.errorbar(x_axis, y_axis, yerr=y_axis_err, capsize=10, ecolor='black', linestyle='')
        plt.axhline(y=0.5, xmin=0, xmax=0.33, linewidth=1, color='k')
        plt.axhline(y=0.2, xmin=0.33, xmax=0.66, linewidth=1, color='k')
        plt.axhline(y=0.1, xmin=0.66, xmax=1.0, linewidth=1, color='k')
        # TODO: remove trhe blue line connecting the bars
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
    plt.close()"""
    exit(0)


if __name__ == "__main__":
    main()

# This script takes the full tables of each metric value and runs either n-way or pairwise comparison
# use all_net_eval instead
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


import training_config
from utils_2 import objective_assessment_table

def main():
    timestep = time.strftime("%Y%m%d-%H%M%S")

    """
    ARGS PARSER
    """
    arguments = True  # Set to False while testing

    if arguments:
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_root', default='D:/Lucha_Data/datasets/',
                            help='sets directory of /datasets folder. Default set to scratch.'
                                 'If on HPC, use /scratch/qbi/uqdlucha/datasets/,'
                                 'If on home PC, us D:/Lucha_Data/datasets/', type=str)

        parser.add_argument('--seed', default=277603, help='sets seed, 0 makes a random int', type=int)
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
    OUTPUT_PATH = os.path.join(args.data_root, '../../output/')
    SUBJECT_PATH = 'Subj_0{}/'.format(str(args.subject))

    # Load training data for GOD and NSD, default is NSD
    if args.dataset == 'NSD':
        VALID_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'valid', 'single_pres', args.ROI,
                                       'Subj_0{}_NSD_single_pres_valid.pickle'.format(args.subject))
        # VALID_DATA_PATH = os.path.join(args.data_root, 'NSD', args.vox_res, 'old_valid', 'max', args.ROI,
        #                                'Subj_0{}_NSD_max_valid.pickle'.format(args.subject))

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
    logging.info('Set up random seeds...\nSeed is: {}'.format(seed))

    """
    DEVICE SETTING
    """
    # Load data pickles for subject
    with open(VALID_DATA_PATH, "rb") as input_file:
        valid_data = pickle.load(input_file)
    logger.info("Loading validation pickles for subject {} from {}".format(args.subject, VALID_DATA_PATH))

    root_path = os.path.join(args.data_root, args.dataset + '/')

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

    # gets all tables for each metric
    table_pcc_pd, table_ssim_pd, table_lpips_pd = objective_assessment_table(model, dataloader_valid, save_path=SAVE_PATH)

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

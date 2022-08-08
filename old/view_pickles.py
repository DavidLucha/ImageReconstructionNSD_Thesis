import pickle
import pandas as pd
import numpy as np
from bdpy import BData
import csv
import os
import torch
import training_config as cfg
import random

LOAD_PATH = "D:/Honours/nsd_pickles/1.8mm/"
pickle_dir_norm = LOAD_PATH + "normed_concat_pickle/subj_04_normed_concat_trial_fmri_rand.pickle"
pickle_dir_raw = LOAD_PATH + "raw_concat_pickle/subj_04_raw_concat_trial_fmri_rand.pickle"
print("Reading betas from pickle file: ", pickle_dir_norm)
print("Reading betas from pickle file: ", pickle_dir_raw)
# sp = single_pres
# s1_sp_betas = pickle.load(pickle_dir_2)
# s1_sp_betas = pd.read_pickle(pickle_dir_2)

with open(pickle_dir_norm, "rb") as input_file:
    train_data_norm = pickle.load(input_file)

with open(pickle_dir_raw, "rb") as input_file:
    train_data_raw = pickle.load(input_file)


list_1 = []
list_2 = [1,2,3]
list_1.append(list_2)
list_3 = [4,5,6]
list_1.append(list_3)




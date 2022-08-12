import pickle
import pandas as pd
import numpy as np
from bdpy import BData
import csv
import os
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn import preprocessing
import training_config as cfg
import random

LOAD_PATH = "D:/Lucha_Data/datasets/NSD_normed/1.8mm/train/single_pres/"
pickle_dir = LOAD_PATH + "Subj_04_NSD_single_pres_train.pickle"
print("Reading betas from pickle file: ", pickle_dir)

LOAD_PATH = "D:/Lucha_Data/datasets/NSD/1.8mm/valid/max/VC/"
pickle_dir = LOAD_PATH + "Subj_04_NSD_max_valid.pickle"
print("Reading betas from pickle file: ", pickle_dir)

with open(pickle_dir, "rb") as input_file:
    train_data_2 = pickle.load(input_file)

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

col = train_data_raw.iloc[[4]]
col = col.to_numpy()
col_tens = torch.FloatTensor(col)
col_norm = F.normalize(col_tens)
col_norm_np = col_norm.numpy()

col_stand = standardize(col_tens)
col_stand_np = col_stand.numpy()

col_normalize = (col - 0.5) / 0.5
# col_normalize_np = col_normalize.numpy()

col_sknorm = preprocessing.normalize(col)
col_skscale = preprocessing.scale(col)


def standardize(fmri):
    mu = torch.mean(fmri)
    std = torch.std(fmri)
    return (fmri - mu) / std


list_1 = []
list_2 = [1,2,3]
list_1.append(list_2)
list_3 = [4,5,6]
list_1.append(list_3)











trans = torch.nn.Sequential(
                transforms.Normalize([0.5],[0.5])
            )

array = np.random.uniform(-100, 100, 400)
tens = torch.tensor(array)

normed_array = preprocessing.scale(array)
print(np.mean(normed_array))
print(np.std(normed_array))

mu = torch.mean(tens)
std = torch.std(tens)
new_tens = (tens - mu)/std

print('new mean {:.5f} and std {:.5f}'.format(torch.mean(new_tens), torch.std(new_tens)))

# tens = torch.randn(-100, 100, (20,))

# norm = F.nor
new_tens = trans(tens)





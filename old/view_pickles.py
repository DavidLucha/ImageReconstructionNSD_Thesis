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

def unpack():
    x = [1, 4]
    return x

x, y = unpack()

data_dir = os.path.join("D:/Honours/nsd_pickles/1.8mm", "raw_concat_pickle")

pickle_dir = os.path.join(data_dir, "subj_01_raw_concat_trial_fmri.pickle")
print("Reading betas from pickle file: ", pickle_dir)
# data = pd.read_pickle(pickle_dir)

# Add the rand sample data to the right of the dataframe for the last key of all array (V1-3+RAND)
rand_dir = os.path.join(data_dir, "subj_01_raw_concat_trial_fmri_rand.pickle")
print("Reading random sample betas from pickle file: ", rand_dir)
# rand_data = pd.read_pickle(rand_dir)
with open(pickle_dir, "rb") as input_file:
    data = pickle.load(input_file)
with open(rand_dir, "rb") as input_file:
    rand_data = pickle.load(input_file)

# Add rand_data to the right of dataframe
merged_data = pd.concat([data, rand_data], axis=1)
merged_count = 12115 + 8145
# Remove duplicate columns
cleaned_merged_data = merged_data.loc[:,~merged_data.columns.duplicated()].copy()

#Using list(df) to get the column headers as a list
column_names = list(cleaned_merged_data.columns)

# Add preprocessing
normed_subject_full_df = preprocessing.scale(cleaned_merged_data)
# bring back column header values
normed_subject_full_df_titled = pd.DataFrame(normed_subject_full_df, columns = column_names)
# read index rows starting from 1
normed_subject_full_df_titled.index = np.arange(1, len(normed_subject_full_df_titled) + 1)


# -------------

sub = 1
betas_dir = "D:/NSD/nsddata_betas/ppdata/subj0{}/func1pt8mm/betas_fithrf_GLMdenoise_RR".format(sub)
rand_ROI_dir = os.path.join(betas_dir, "subj_0{}_masked_betas_session01_rand_samp.txt".format(sub))

with open(rand_ROI_dir, 'r') as f:
    rand_array = pd.read_csv(f, sep=" ", header=None)[0]
    # rand_array_array = rand_array[0]

ROI_list_root = 'D:/NSD/inode/full_roi/'
# Load the master ROI list
ROI_list_dir = os.path.join(ROI_list_root, "subj_0{}_ROI_labels.csv".format(sub))
with open(ROI_list_dir, 'r') as f:
    ROI_list = pd.read_csv(f, sep=",", dtype=int)
V1toV3_array = ROI_list[ROI_list['ROI_Label'] < 7]['Voxel_ID']

V1_to_V3_n_rand = V1toV3_array.append(rand_array, ignore_index=True)

select_voxels = cleaned_merged_data.loc[:, V1_to_V3_n_rand]


# ----------------------
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





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
import statistics
import time

from utils_2 import nway_comp, pairwise_comp


# string = 'Study3_SUBJ04_1pt8mm_V1_to_V3_n_HVC_max_Stage3_20220822-062600'

# Walk through folders in all eval network folders
networks = 'D:/Lucha_Data/final_networks/output/1pt8mm/'
save_path = 'D:/Lucha_Data/final_networks/output/'

nway = True
pairwise = False

count = 0

# Set n way comparisons
ns = [2, 5, 10]
repeats = 10

# Empty list to house all networks pcc evals
pcc_master = {}
ssim_master = {}
lpips_master = {}

for folder in os.listdir(networks):
    start = time.time()
    count += 1

    # print(folder)
    folder_dir = os.path.join(networks, folder)

    # Get name of network
    sep = '_Stage3_'
    network = folder.split(sep, 1)[0]
    print('Evaluating network: {}'.format(network))

    # Load data
    print('Reading Data')
    pcc = pd.read_excel(os.path.join(folder_dir, 'pcc_table.xlsx'), engine='openpyxl', index_col=0)
    ssim = pd.read_excel(os.path.join(folder_dir, 'ssim_table.xlsx'), engine='openpyxl', index_col=0)
    lpips = pd.read_excel(os.path.join(folder_dir, 'lpips_table.xlsx'), engine='openpyxl', index_col=0)
    print('Data Ready')

    if count == 1:
        print('Saving reconstruction names')
        recon_names = list(pcc.columns)

    if nway:
        print('Running n-way comparisons')
        for n in ns:
            pcc_master[n] = {}
            ssim_master[n] = {}
            lpips_master[n] = {}

            print('Running n-way comparisons')
            pcc_nway_out = nway_comp(pcc, n=n, repeats=repeats, metric="pcc")
            print('PCC Complete')
            ssim_nway_out = nway_comp(ssim, n=n, repeats=repeats, metric="ssim")
            print('SSIM Complete')
            lpips_nway_out = nway_comp(lpips, n=n, repeats=repeats, metric="lpips")
            print('LPIPS Complete')

            # So, for n, add a dictionary that has the n as key, and another dictionary as the value
            # The second dictionary has run name as key and the accuracies of the repeats for values
            pcc_master[n][network] = pcc_nway_out
            ssim_master[n][network] = ssim_nway_out
            lpips_master[n][network] = lpips_nway_out
            print('Evaluations saved to master list.')

    if pairwise:
        # pcc_nway_out = nway_comp(data, n=2, repeats=10, metric="pcc")
        print('Running pairwise comparisons')
        pcc_pairwise_out = pairwise_comp(pcc, metric="pcc")
        print('PCC Complete')
        ssim_pairwise_out = pairwise_comp(ssim, metric="ssim")
        print('SSIM Complete')
        lpips_pairwise_out = pairwise_comp(lpips, metric="lpips")
        print('LPIPS Complete')

        pcc_master[network] = pcc_pairwise_out
        ssim_master[network] = ssim_pairwise_out
        lpips_master[network] = lpips_pairwise_out
        print('Evaluations saved to master list.')

    end = time.time()
    print('Time per run =', end - start)

    if count == 3:
        break

if pairwise:
    pcc_save = pd.DataFrame(pcc_master, index=recon_names)
    ssim_save = pd.DataFrame(ssim_master, index=recon_names)
    lpips_save = pd.DataFrame(lpips_master, index=recon_names)
    print('Dataframes established.')

    print('Saving data...')
    pcc_save.to_excel(os.path.join(save_path, "pcc_master_out.xlsx"))
    ssim_save.to_excel(os.path.join(save_path, "ssim_master_out.xlsx"))
    lpips_save.to_excel(os.path.join(save_path, "lpips_master_out.xlsx"))
    print('Complete.')

exit(69)












# list_tot = {2: {},
#             5: {},
#             10: {}}

list_tot = {}

list_1 = [1, 2, 3, 4]
list_2 = [5, 6, 7, 8]

# list_tot.append(list_1)
# list_tot.append(list_2)

key_list = ['hello', 'there']
value_list = [list_1, list_2]

for n in [2, 5, 10]:
    list_tot[n] = {}
    for value in value_list:
        for key in key_list:
            list_tot[n][key] = value

# list_pd = pd.DataFrame(list_tot)

# twoway = list_pd[2]

with pd.ExcelWriter('D:/Lucha_Data/final_networks/output/test.xlsx') as writer:
    for n in [2, 5, 10]:
        list = pd.DataFrame(list_tot[n])
        list.to_excel(writer, sheet_name='{}-way_comparison'.format(n))
















path = 'D:/Lucha_Data/datasets/output/NSD/1pt8mm/max/VC/Subj_01/evaluation/Study1_SUBJ01_1pt8mm_VC_max_Stage3_20220817-112810_20220902-100454'
file = os.path.join(path, 'pcc_table.xlsx')

data = pd.read_excel(file, engine='openpyxl', index_col=0)

repeats = [10, 25, 50, 100, 250]
for repeat in repeats:
    print(repeat)
    pcc_nway_out = nway_comp(data, n=2, repeats=repeat, metric="pcc")

# pcc_nway_out = nway_comp(data, n=2, repeats=10, metric="pcc")
pcc_pairwise_out = pairwise_comp(data, metric="pcc")

raise Exception('stop')





# Testing the rand and rand replace thing
test = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test = pd.DataFrame(test)
for i in range(4):
    rand = test.sample(n=5, replace=False)
    rand_replace = test.sample(n=5, replace=True)
    print('rand at {}: {}'.format(i, rand))
    print('rand_replace at {}: {}'.format(i, rand_replace))





# developing the new eval code.
n = 5
total_score = 0
trials = 0

# n-way comparison
for i, row in data.iterrows():
    trials += 1
    print('Comparison {}'.format(trials))
    score_pcc = 0
    row = row.to_frame()
    matched = row.loc[i]
    print('Value of matched is: ',matched)
    new_row = row.drop([i])
    rand = row.sample(n=n-1, replace=False, random_state=-2012)
    for idx, rand_val in rand.iterrows():
        if matched.item() > rand_val.item():
            # TODO: Should this not be abs? No.
            score_pcc += 1
    if score_pcc == n - 1:
        total_score += 1
        print('Winner!')
    else:
        print('Recon? More like bozo.')
    print(rand)
    if trials==10:
        break

accuracy = total_score / trials * 100
print('Accuracy rate is {}%'.format(accuracy))


rowwise_full = []
trials = 0

# pairwise comparison
for i, row in data.iterrows():
    trials += 1
    score_pcc = 0
    row_count = 0
    print('Comparison {}'.format(trials))
    row = row.to_frame()
    matched = row.loc[i]
    print('Value of matched is: ',matched)
    new_row = row.drop([i])
    # rand = row.sample(n=n-1, replace=False, random_state=2012)
    for idx, comparison in new_row.iterrows():
        row_count += 1
        if matched.item() > comparison.item():
            # TODO: Should this not be abs? No.
            score_pcc += 1

    rowwise_accuracy = score_pcc / row_count * 100
    print('Recon of {} accuracy is {:.2f}'.format(i, rowwise_accuracy))
    rowwise_full.append(rowwise_accuracy)

    if trials==10:
        break

print('Average pairwise accuracy: {:.2f} \n'
      'Standard deviation of pairwise accuracy: {:.2f}'.format(statistics.mean(rowwise_full),
                                                               statistics.stdev(rowwise_full)))

raise Exception('check')


replace = 'David is a fucking_'
list = [1, 3, 'David is a fucking_goat']
new_string = list[2].replace(replace,'')
print(new_string)

single_pres = 'D:/Lucha_Data/datasets/NSD/1pt8mm/valid/single_pres/VC/'
single_pres = os.path.join(single_pres, 'Subj_01_NSD_single_pres_valid.pickle')
with open(single_pres, "rb") as input_file:
    data_final = pickle.load(input_file)

data_dir = os.path.join("D:/Honours/nsd_pickles/1pt8mm", "raw_concat_pickle")

pickle_dir = os.path.join(data_dir, "subj_01_raw_concat_trial_fmri.pickle")
print("Reading betas from pickle file: ", pickle_dir)
# data = pd.read_pickle(pickle_dir)

with open(pickle_dir, "rb") as input_file:
    data_full = pickle.load(input_file)

new_list=[]
for i in data_final:
    fmri = i['fmri'].tolist()
    image = i['image']
    comb = fmri.insert(0, image)
    # image = i['image']
    # comb = fmri.insert(0, image)
    new_list.append(fmri)

np.savetxt("D:/Lucha_Data/misc/subj1_valid.csv", new_list, delimiter=',', fmt='%s')











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





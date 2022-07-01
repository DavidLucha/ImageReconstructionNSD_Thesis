import pickle
import pandas as pd
import numpy as np
from bdpy import BData
import csv
import os
import torch
import training_config as cfg

stage = 'stage_1'
lr = cfg.learning_rate

dalo = None
if dalo is not None:
    print('helo')
if dalo is None:
    print('helo none')


# Potentiation code
# xc = a * b c
def potentiation(start_lr, decay_lr, epochs):
    x_c = start_lr * (decay_lr ** epochs)
    return x_c


# Test
# Start = 10, 2, 10 times
x_c = potentiation(0.0003, 0.98, 10)
print(x_c)

x = torch.tensor(1)
print(torch.sigmoid(x))

-torch.log(x - x)

"""
MARIA DATA (BOLD)


file_name = "D:/Honours/Maria Replication/bold_valid_all_fixed.pickle"
train_img_file = "D:/Honours/Maria Replication/stimuli_train.pickle"
val_img_file = "D:/Honours/Maria Replication/stimuli_valid.pickle"

objects =  pd.read_pickle(file_name)

objects[0][4]['fmri']
ex_pairing = objects[0][4]
ex_obj = objects[0][4]['fmri']
fmri_size = len(ex_obj)
print(ex_obj)

df = pd.DataFrame.from_dict(ex_pairing)
df.to_csv (r'dict_test.csv', index = False, header=True)

pd_objects = pd.read_pickle(file_name)
pd_train_img_obj = pd.read_pickle(train_img_file)
pd_val_img_obj = pd.read_pickle(val_img_file)

pd_train_img_obj[1]
"""

"""
# GOD Dataset from Gaviz
gaviz_fmri_fn = "D:/Honours/Gaviz Downloads/images_112.npz"
gaviz_fmri = np.load(gaviz_fmri_fn)

SUBJ01PATH = "D:/Honours/Gaviz Downloads/sbj_1.npz"
gav_subj_4 = np.load(SUBJ01PATH)
"""

# LOAD THE GOD DATA PER SUBJ
# Testing the bdpy import
subject = 'Subject2'

subj_PATH = "D:/Honours/Object Decoding Dataset/7387130/" + subject + '.h5'
subj_data = BData(subj_PATH)

subj_data.show_metadata()

# Trying to pull voxels if datatype is 1
# met_test = subj_data.select(condition='DataType')
training_idx = 1200
test_idx_start = 1200
test_idx_end = 2950

# GRAB ALL VOXELS AND SELECT 1200
subj_fmri = subj_data.select('ROI_VC')[test_idx_start:test_idx_end] # Picking those with datatype = 1 manually, training
# subj_fmri_test = subj_data.select('ROI_VC')[1201:2950]
test_grab = subj_data.metaData.value[7]
test_grab_2 = subj_data.metaData.value[7]
np.savetxt("subj_1_vox_id.csv", test_grab, delimiter=",")
np.savetxt("subj_2_vox_id.csv", test_grab_2, delimiter=",")


image_idx_test = subj_data.select('image_index')[test_idx_start:test_idx_end]
subj_img_lbl = pd.DataFrame(image_idx_test, dtype='int')
# np.savetxt("image_order.csv", image_idx, delimiter=",")

# LOAD CSV
# Make array with image labels in order + file name with file path appended

TRAINING_IMAGE_LIST_PATH = "D:/Honours/Object Decoding Dataset/images_passwd/images/image_training_id_nmd.csv"
with open(TRAINING_IMAGE_LIST_PATH, 'r') as f:
    image_list = list(csv.reader(f, delimiter=","))

pd_img_list = pd.DataFrame(image_list)

subj_img_lbl.columns = ['Image_ID']
pd_img_list.columns = ['Image_ID', 'File_Name']

subj_img_lbl = subj_img_lbl.astype('int32')
pd_img_list = pd_img_list.astype({'Image_ID': 'int32'})
pd_img_list = pd_img_list.astype({'File_Name': 'string'})

# img_file_names = pd.merge(subj_img_lbl, pd_img_list, left_index=True, right_index=True)
img_file_names = subj_img_lbl.merge(pd_img_list, how='inner', on='Image_ID')

IMG_LOCATION_PATH = "D:/Honours/Object Decoding Dataset/images_passwd/images/training/"

img_file_names['File_Name'] = IMG_LOCATION_PATH + img_file_names['File_Name'].astype(str)
np_img_paths = img_file_names.to_numpy()

######################

# Combine to arrays into list of dicts
# [{'fmri': ROW_FROM_DATA_V1_TRAINING, 'image': FILE_NAME_FROM_IMG_FILES},
# {'fmri': ROW_FROM_DATA_V1_TRAINING, 'image': FILE_NAME_FROM_IMG_FILES},
# {'fmri': ROW_FROM_DATA_V1_TRAINING, 'image': FILE_NAME_FROM_IMG_FILES},
# {'fmri': ROW_FROM_DATA_V1_TRAINING, 'image': FILE_NAME_FROM_IMG_FILES},]

# subj_fmri[0:3, 0:5]

"""
key_labels = ['fmri', 'image']
test_fm = subj_fmri[0:3]
# np_img_paths[1, 1]
test_im = np_img_paths[1, 1]
"""

#STIMULI PATHS
STIMULI_PATHS = np_img_paths[:,1]

# create dataset: fmri + image paths
fmri_image_dataset = []
for idx, (vox, pix) in enumerate(zip(subj_fmri, STIMULI_PATHS)):
    fmri_image_dataset.append({'fmri': vox, 'image': pix})

# return fmri_image_dataset


"""
image_list_add = np.array(image_idx, dtype=int)
mapping = dict(zip(image_list[:,0], range(len(image_list))))
comb_list = np.hstack((image_list_add, np.array([image_list[mapping[key],1:]
                            for key in image_list_add[:,0]])))
"""
"""
# Adds a row of numbers but is not v efficient
rowNums = np.arange(1,1200)
rowNums = rowNums[..., None]
img_num = np.concatenate((np_image_fn,rowNums),axis=1)
"""

LOAD_PATH = "D:/Lucha_Data/datasets/GOD/"
S4_train = pd.read_pickle(LOAD_PATH + 'GOD_Subject4_train_normed.pickle')
S4_test = pd.read_pickle(LOAD_PATH + 'GOD_Subject4_valid_normed.pickle')
S4_test_avg = pd.read_pickle(LOAD_PATH + 'GOD_Subject4_valid_normed_avg.pickle')


LOAD_PATH = "D:/Lucha_Data/datasets/GOD/"

subjects = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5']

subj_01 = pd.read_pickle(LOAD_PATH + 'fmri_image_' + subjects[0] + '.pickle')
subj_02 = pd.read_pickle(LOAD_PATH + 'fmri_image_' + subjects[1] + '.pickle')
subj_03 = pd.read_pickle(LOAD_PATH + 'fmri_image_' + subjects[2] + '.pickle')
subj_04 = pd.read_pickle(LOAD_PATH + 'fmri_image_' + subjects[3] + '.pickle')
subj_05 = pd.read_pickle(LOAD_PATH + 'fmri_image_' + subjects[4] + '.pickle')

LOAD_PATH = "D:/Lucha_Data/datasets/GOD/"
full_train_data = pd.read_pickle(LOAD_PATH + '_OLD_GOD_allsub_training.pickle')
new_train_data = pd.read_pickle(LOAD_PATH + 'GOD_all_subjects_train.pickle')
valid_data = pd.read_pickle(LOAD_PATH + 'GOD_all_subjects_valid.pickle')
valid_data_avg = pd.read_pickle(LOAD_PATH + 'GOD_all_subjects_valid_avg.pickle')

# Append all subject lists

training_data = subj_01 + subj_02 + subj_03 + subj_04 + subj_05


range = np.arange(1,60,1)

for i in range:
    # print(i)
    modulo = i % 20
    # print(i, 'modulo: ', modulo)
    if not i % 20: # Thought of, if no remainder
        print(i, 'hello world')

x = torch.tensor([3.0])
metric_value = torch.tensor(x, dtype=torch.float64).item()
print(metric_value)
metric_clone = x.detach().clone().item()
print(metric_clone)

for i in range(3):
    print(i)

encoder_channels = [64, 128, 256]
print(encoder_channels[1])


import numpy as np
Subjects = ['sbj_1', 'sbj_2', 'sbj_3']
LOAD_PATH = "D:/Honours/Gaviz Downloads/" + Subjects[1] + '.npz'
sbj_2 = np.load(LOAD_PATH)
# sbj_1 = 4466
# sbj_2 = 4404

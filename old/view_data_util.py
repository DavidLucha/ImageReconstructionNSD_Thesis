import pickle
import pandas as pd
import numpy as np
from bdpy import BData
import csv
import os
import torch
import training_config as cfg
import random

identity = lambda x: x

range = np.arange(1,10)
print(identity(range))

def mami(stage):
    if stage == 2:
        stage_out = 2
        print('this is stage 2')
        return stage_out
    mami_out = 4
    print('this will happen')
    return mami_out

out = mami(2)

print(out)

# BREAK TEST
counter = 0
max_count = 700
while counter < max_count:
    for i in range(1,10):
        for n in range(0,100):
            if counter < max_count:
                if counter >= 680:
                    print('more stuff, iter: {}'.format(counter))
                counter += 1
            else:
                break
        print('finished, counter is {}. Continue with saving shit.'.format(counter))
print('save model at counter {}'.format(counter))
print('welcome back.')


"""
Testing random selection of dataset
"""
random.seed(76)
# 1 - 37, 3 - 29, 4 - 27

LOAD_PATH = "D:/Lucha_Data/datasets/NSD/1.8mm/"
pickle_dir = LOAD_PATH + "train/max/Subj_04_NSD_max_train.pickle"
print("Reading betas from pickle file: ", pickle_dir)
# sp = single_pres
# s1_sp_betas = pickle.load(pickle_dir)
# s1_sp_betas = pd.read_pickle(pickle_dir)

with open(pickle_dir, "rb") as input_file:
    train_data = pickle.load(input_file)

# Could use random.sample but this would not be reproducible.
# We would want the same subset of images between 1.8mm and 3mm
large = random.sample(train_data, 7500)
medium = random.sample(large, 4000)
small = random.sample(medium, 1200)

# Get random arrays and save to pickles for reproducibility
get_array = True
if get_array:
    # So we will use an array and pickle load this
    data_avail = [8859, 8188, 7907] # 37, 29, 27
    # data_size = [1200, 4000, 7500, 8000]
    data_size = [7500, 4000, 1200]
    # data_avail[1:]
    output_path = "D:/Lucha_Data/datasets/NSD/misc/dataset_randomness"

    for n in data_avail:
        # Gets the 8000 from either avail
        # Or just do it in the code - to save space
        rnd_array = np.random.randint(0, n-1, data_size[0])
        print(rnd_array, len(rnd_array), min(rnd_array), max(rnd_array))

        with open(os.path.join(output_path, data_size[0] + '_NSD_single_pres_' + n + '_train_array.pickle'), 'wb') as f:
            pickle.dump(rnd_array, f)

    med_array = np.random.randint(0, data_size[0]-1, data_size[1])
    print(med_array, len(med_array), min(med_array), max(med_array))
    small_array = np.random.randint(0, data_size[1]-1, data_size[2])
    print(small_array, len(small_array), min(small_array), max(small_array))

    with open(os.path.join(output_path, data_size[1] + '_NSD_single_pres_train_array.pickle'), 'wb') as f:
        pickle.dump(small_array, f)

    with open(os.path.join(output_path, data_size[2] + '_NSD_single_pres_train_array.pickle'), 'wb') as f:
        pickle.dump(med_array, f)






########################################### END ################################################

stage = 'stage_1'
lr = cfg.learning_rate

# Test loading betas from NSD concat
# subj_04_betas = pd.read_pickle("D:/Lucha_Data/datasets/NSD/1.8mm/raw_concat_pickle/subj_04_raw_concat_trial_fmri.pickle")
# subj_04_betas_normed = pd.read_pickle("D:/Lucha_Data/datasets/NSD/1.8mm/normed_concat_pickle/subj_04_normed_concat_trial_fmri.pickle")

# Test loading list dicts from NSD concat
LOAD_PATH = "D:/Honours/nsd_pickles/"
pickle_dir = LOAD_PATH + "normed_concat_pickle/subj_07_normed_concat_trial_fmri.pickle"
print("Reading betas from pickle file: ", pickle_dir)
s7_fmri = pd.read_pickle(pickle_dir)
S7_train = pd.read_pickle(LOAD_PATH + 'train/max/Subj_07_NSD_max_train.pickle')
S7_test = pd.read_pickle(LOAD_PATH + 'valid/max/Subj_07_NSD_max_valid.pickle')
S7_train_single = pd.read_pickle(LOAD_PATH + 'train/single_pres/Subj_07_NSD_single_pres_train.pickle')
S7_test_single = pd.read_pickle(LOAD_PATH + 'valid/single_pres/Subj_07_NSD_single_pres_valid.pickle')

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


# Testing formatting
for x in [3, 14]:
    print("This is the {:02} number".format(x))

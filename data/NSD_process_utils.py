# Dependencies
import pandas as pd
import os.path
import os
import numpy as np
import h5py
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import csv

"""
NSD Preprocessing
"""


class NSDProcess:
    def __init__(self, root_path, download_path=None):
        # Set directory for stimuli file
        self.stimuli_file = os.path.join(root_path, "NSD/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
        self.download_path = download_path
        if download_path is not None:
            download = True
            # Change the output path
            self.output_path = os.path.join(root_path, "NSD/nsddata_stimuli/stimuli/nsd/shared_stimuli/")
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

    def read_images(self, image_index, show=False, download=False):
        """read_images reads a list of images, and returns their data

        Parameters
        ----------
        image_index : list of integers
            which images indexed in the 73k format to return
        show : bool, optional
            whether to also show the images, by default False

        Returns
        -------
        numpy.ndarray, 3D
            RGB image data
        """
        # if not hasattr(self, 'stim_descriptions'):
        #     self.stim_descriptions = pd.read_csv(
        #         self.stimuli_description_file, index_col=0)

        # Set directory for stimuli file
        # root_path = "D:"
        # self.stimuli_file = os.path.join(root_path, "NSD/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")

        sf = h5py.File(self.stimuli_file, 'r')
        sdataset = sf.get('imgBrick')
        if show:
            f, ss = plt.subplots(1, len(image_index),
                                 figsize=(6 * len(image_index), 6))
            if len(image_index) == 1:
                ss = [ss]
            for s, d in zip(ss, sdataset[image_index]):
                s.axis('off')
                s.imshow(d)
                # d = Image.fromarray(d)
                # d.save(os.path.join(self.output_path + self.download_path + f"_nsd{image_index}.png"))
            plt.show()

        if download:
            for i in image_index:
                im = Image.fromarray(sdataset[i])  #.astype(np.uint8))
                im.save(os.path.join(self.output_path + self.download_path + "_nsd{:05d}.png".format(i)))
            # output_path = ""

        # return sdataset[image_index]


def nsd_data_dict_prep(data_dir, image_list_dir, subj_list, vox_res, data_type, save_path, single_pres=False):
    """
    Loads beta, pulls voxels from ROI and image index. (TRAINING)
    Combines these into a list of dictionaries formatted as:
    [{'fmri': VOXEL_DATA_ARRAY, 'image': IMAGE_PATH}, ...
    {'fmri': VOXEL_DATA_ARRAY, 'image': IMAGE_PATH}]

    Saves these training and test splits per subject

    Per voxel resolution:
    Each subject has:
    - Full train split (27k)
    - Full test split (3k)
    - Maximum single presentation (need to account for max trials per subject)
    - 4000 single presentation
    - 1200 single presentation

    :param data_dir: should be:
        "D:/Honours/Object Decoding Dataset/7387130/"
    :param image_list_dir: should lead to numbered list of image names:
        "D:/Honours/Object Decoding Dataset/images_passwd/images/image_training_id_nmd.csv"

    Note: Works for the GOD data, unsure if this will work for NSD

    """
    train = False
    valid = False
    if data_type == 'train':
        train = True
    if data_type == 'valid':
        valid = True
    if data_type == 'both':
        train = True
        valid = True

    session_count = [0, 37, 37, 29, 27, 37, 29, 37, 27]

    for s in subj_list:
        sub = "Subj_0{}".format(s)

        trial_count = session_count[s] * 750

        sub_img_list = os.path.join(image_list_dir, "{}_Trial_Lists_Sorted.csv".format(sub))
        print("sub_img_list is located at: {}".format(sub_img_list))

        with open(sub_img_list, 'r') as f:
            image_list_ori = pd.read_csv(f, sep=",", dtype=int)

        image_list = image_list_ori.set_index('Trial_No', drop=False)

        if not single_pres:
            print('Compiling data from whole range (all three presentations)')
            # Get dset with only shared or not
            train_list_full = image_list[image_list['Shared'] == 0]
            test_list_full = image_list[image_list['Shared'] == 1]

            # Cuts down datasets to trial count per subject (factoring in withheld NSD data for algonauts)
            train_list = train_list_full.loc[train_list_full["Trial_No"] <= trial_count]
            test_list = test_list_full.loc[test_list_full["Trial_No"] <= trial_count]

            # Pull array
            train_array = train_list['Trial_No'].to_numpy()
            test_array = test_list['Trial_No'].to_numpy()
            pres = "max"

        if single_pres:
            print('Compiling data from only first of three presentations')
            # Get dset with only shared or not
            train_list_full = image_list[image_list['Shared'] == 0]
            test_list_full = image_list[image_list['Shared'] == 1]

            # Cuts down datasets to trial count per subject (factoring in withheld NSD data for algonauts)
            train_list = train_list_full.loc[train_list_full["Trial_No"] <= trial_count]
            test_list = test_list_full.loc[test_list_full["Trial_No"] <= trial_count]

            # Gets dset with just first presentation | Good for study 2
            single_pres_train_list = train_list[train_list['Nth_Pres'] == 1]
            single_pres_test_list = test_list[test_list['Nth_Pres'] == 1]

            # Pull array
            train_array = single_pres_train_list['Trial_No'].to_numpy()
            test_array = single_pres_test_list['Trial_No'].to_numpy()
            pres = "single_pres"

        # File name is Subj_01_nsd00001.png format

        # We'll call the images from the Lucha_Data dataset
        # FILE_NAME_OUTPUT = '/images/' + image_type  # Defines the path in the dictionary outputs
        pickle_dir = os.path.join(data_dir, "subj_0{}_normed_concat_trial_fmri.pickle".format(s))
        print("Reading betas from pickle file: ",pickle_dir)
        data = pd.read_pickle(pickle_dir)

        if train:
            image_type = 'train/'
            FILE_NAME_OUTPUT = '/images/' + image_type  # Defines the path in the dictionary outputs
            fmri_image_dataset = []

            fmri = data.loc[train_array]
            # Convert fMRI data for zipping
            fmri_zip = fmri.to_numpy()
            # Make dataframe with just image_idx
            image_idx = train_list.filter(['Image_Idx'], axis=1)
            # Update numbering with leading 0s
            image_names = image_idx['Image_Idx'].astype(str).str.zfill(5)
            # Convert back to dframe
            image_names = image_names.to_frame(name='Image_Idx')
            # Update values in cells to full image paths
            image_names['Image_Idx'] = FILE_NAME_OUTPUT + sub + "_nsd" + image_names['Image_Idx'] + ".png"
            # Set to strings
            image_names = image_names['Image_Idx'].astype(str)
            # Combine for list of dictionaries
            for idx, (vox, pix) in enumerate(zip(fmri_zip, image_names)):
                fmri_image_dataset.append({'fmri': vox, 'image': pix})
            # Save pickle
            output_path = os.path.join(save_path, image_type, pres)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, sub + '_NSD_' + pres + '_train.pickle'),
                      'wb') as f:
                # e.g., /Subj_01_NSD_max_train.pickle OR /Subj_01_NSD_single_pres_train.pickle
                pickle.dump(fmri_image_dataset, f)

        if valid:
            image_type = 'valid/'
            FILE_NAME_OUTPUT = '/images/' + image_type  # Defines the path in the dictionary outputs
            fmri_image_dataset = []

            fmri = data.loc[test_array]
            # Convert fMRI data for zipping
            fmri_zip = fmri.to_numpy()
            # Make dataframe with just image_idx
            image_idx = test_list.filter(['Image_Idx'], axis=1)
            # Update numbering with leading 0s
            image_names = image_idx['Image_Idx'].astype(str).str.zfill(5)
            # Convert back to dframe
            image_names = image_names.to_frame(name='Image_Idx')
            # Update values in cells to full image paths
            image_names['Image_Idx'] = FILE_NAME_OUTPUT + "shared_nsd" + image_names['Image_Idx'] + ".png"

            image_names = image_names['Image_Idx'].astype(str)

            for idx, (vox, pix) in enumerate(zip(fmri_zip, image_names)):
                fmri_image_dataset.append({'fmri': vox, 'image': pix})

            # raise Exception()
            output_path = os.path.join(save_path, image_type, pres)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, sub + '_NSD_' + pres + '_valid.pickle'),
                      'wb') as f:
                # e.g., /Subj_01_NSD_max_valid.pickle OR /Subj_01_NSD_single_pres_valid.pickle
                pickle.dump(fmri_image_dataset, f)


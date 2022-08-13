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


def nsd_data_dict_prep(data_dir, image_list_dir, subj_list, vox_res, data_type, save_path, norm, single_pres=False):
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

    DEFAULT was to use the normed pickles, but I think there's an issue with this.
    I think it's best to just normalize as we go into the network.

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

        # switch to pull pickles for each ROI comparison
        pull_ROI = True
        # s=1 #  TODO: remove
        if pull_ROI:
            # Here we will define the arrays in order to pull the correct columns from the overall dataset
            ROI_list_root = 'D:/NSD/inode/full_roi/'
            # Load the master ROI list
            ROI_list_dir = os.path.join(ROI_list_root, "subj_0{}_ROI_labels.csv".format(s))
            print("ROI list is located at: {}".format(ROI_list_dir))
            betas_dir = "D:/NSD/nsddata_betas/ppdata/subj0{}/func1pt8mm/betas_fithrf_GLMdenoise_RR".format(s)
            rand_ROI_dir = os.path.join(betas_dir, "subj_0{}_masked_betas_session01_rand_samp.txt".format(s))

            with open(ROI_list_dir, 'r') as f:
                ROI_list = pd.read_csv(f, sep=",", dtype=int)

            with open(rand_ROI_dir, 'r') as f:
                # Grab ROI IDs of random voxel selection
                rand_array = pd.read_csv(f, sep=" ", header=None)[0]

            # Gets the voxel IDs of those matching ROI label (see supplementary for data manual)
            # .isin checks ROI_Label for those values and returns them
            # e.g. 1, 2 are V1v and V1d, respectively - therefore pulls voxel IDs of V1
            # All voxels (only included to make it easier to iterate through)
            VC_array = ROI_list['Voxel_ID']
            V1_array = ROI_list[ROI_list['ROI_Label'].isin([1, 2])]['Voxel_ID']
            V2_array = ROI_list[ROI_list['ROI_Label'].isin([3, 4])]['Voxel_ID']
            V3_array = ROI_list[ROI_list['ROI_Label'].isin([5, 6])]['Voxel_ID']
            V1toV3_array = ROI_list[ROI_list['ROI_Label'] < 7]['Voxel_ID']
            V4_array = ROI_list[ROI_list['ROI_Label'] == 7]['Voxel_ID']
            # HVC includes faces-, places-, body- selective areas (not V4)
            HVC_array = ROI_list[ROI_list['ROI_Label'] >= 8]['Voxel_ID']
            # Everything but V4
            V1toV3_HVC_array = ROI_list[ROI_list['ROI_Label'] != 7]['Voxel_ID']
            # Add the rand IDs onto V1-V3 array
            V1toV3_rand_array = V1toV3_array.append(rand_array, ignore_index=True)

            all_arrays = {
                "VC": VC_array,
                "V1": V1_array,
                "V2": V2_array,
                "V3": V3_array,
                "V1_to_V3": V1toV3_array,
                "V4": V4_array,
                "HVC": HVC_array,
                "V1_to_V3_n_HVC": V1toV3_HVC_array,
                "V1_to_V3_n_rand": V1toV3_rand_array}

            # for key in all_arrays:
            #     print(key)
            #     print(all_arrays[key])

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
        pickle_dir = os.path.join(data_dir, "subj_0{}_{}_concat_trial_fmri.pickle".format(s,norm))
        print("Reading betas from pickle file: ", pickle_dir)
        # data = pd.read_pickle(pickle_dir)
        with open(pickle_dir, "rb") as input_file:
            data = pickle.load(input_file)

        # Add the rand sample data to the right of the dataframe for the last key of all array (V1-3+RAND)
        rand_dir = os.path.join(data_dir, "subj_0{}_{}_concat_trial_fmri_rand.pickle".format(s,norm))
        print("Reading random sample betas from pickle file: ", rand_dir)
        # rand_data = pd.read_pickle(rand_dir)
        with open(rand_dir, "rb") as input_file:
            rand_data = pickle.load(input_file)

        # Add rand_data to the right of dataframe
        merged_data = pd.concat([data, rand_data], axis=1)
        # Remove duplicate columns
        # Randomly sampled voxels can include whole brain except for V1-V3 (double ups in HVC are possible)
        cleaned_merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()].copy()
        # Using list(df) to get the column headers as a list
        column_names = list(cleaned_merged_data.columns)

        # Scale whole dataset?
        preprocess = True
        if preprocess:
            # Add preprocessing
            data = preprocessing.scale(cleaned_merged_data)
            # bring back column header values
            data = pd.DataFrame(data, columns=column_names)
            # read index rows starting from 1
            data.index = np.arange(1, len(data) + 1)

        if train:
            image_type = 'train/'
            FILE_NAME_OUTPUT = '/images/' + image_type  # Defines the path in the dictionary outputs
            fmri_image_dataset = []

            if not single_pres: # ROI selection is only for max voxels
                for key, array in all_arrays.items():
                    fmri_image_dataset = []
                    # Select voxels from fmri dataframe
                    # Grab columns (voxel_IDs) for each ROI array
                    fmri_voxels = data.loc[:, array]

                    # if key == "V1_to_V3_n_rand":
                    #     fmri_voxels = pd.concat([fmri_voxels, rand_data], axis=1)

                    # cut down fmri dataframe to training trials
                    fmri = fmri_voxels.loc[train_array]

                    # raise Exception('check voxels and array')

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
                    output_path = os.path.join(save_path, image_type, pres, key)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    with open(os.path.join(output_path, sub + '_NSD_' + pres + '_train.pickle'),
                              'wb') as f:
                        # e.g., /Subj_01_NSD_max_train.pickle OR /Subj_01_NSD_single_pres_train.pickle
                        pickle.dump(fmri_image_dataset, f)

            if single_pres:
                fmri_image_dataset = []
                # Grab columns (voxel_IDs) just for VC (excluding rand sampled)
                fmri_voxels = data.loc[:, VC_array]
                # cut down fmri dataframe to training trials
                fmri = fmri_voxels.loc[train_array]
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

            if not single_pres: # ROI selection is only for max voxels
                for key, array in all_arrays.items():
                    fmri_image_dataset = []
                    # Select voxels from fmri dataframe
                    # Grab columns (voxel_IDs) for each ROI array
                    fmri_voxels = data.loc[:, array]

                    # if key == "V1_to_V3_n_rand":
                    #     fmri_voxels = pd.concat([fmri_voxels, rand_data], axis=1)

                    # cut down fmri dataframe to training trials
                    fmri = fmri_voxels.loc[test_array]

                    # raise Exception('check voxels and array')

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
                    output_path = os.path.join(save_path, image_type, pres, key)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    with open(os.path.join(output_path, sub + '_NSD_' + pres + '_valid.pickle'),
                              'wb') as f:
                        # e.g., /Subj_01_NSD_max_valid.pickle OR /Subj_01_NSD_single_pres_valid.pickle
                        pickle.dump(fmri_image_dataset, f)

            if single_pres: # ROI selection is only for max voxels
                fmri_image_dataset = []
                # Grab columns (voxel_IDs) just for VC (excluding rand sampled)
                fmri_voxels = data.loc[:, VC_array]
                # cut down fmri dataframe to test trials
                fmri = fmri_voxels.loc[test_array]
                # raise Exception('check voxels and array')

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


def nsd_data_dict_prep_3mm(data_dir, image_list_dir, subj_list, vox_res, data_type, save_path, norm, single_pres=False):
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
            # We don't use this for 3mm
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
        pickle_dir = os.path.join(data_dir, "subj_0{}_{}_concat_trial_fmri.pickle".format(s,norm))
        print("Reading betas from pickle file: ",pickle_dir)
        data = pd.read_pickle(pickle_dir)

        # Using list(df) to get the column headers as a list
        column_names = list(data.columns)

        # Scale whole dataset?
        preprocess = True
        if preprocess:
            # Add preprocessing
            data = preprocessing.scale(data)
            # bring back column header values
            data = pd.DataFrame(data, columns=column_names)
            # read index rows starting from 1
            data.index = np.arange(1, len(data) + 1)

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


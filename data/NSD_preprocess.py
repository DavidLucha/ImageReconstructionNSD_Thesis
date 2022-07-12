# Dependencies
import pandas as pd
import os.path
import os
import numpy as np
import h5py
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
from NSD_process_utils import NSDProcess, nsd_data_dict_prep

"""
Training/Test Split Image Extraction
Pulling images from hdf5 format to png in separate folders per subject
Change the output data and all that
"""
img_extract = False

if img_extract:
    # try:
    subj = ["Subj_01", "Subj_02", "Subj_03", "Subj_04", "Subj_05", "Subj_06", "Subj_07", "Subj_08"]
    # For test | Use above when ready
    # subj = ["Subj_01"]
    # Path for trial list xlsx
    trial_path = "D:/NSD/trial_information"
    for subj in subj:
        print(subj)
        trial_list = pd.read_csv(os.path.join(trial_path, subj + "_Trial_Lists_Sorted.csv"), sep=",", dtype=int)
        # Gets dset with just first presentation | Good for study 2
        single_presentation_list = trial_list[trial_list['Nth_Pres'] == 1]
        # Get dset with only shared or not
        train_list = trial_list[trial_list['Shared'] == 0]
        test_list = trial_list[trial_list['Shared'] == 1]
        # Pull array
        # train_array = train_list['Trial_No'].to_numpy()
        # test_array = test_list['Trial_No'].to_numpy()
        test_array_uniq = test_list.Image_Idx.unique()
        train_array_uniq = train_list.Image_Idx.unique()
        print(len(train_array_uniq))

        # Download images?
        # Downloads shared images
        # Note: change the output path in NSDProcess
        shared_download = False
        if subj == "Subj_01":
            if shared_download:
                # Test loading images and saving as PNG
                # Set root to NSD folder
                nsd = NSDProcess("D:", "shared")
                # 0-indexed (0-72999)
                nsd.read_images(test_array_uniq, show=False, download=True)

        # Downloads only unique images to one full folder
        download = False
        if download:
            # Test loading images and saving as PNG
            # Set root to NSD folder
            nsd = NSDProcess("D:", subj)
            # 0-indexed (0-72999)
            nsd.read_images(train_array_uniq, show=False, download=True)

"""
Image Processing
"""
# Test loading images and saving as PNG
# Set root to NSD folder
# nsd = NSDProcess("D:", "test")
# 0-indexed (0-72999)
# img = nsd.read_images([0, 2950, 2951, 72999], show=True, download=True)


"""
fMRI Y-Trial Number and X-Voxel ID Formatting
"""
# Test grabbing voxel_id, and voxel values, transpose
# Such that, voxels_id are columns, and trials are rows 0-749

concat = False

if concat:
    data_dir = "D:/NSD/nsddata_betas/ppdata/"
    session_count = [0, 37, 37, 29, 27, 37, 29, 37, 27]
    out_path = "D:/Lucha_Data/datasets/NSD/"
    vox_res = "1.8mm"

    # Full for session range:
    #         for session in range(1, session_count[subject]+1):
    for subject in range(1, 9):
    # for subject in [4]: # TODO: Remove this and uncomment above
        print(subject)
        # subject_fmri = []
        subject_fmri_dict = []
        print("Concatenating data for Subject %d" % subject)
        sub_dir = os.path.join(data_dir, f"subj0{subject}", "func1pt8mm", "betas_fithrf_GLMdenoise_RR")
        print("Data folder at:", sub_dir)
        subject_full_df = pd.DataFrame()
        print(session_count[subject])
        for session in range(1, session_count[subject] + 1):
        # for session in [2, 14]:
            # print(session)
            # Change directory format for below and above 10
            # i.e., Sessions under 10 have 0x format
            # this can be down with leading format print("{:02d}".format(number))
            betas_dir = os.path.join(sub_dir, "subj_0{0}_masked_betas_session{1:02d}.txt".format(subject, session))
            betas_file = "subj_0{0}_masked_betas_session{1:02d}.txt".format(subject, session)
            print("Adding data from {}".format(betas_file))
            # Reads betas
            betas = pd.read_csv(betas_dir, sep=" ", header=None)

            # Save the column of voxel ids for norming
            vox_id = betas[0].to_numpy()

            # Makes voxel_ids the index
            betas = betas.set_index([0], drop=True)

            # Drops X, Y, Z columns, transposes dataframe
            betas_b = betas.drop([1, 2, 3], axis=1).T
            if session == 1:
                # Start index from 1 (really only important for first session)
                subject_full_df = betas_b
            elif session != 1:
                subject_full_df = subject_full_df.append(betas_b, ignore_index=True)

        raw_out = False
        save = True

        if raw_out:
            subject_full_df.index = np.arange(1, len(subject_full_df) + 1)

            if save:
                # Save concatenated %age signal change betas per voxel per trial as pickle
                raw_pickle_out = os.path.join(out_path, vox_res, "raw_concat_pickle/")
                if not os.path.exists(raw_pickle_out):
                    os.makedirs(raw_pickle_out)
                subject_full_df.to_pickle(os.path.join(raw_pickle_out, "subj_0{0}_raw_concat_trial_fmri.pickle".format(subject)))

        # Normalise, but could wait until I reload
        # Gets rid of index ordering and column names
        # The bottom reformats the columns and indices to work properly
        norm = True
        if norm:
            normed_subject_full_df = preprocessing.scale(subject_full_df)
            normed_subject_full_df_titled = pd.DataFrame(normed_subject_full_df, columns = vox_id)
            normed_subject_full_df_titled.index = np.arange(1, len(subject_full_df) + 1)

            if save:
                # Save concatenated normalised %age signal change betas per voxel per trial as pickle
                normed_pickle_out = os.path.join(out_path, vox_res, "normed_concat_pickle/")
                if not os.path.exists(normed_pickle_out):
                    os.makedirs(normed_pickle_out)
                normed_subject_full_df_titled.to_pickle(os.path.join(normed_pickle_out, "subj_0{0}_normed_concat_trial_fmri.pickle".format(subject)))

        # raise Exception("Check normed vs non-normed")

        print('Subject {} complete.'.format(subject))
    print("Finished.")

"""
Creating trainable datasets (list of dicts)
    Per voxel resolution:
    Each subject has:
    - Full train split (27k)
    - Full test split (3k)
    - Maximum single presentation (need to account for max trials per subject)
        - Could just do 7500 now and come back later
    - 4000 single presentation
    - 1200 single presentation
    
    Params:
    vox_res
    data_dir - full pickles
    image_list_dir - main csvs from trial image
    norm = True
    subj = [] - array of subjects getting processed
    data_type = train/test both?
"""

dict_prep = True

if dict_prep:
    # Hello there. - Obi Wan
    subj_list = [1, 2, 3, 4, 5, 6, 7, 8]  # 1, 2, 3, 4, 5, 6, 7, 8
    vox_res = "1.8mm"
    image_list_dir = "D:/NSD/trial_information"
    data_type = "both"
    save_path = os.path.join("D:/Lucha_Data/datasets/NSD", vox_res)
    # Can change to raw %age change below if you want
    data_dir = os.path.join(save_path, "normed_concat_pickle")

    # Run data
    # All data
    nsd_data_dict_prep(data_dir, image_list_dir, subj_list, vox_res, data_type, save_path, False)
    # Using first presentation only
    nsd_data_dict_prep(data_dir, image_list_dir, subj_list, vox_res, data_type, save_path, True)

# Dependencies
import pandas as pd
import os.path
import os
import numpy as np
import h5py
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image
from NSD_process_utils import NSDProcess

"""
Training/Test Split
"""
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
    shared_download = True
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
# except Exception:
#     pass

"""
Image Processing
"""
# Test loading images and saving as PNG
# Set root to NSD folder
# nsd = NSDProcess("D:", "test")
# 0-indexed (0-72999)
# img = nsd.read_images([0, 2950, 2951, 72999], show=True, download=True)


"""
fMRI Concatenation
"""
# Test grabbing voxel_id, and voxel values, transpose
# Such that, voxels_id are columns, and trials are rows 0-749
data_dir = "D:/NSD/nsddata_betas/ppdata/"
session_count = [0, 37, 37, 29, 27, 37, 29, 37, 37]

# Full for session range:
#         for session in range(1, session_count[subject]+1):

concat = False

if concat:
    for subject in range(1, 9):
        print(subject)
        # subject_fmri = []
        subject_fmri_dict = []
        if subject == 2:
            print("Testing data with Subject %d" % subject)
            sub_dir = os.path.join(data_dir, f"subj0{subject}", "func1pt8mm", "betas_fithrf_GLMdenoise_RR")
            print(sub_dir)
            subject_full_df = pd.DataFrame()
            for session in range(1, 3):
                print(session)
                # Change directory format for below and above 10
                # i.e., Sessions under 10 have 0x format
                # this can be down with leading format print("{:02d}".format(number))
                if session <= 9:
                    betas_dir = os.path.join(sub_dir, "subj_0{0}_masked_betas_session0{1}.txt".format(subject, session))
                else:
                    betas_dir = os.path.join(sub_dir, "subj_0{0}_masked_betas_session{1}.txt".format(subject, session))
                print(betas_dir)
                # Reads betas
                betas = pd.read_csv(betas_dir, sep=" ", header=None)
                # Makes voxel_ids the index
                betas = betas.set_index([0], drop=True)
                # Drops X, Y, Z columns, transposes dataframe
                betas_b = betas.drop([1, 2, 3], axis=1).T
                if session == 1:
                    # Start index from 1 (really only important for first session)
                    subject_full_df = betas_b
                elif session != 1:
                    subject_full_df = subject_full_df.append(betas_b, ignore_index=True)
            subject_full_df.index = np.arange(1, len(subject_full_df) + 1)
            normed_subject_full_df = preprocessing.scale(subject_full_df)
            # Norming works, but it gets rid of the voxel_id header and 1-indexed index.

            # Normalize per
            # normed_fmri = preprocessing.scale(fmri)  # TODO: This






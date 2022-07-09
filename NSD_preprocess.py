# Dependencies
import pandas as pd
import os.path
import os
import numpy as np
from sklearn import preprocessing

# Test grabbing voxel_id, and voxel values, transpose
# Such that, voxels_id are columns, and trials are rows 0-749
data_dir = "D:/NSD/nsddata_betas/ppdata/"
session_count = [0, 37, 37, 29, 27, 37, 29, 37, 37]

# Full for session range:
#         for session in range(1, session_count[subject]+1):

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






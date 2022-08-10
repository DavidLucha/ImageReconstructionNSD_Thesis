# Dependencies
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
import pickle
import random
from NSD_process_utils import NSDProcess, nsd_data_dict_prep, nsd_data_dict_prep_3mm

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
    out_path = "D:/Honours/nsd_pickles/"
    # ------------- CHANGE THESE TWO -------------- #
    vox_res = "3mm"
    data_type = "3mm"  #  "1.8mm" is main 1.8mm, "3mm" is 3mm and "rand" is for the random selection of voxels
    # -------------------- END -------------------- #

    # Full for session range:
    for subject in range(1, 9):
    # for subject in [1]: # Remove this and uncomment above
        print(subject)
        # subject_fmri = []
        subject_fmri_dict = []
        print("Concatenating data for Subject %d" % subject)
        sub_dir = os.path.join(data_dir, f"subj0{subject}", "func1pt8mm", "betas_fithrf_GLMdenoise_RR")
        print("Data folder at:", sub_dir)
        subject_full_df = pd.DataFrame()
        print(session_count[subject])
        for session in range(1, session_count[subject] + 1):
        # for session in [2, 14]:  # Test Line - Uncomment
            # print(session)
            # Change directory format for below and above 10
            # i.e., Sessions under 10 have 0x format
            # this can be down with leading format print("{:02d}".format(number))

            """One way we could do it is, we need full voxels for study 1 and study 2. So that's fine.
            Everything as normal. But then for the final study I'm trying to decide whether it's better to do either
            of following.
            1. Make the concat as per normal, then load in the ROI label doc, grab voxel IDs where ROI == ROI 
                and then use that to filter out columns.
            2. Bring the ROI labels in now, make different cocnat pickles that contain all trials, and then one for 
                each dataset. Eg. subj_01_normed_V1_pickle and then have to grab training and trial subject from
                each of those. I think 1 is better. Too much double handling here."""
            if data_type == "3mm":
                betas_dir = os.path.join(sub_dir,
                                         "subj_0{0}_masked_betas_session{1:02d}_3mm.txt".format(subject, session))
            elif data_type == "rand":
                betas_dir = os.path.join(sub_dir,
                                         "subj_0{0}_masked_betas_session{1:02d}_rand_samp.txt".format(subject, session))
            else:
                betas_dir = os.path.join(sub_dir, "subj_0{0}_masked_betas_session{1:02d}.txt".format(subject, session))
            # betas_file = "subj_0{0}_masked_betas_session{1:02d}.txt".format(subject, session)
            print("Adding data from {}".format(betas_dir))
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

        raw_out = True
        save = True

        if raw_out:
            subject_full_df.index = np.arange(1, len(subject_full_df) + 1)

            if save:
                # Save concatenated %age signal change betas per voxel per trial as pickle
                raw_pickle_out = os.path.join(out_path, vox_res, "raw_concat_pickle/")
                if not os.path.exists(raw_pickle_out):
                    os.makedirs(raw_pickle_out)
                if data_type == "3mm":
                    pickle_out = os.path.join(raw_pickle_out,
                                                     "subj_0{0}_raw_concat_trial_fmri_3mm.pickle".format(subject))
                if data_type == "rand":
                    pickle_out = os.path.join(raw_pickle_out,
                                                     "subj_0{0}_raw_concat_trial_fmri_rand.pickle".format(subject))
                else:
                    pickle_out = os.path.join(raw_pickle_out,
                                                     "subj_0{0}_raw_concat_trial_fmri.pickle".format(subject))

                subject_full_df.to_pickle(pickle_out)

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
                if data_type == "3mm":
                    norm_pickle_out = os.path.join(normed_pickle_out,
                                                     "subj_0{0}_normed_concat_trial_fmri_3mm.pickle".format(subject))
                if data_type == "rand":
                    norm_pickle_out = os.path.join(normed_pickle_out,
                                                     "subj_0{0}_normed_concat_trial_fmri_rand.pickle".format(subject))
                else:
                    norm_pickle_out = os.path.join(normed_pickle_out,
                                                     "subj_0{0}_normed_concat_trial_fmri.pickle".format(subject))

                normed_subject_full_df_titled.to_pickle(norm_pickle_out)

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

dict_prep = False

if dict_prep:
    # Hello there. - Obi Wan
    subj_list = [1, 2, 3, 4, 5, 6, 7, 8]  # 1, 2, 3, 4, 5, 6, 7, 8

    # ------ TODO: CHANGE THIS -------- #
    vox_res = "3mm"
    image_list_dir = "D:/NSD/trial_information"
    data_type = "both"
    # make pickles from normed or raw
    norm = "raw"
    save_path = os.path.join("D:/Lucha_Data/datasets/NSD", vox_res)
    # Can change to raw %age change below if you want
    data_dir = os.path.join("D:/Honours/nsd_pickles", vox_res, norm + "_concat_pickle")
    # -------------- END ---------------#

    # Runs the 3mm process
    mm_3 = True
    if mm_3:
        # Gets max, all pres - not currently planned use
        nsd_data_dict_prep_3mm(data_dir, image_list_dir, subj_list, vox_res, data_type, save_path, norm, False)
        # Using first presentation only
        nsd_data_dict_prep_3mm(data_dir, image_list_dir, subj_list, vox_res, data_type, save_path, norm, True)

    # Runs the 1.8mm process
    mm_1pt8 = False
    if mm_1pt8:
        # All data
        nsd_data_dict_prep(data_dir, image_list_dir, subj_list, vox_res, data_type, save_path, norm, False)
        # Using first presentation only
        nsd_data_dict_prep(data_dir, image_list_dir, subj_list, vox_res, data_type, save_path, norm, True)




"""
Testing random selection of dataset

Only pulls arrays, can use this to save data sets or just to cut it down from the training stages
First gets 7500 from each of the possible single presentation set sizes [8859, 8188, 7907]
Then grabs 4000 from that 7500 and then 1200 from 4000. Making sure we're controlling as much as possible.
Saving pickles for reproducibility. The same random selection is made for each participants (for 4000, 1200). 
Initial 7500 pick varies depending on the available training set size of the participant.
"""

# Get random arrays and save to pickles for reproducibility
get_array = False
if get_array:
    random.seed(10000)  # was 76
    # So we will use an array and pickle load this
    data_avail = [8859, 8188, 7907]  # 37, 29, 27
    # data_size = [1200, 4000, 7500, 8000]
    data_size = [7500, 4000, 1200]
    # data_avail[1:]
    output_path = "D:/Lucha_Data/datasets/NSD/misc/dataset_randomness"
    save = True

    for n in data_avail:
        # Gets the 8000 from either avail
        # Or just do it in the code - to save space
        rnd_array = np.random.randint(0, n - 1, data_size[0])
        print(rnd_array, len(rnd_array), min(rnd_array), max(rnd_array))
        if save:
            with open(os.path.join(output_path, '7500_NSD_single_pres_{}_train_array.pickle'.format(n)), 'wb') as f:
                pickle.dump(rnd_array, f)

    med_array = np.random.randint(0, data_size[0] - 1, data_size[1])
    print(med_array, len(med_array), min(med_array), max(med_array))
    small_array = np.random.randint(0, data_size[1] - 1, data_size[2])
    print(small_array, len(small_array), min(small_array), max(small_array))

    if save:
        with open(os.path.join(output_path, '4000_NSD_single_pres_train_array.pickle'), 'wb') as f:
            pickle.dump(med_array, f)

        with open(os.path.join(output_path, '1200_NSD_single_pres_train_array.pickle'), 'wb') as f:
            pickle.dump(small_array, f)


# var_training set grabs the variable-sized training sets based on the set random arrays from above
var_training_set = True

if var_training_set:
    input_path = "D:/Lucha_Data/datasets/NSD/"
    array_path = "misc/dataset_randomness"
    voxel_res = "3mm"
    fmri_path = os.path.join(input_path, voxel_res + "/train/single_pres")
    subj = [1, 2, 3, 4, 5, 6, 7, 8]

    for s in subj:
        data_path = os.path.join(fmri_path, "Subj_0{}_NSD_single_pres_train.pickle".format(s))
        # Load data
        with open(data_path, "rb") as input_file:
            data = pickle.load(input_file)

        # Load small and medium array
        small_array_path = os.path.join(input_path, array_path, "1200_NSD_single_pres_train_array.pickle")
        med_array_path = os.path.join(input_path, array_path, "4000_NSD_single_pres_train_array.pickle")

        with open(small_array_path, "rb") as input_file:
            small_array = pickle.load(input_file)

        with open(med_array_path, "rb") as input_file:
            med_array = pickle.load(input_file)

        if s in [1, 2, 5, 7]:
            print("I, {}, am large (37 session)".format(s))
            big_array_path = os.path.join(input_path, array_path, "7500_NSD_single_pres_8859_train_array.pickle")

        elif s in [3, 6]:
            print("I, {}, am less large (29 sessions)".format(s))
            big_array_path = os.path.join(input_path, array_path, "7500_NSD_single_pres_8188_train_array.pickle")
        elif s in [4, 8]:
            print("I, {}, am even less larger (27 sessions)".format(s))
            big_array_path = os.path.join(input_path, array_path, "7500_NSD_single_pres_7907_train_array.pickle")

        with open(big_array_path, "rb") as input_file:
            big_array = pickle.load(input_file)

        large_set_data = [data[i] for i in big_array]
        med_set_data = [large_set_data[i] for i in med_array]
        small_set_data = [med_set_data[i] for i in small_array]

        # list = [1,3,5]
        # new_list = [data[i] for i in list]

        sizes = [1200, 4000, 7500]
        for size in sizes:
            # Save pickle
            output_path = os.path.join(fmri_path, str(size))
            if not os.path.exists(output_path):
                os.makedirs(output_path)

            if size == 1200:
                out_data = small_set_data
            elif size == 4000:
                out_data = med_set_data
            else:
                out_data = large_set_data

            print('Saving Subject {} data ({}) to {}...'.format(s, size, output_path))

            with open(os.path.join(output_path, 'Subj_0{}_{}_NSD_single_pres_train.pickle'.format(s, size)), 'wb') as f:
                # e.g., /Subj_01_NSD_max_train.pickle OR /Subj_01_NSD_single_pres_train.pickle
                pickle.dump(out_data, f)


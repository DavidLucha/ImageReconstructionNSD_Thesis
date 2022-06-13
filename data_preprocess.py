"""
Run this script to data pre-process
Concats fmri and image file
"""
import os.path
import torch
import pickle
from bdpy import BData
import csv
import pandas as pd
import shutil
import random
import os
from PIL import Image
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
from utils_2 import GreyToColor


def ren_data_prep(data_dir, image_list_dir, norm=True):
    """
    Loads H5 data, pulls voxels from ROI and image index. (TRAINING)
    Combines these into a list of dictionaries formatted as:
    [{'fmri': VOXEL_DATA_ARRAY, 'image': IMAGE_PATH}, ...
    {'fmri': VOXEL_DATA_ARRAY, 'image': IMAGE_PATH}]

    Saves these training sets per subject

    :param data_dir: should be:
        "D:/Honours/Object Decoding Dataset/7387130/"
    :param image_list_dir: should lead to numbered list of image names:
        "D:/Honours/Object Decoding Dataset/images_passwd/images/image_training_id_nmd.csv"

    Note: Works for the GOD data, unsure if this will work for NSD

    """
    DATA_TYPE = 'train' # train or valid
    DATASET = 'GOD'
    # TRAIN_IMG_PATH = "D:/Honours/Object Decoding Dataset/images_passwd/images/training/"
    SAVE_PATH = "D:/Lucha_Data/datasets/"
    if DATA_TYPE == 'train':
        image_type = 'train/'
    if DATA_TYPE == 'valid':
        image_type = 'valid/'
    FILE_NAME_OUTPUT = '/images/' + image_type # Defines the path in the dictionary outputs

    norm = True # TODO: Can remove

    with open(image_list_dir, 'r') as f:
        image_list = list(csv.reader(f, delimiter=","))

    image_list_dir = image_list_dir

    image_list = pd.DataFrame(image_list)
    image_list.columns = ['Image_ID', 'File_Name']
    image_list = image_list.astype({'Image_ID': 'int32'})
    image_list = image_list.astype({'File_Name': 'string'})

    subjects = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5']
    training_data = []

    for sub in subjects:
        fmri_image_dataset = []
        data = BData(data_dir + sub + '.h5')
        # data = BData(data_dir + 'Subject1.h5') # TODO: COMMENT THIS OUT (Testing only)
        if DATA_TYPE == 'train':
            fmri = data.select('ROI_VC')[:1200]
            image_idx = data.select('image_index')[:1200]
        if DATA_TYPE == 'valid':
            fmri = data.select('ROI_VC')[1200:2950] # Corresponds to trials in test phase (50 images x 35 presentations)
            image_idx = data.select('image_index')[1200:2950]
        image_idx = pd.DataFrame(image_idx, dtype='int')
        image_idx.columns = ['Image_ID']
        image_names= image_idx.merge(image_list, how='left', on='Image_ID')
        image_names['File_Name'] = FILE_NAME_OUTPUT + image_names['File_Name'].astype(str)
        image_names = image_names.to_numpy()
        image_names = image_names[:,1]

        # Normalise fMRI
        if norm:
            normed_fmri = preprocessing.scale(fmri)
            # fmri_pd = pd.DataFrame(normed_fmri, dtype='float')
            fmri_pd = normed_fmri
            norm_status = '_normed'

        if not norm:
            # fmri_pd = pd.DataFrame(fmri, dtype='float')
            norm_status = '_raw'
            fmri_pd = fmri

        # Plot distribution
        # fig, ax = plt.subplots()
        # test = np.random.randint(0,1200,8)
        vis = False

        if vis:
            test = [167]
            data_plot = pd.DataFrame(fmri_pd, dtype='float')
            sub_data = data_plot[0:4].T
            # Checking proportion of raw data about abs(3)
            # sub_data_abs = abs(sub_data)
            # counts = sub_data[sub_data > 3.0].count()
            # sub_data.count()
            # prop = counts/(sub_data.count())
            sub_data.plot.box()
            plt.show()

            for x in test:
                data_plot_sub = data_plot.iloc[x]
                hist = data_plot_sub.hist(bins=20)
                plt.show()

            # full_hist = data_plot.hist(bins=10)
            plt.show()
            # plt.close()


        for idx, (vox, pix) in enumerate(zip(fmri_pd, image_names)):
            fmri_image_dataset.append({'fmri': vox, 'image': pix})

        # print(pd.DataFrame.mean(fmri_pd))
        # print(fmri_pd.stack().std())
        # print(fmri_pd[8].mean())

        # Adds the subject to full training list
        training_data.extend(fmri_image_dataset)

        with open(os.path.join(SAVE_PATH, DATASET, DATASET + '_' + sub + '_' + DATA_TYPE + norm_status + '.pickle'), 'wb') as f:
            pickle.dump(fmri_image_dataset, f)

    # Append all lists for full training dataset
    with open(os.path.join(SAVE_PATH, DATASET, DATASET + '_' + 'all_subjects_' + DATA_TYPE + norm_status + '.pickle'), 'wb') as f:
        pickle.dump(training_data, f)


def ren_data_prep_average(data_dir, image_list_dir, norm = True, avg = True):
    """
    Loads H5 data, pulls voxels from ROI and image index. (TEST)
    Averages by each image presentation
    Combines these into a list of dictionaries formatted as:
    [{'fmri': VOXEL_DATA_ARRAY, 'image': IMAGE_PATH}, ...
    {'fmri': VOXEL_DATA_ARRAY, 'image': IMAGE_PATH}]

    Saves these training sets per subject

    :param data_dir: should be:
        "D:/Honours/Object Decoding Dataset/7387130/"
    :param image_list_dir: should lead to numbered list of image names:
        "D:/Honours/Object Decoding Dataset/images_passwd/images/image_training_id_nmd.csv"

    Note: Works for the GOD data, unsure if this will work for NSD

    """
    DATA_TYPE = 'valid' # train or valid
    DATASET = 'GOD'
    # TRAIN_IMG_PATH = "D:/Honours/Object Decoding Dataset/images_passwd/images/training/"
    SAVE_PATH = "D:/Lucha_Data/datasets/"
    if DATA_TYPE == 'train':
        image_type = 'train/'
    if DATA_TYPE == 'valid':
        image_type = 'valid/'

    FILE_NAME_OUTPUT = '/images/' + image_type  # Defines the path in the dictionary outputs

    with open(image_list_dir, 'r') as f:
        image_list = list(csv.reader(f, delimiter=","))

    image_list_dir = image_list_dir

    image_list = pd.DataFrame(image_list)
    image_list.columns = ['Image_ID', 'File_Name']
    image_list = image_list.astype({'Image_ID': 'int32'})
    image_list = image_list.astype({'File_Name': 'string'})

    subjects = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5']
    training_data = []

    for sub in subjects:
        fmri_image_dataset = []
        data = BData(data_dir + sub + '.h5')
        # data = BData(data_dir + 'Subject1.h5') # TODO: Get rid of this | Testing only
        if DATA_TYPE == 'train':
            fmri = data.select('ROI_VC')[:1200]
            image_idx = data.select('image_index')[:1200]
        if DATA_TYPE == 'valid':
            fmri = data.select('ROI_VC')[1200:2950] # Corresponds to trials in test phase (50 images x 35 presentations)
            image_idx = data.select('image_index')[1200:2950]

        if avg:
            image_idx = pd.DataFrame(image_idx, dtype='int')
            image_idx.columns = ['Image_ID']
            image_names= image_idx.merge(image_list, how='left', on='Image_ID') # Works for testing
            file_names = image_names.loc[:,'File_Name']
            fmri_df = pd.DataFrame(fmri, dtype='float')
            fmri_image = pd.concat([file_names, fmri_df], axis=1)
            # fmri_image.to_csv(r'fmri_image_test.csv', index=False, header=True) # To test calcs, all good.
            avg_fmri = fmri_image.groupby('File_Name').mean()
            avg_status = '_avg'

            # Normalise data
            # TODO: Check of Horikawa already normed
                # TODO: I'd plot all the data as a histogram plot. So, if they are between roughly -3 and 3, it is already good to go!
                # Seems to be evidence that they have
            if norm:
                normed_fmri = preprocessing.scale(avg_fmri)
                # fmri_pd = pd.DataFrame(normed_fmri, dtype='float')
                fmri_pd = normed_fmri
                norm_status = '_normed'
                # TEST MEAN AND SD
                # print(pd.DataFrame.mean(fmri_pd))
                # print(fmri_pd.stack().std())
                # print(fmri_pd[8].mean())

            if not norm:
                # fmri_pd = pd.DataFrame(avg_fmri, dtype='float')
                fmri_pd = fmri
                norm_status = '_raw'

            avg_row_names = pd.DataFrame(avg_fmri)
            avg_row_names = avg_row_names.index.values

            image_names = FILE_NAME_OUTPUT + avg_row_names

        if not avg:
            image_idx = pd.DataFrame(image_idx, dtype='int')
            image_idx.columns = ['Image_ID']
            image_names = image_idx.merge(image_list, how='left', on='Image_ID')  # Works for testing
            image_names['File_Name'] = FILE_NAME_OUTPUT + image_names['File_Name'].astype(str)
            image_names = image_names.to_numpy()
            image_names = image_names[:, 1]
            avg_status = ''

            # Normalise fMRI
            if norm:
                normed_fmri = preprocessing.scale(fmri)
                # fmri_pd = pd.DataFrame(normed_fmri, dtype='float')
                fmri_pd = normed_fmri
                norm_status = '_normed'

            if not norm:
                # fmri_pd = pd.DataFrame(fmri, dtype='float')
                fmri_pd = fmri
                norm_status = '_raw'

        for idx, (vox, pix) in enumerate(zip(fmri_pd, image_names)):
            fmri_image_dataset.append({'fmri': vox, 'image': pix})

        # Adds the subject to full training list
        training_data.extend(fmri_image_dataset)

        with open(os.path.join(SAVE_PATH, DATASET, DATASET + '_' + sub + '_' + DATA_TYPE + norm_status + avg_status + '.pickle'), 'wb') as f:
            pickle.dump(fmri_image_dataset, f)

    # Append all lists for full training dataset
    with open(os.path.join(SAVE_PATH, DATASET, DATASET + '_' + 'all_subjects_' + DATA_TYPE + norm_status + avg_status + '.pickle'), 'wb') as f:
        pickle.dump(training_data, f)


if __name__ == "__main__":

    """
    Script for taking GOD data voxels, and turning them into list of dictionaries
    """
    fmri_image_comp = False

    if fmri_image_comp:
        training = True # CHANGE THIS DEPENDING ON WHAT YOU'RE DOING

        data_dir = "D:/Honours/Object Decoding Dataset/7387130/"  # Where h5 files are stored

        if training:
            image_list_dir = "D:/Honours/Object Decoding Dataset/images_passwd/images/image_training_id_nmd.csv"
            # subj_pick_PATH = "D:/Honours/Object Decoding Dataset/7387130/Subject Training Pickles/"
            ren_data_prep(data_dir, image_list_dir, norm=True)

        # training = False

        if not training:
            image_list_dir = "D:/Honours/Object Decoding Dataset/images_passwd/images/image_test_id_nmd.csv"
            ren_data_prep_average(data_dir, image_list_dir, norm=True, avg=True)  # Avg Data
            ren_data_prep_average(data_dir, image_list_dir, norm=True, avg=False)

            # Concat all pickles
        # training_data = concatenate_bold_data(subj_pick_PATH)

        # with open(os.path.join(SAVE_PATH, 'bold_train', 'bold_train_norm.pickle'), 'wb') as f:
        #     pickle.dump(train_data, f)
        # Script End


    # Grab random selection of 10k images from validation set (GOD ImageNet 2011)
    val_path = 'D:/Honours/val/'
    square_dir = 'D:/Lucha_Data/datasets/GOD/images/sq_pretrain/'
    pretrain_dir = 'D:/Lucha_Data/datasets/GOD/images/pretrain_all/'

    """
    Randomly select images from validation if x and y > 100
    Also filter out if aspect ratio is less than 0.67 or above 1.4
    
    NOTE: Some train VAEs on 60000 for 20 epochs = 1.2mil.
    @ 20k samples, we do 75 epochs for 1.5mil
    
    """
    rnd_sample = False

    if rnd_sample:
        sample = random.sample(os.listdir(val_path), 49999)
        sample_count = 0
        for fname in sample:
            srcpath = os.path.join(val_path, fname)
            with Image.open(srcpath) as im:
                x, y = im.size
                ratio = x / y
            if x > 100 and y > 100 and 0.67 < ratio < 1.4:
                shutil.copy(srcpath, pretrain_dir)
                sample_count += 1

    # print(sample_count)
                # if sample_count == 25000:
                #     break

    """
     Delete random batch of images to get from 31042 to 30000
    """
    delete_rnd = True

    if delete_rnd:
        sample = random.sample(os.listdir(pretrain_dir), 1042)
        for fname in sample:
            filepath = os.path.join(pretrain_dir, fname)
            os.remove(filepath)

    """
    Copy all square images from validation to sqr_pretrain
    """
    copy_fn = False # Do copy function?

    if copy_fn:
        copy_count = 0
        for fname in os.listdir(val_path):
            filepath = os.path.join(val_path, fname)
            with Image.open(filepath) as im:
                x, y = im.size
            if x == y and x >= 100 and y >= 100:
                shutil.copy(filepath, square_dir)
                copy_count += 1
            # if copy_count == 10000:
            #     break

    """
    Delete all images with size less than 100x
    """
    delete_small = False

    if delete_small:
        for fname in os.listdir(pretrain_dir):
            filepath = os.path.join(pretrain_dir, fname)
            with Image.open(filepath) as im:
                x, y = im.size
            if x < 100 or y < 100:
                os.remove(filepath)

    """
    Read shape of images
    """
    read_shape = False

    if read_shape:
        path = 'D:/Lucha_Data/datasets/GOD/images/train/'
        transform = transforms.ToTensor()

        for fname in os.listdir(path):
            filepath = os.path.join(path, fname)
            count = 0
            with Image.open(filepath) as im:
                image_tens = transform(im)
                image_shape = image_tens.shape
                # print(image_shape)
            if not image_shape == [3, 500, 500]:
                print(fname, image_shape)
                count = count + 1
                # print(fname)
        print(count)

        # Test opening greyscale image post transform
        # transform = transforms.ToTensor()
        grey = GreyToColor(500)

        with Image.open('D:/Lucha_Data/datasets/GOD/images/train/n03512147_47076.JPEG') as image:
            image = transform(image)
            new_image = grey(image)
            print(type(new_image))
            print(new_image.shape)
            plt.imshow(new_image.permute(1, 2, 0))
            plt.show()

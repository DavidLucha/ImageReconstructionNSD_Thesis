"""
Run this script to data pre-process
Concats fmri and image file
"""
import os.path
import pickle
from bdpy import BData
import csv
import pandas as pd


def ren_data_prep(data_dir, image_list_dir):
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

    """

    TRAIN_IMG_PATH = "D:/Honours/Object Decoding Dataset/images_passwd/images/training/"
    SAVE_PATH = "D:/Honours/Object Decoding Dataset/7387130/Subject Training Pickles/"

    image_list_dir = image_list_dir
    with open(image_list_dir, 'r') as f:
        image_list = list(csv.reader(f, delimiter=","))

    image_list = pd.DataFrame(image_list)
    image_list.columns = ['Image_ID', 'File_Name']
    image_list = image_list.astype({'Image_ID': 'int32'})
    image_list = image_list.astype({'File_Name': 'string'})

    subjects = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5']
    training_data = []

    for sub in subjects:
        fmri_image_dataset = []
        data = BData(data_dir + sub + '.h5')
        fmri = data.select('ROI_VC')[:1200]
        image_idx = data.select('image_index')[:1200]
        image_idx = pd.DataFrame(image_idx, dtype='int')
        image_idx.columns = ['Image_ID']
        image_names = image_idx.merge(image_list, how='inner', on='Image_ID')
        image_names['File_Name'] = TRAIN_IMG_PATH + image_names['File_Name'].astype(str)
        image_names = image_names.to_numpy()
        image_names = image_names[:,1]

        for idx, (vox, pix) in enumerate(zip(fmri, image_names)):
            fmri_image_dataset.append({'fmri': vox, 'image': pix})

        # Adds the subject to full training list
        training_data.extend(fmri_image_dataset)

        with open(SAVE_PATH + 'fmri_image_' + sub + '.pickle', 'wb') as f:
            pickle.dump(fmri_image_dataset, f)

    # Append all lists for full training dataset
    with open(SAVE_PATH + 'fmri_image_' + 'all_subjects' + '.pickle', 'wb') as f:
        pickle.dump(training_data, f)


if __name__ == "__main__":

    data_path = "D:/Honours/Object Decoding Dataset/7387130/"
    image_list_path = "D:/Honours/Object Decoding Dataset/images_passwd/images/image_training_id_nmd.csv"
    subj_pick_PATH = "D:/Honours/Object Decoding Dataset/7387130/Subject Training Pickles/"

    # Save pickles for all subjects
    ren_dataset = ren_data_prep(data_path, image_list_path)

    # Concat all pickles
    # training_data = concatenate_bold_data(subj_pick_PATH)

    # with open(os.path.join(SAVE_PATH, 'bold_train', 'bold_train_norm.pickle'), 'wb') as f:
    #     pickle.dump(train_data, f)
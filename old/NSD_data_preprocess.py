import os
import os.path as op
import glob
import nibabel as nb
import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nilearn import plotting

import urllib.request, zipfile
from pycocotools.coco import COCO

from nsd_access.nsda import NSDAccess



nsda = NSDAccess('D:/NSD/')

# get the betas for a given subject, session and set of trials
betas = nsda.read_betas(subject='subj01',
                        session_index=1,
                        trial_index=[], # empty list as index means get all for this session
                        data_type='betas_fithrf_GLMdenoise_RR',
                        data_format='func1pt8mm')

# CURRENTLY USING THE NIFTI FILE BECAUSE NO MASK

print(betas.shape)
# betas.head()

############################

# just list the available atlases for a given subject/data format - returns a list of names
# will, for surfaces, mix both mapper results and atlas data.
atlases = nsda.list_atlases(subject='subj01', data_format='func1pt8mm');
# similar functionality should be created for mapper data when fsaverage mapper results exist.
atlases

atlas = 'prf-visualrois'
mmp1, atlas_mapping = nsda.read_atlas_results(subject='subj01',
                               atlas=atlas,
                               data_format='func1pt8mm')
############################

# get surface-based or volume-based atlas values and their mapping
# for now, demoing volume for lack of surface mapper results

atlas = 'HCP_MMP1'
mmp1, atlas_mapping = nsda.read_atlas_results(subject='subj01',
                               atlas=atlas,
                               data_format='func1pt8mm')
plt.hist(mmp1.ravel(),
         range=[0,int(np.max(mmp1))+1],
         bins=int(np.max(mmp1))+1,
         alpha=0.4,
         cumulative=True,
         label='both hemispheres');

# or, only get the right hemisphere
mmp1_rh, atlas_mapping_rh = nsda.read_atlas_results(subject='subj01',
                               atlas=f'rh.{atlas}',
                               data_format='func1pt8mm')
plt.hist(mmp1_rh.ravel(),
         range=[0,int(np.max(mmp1))+1],
         bins=int(np.max(mmp1))+1,
         alpha=0.4,
         color='r',
         cumulative=True,
         label='right hemisphere only');
plt.gca().set_xlabel('labels')
plt.gca().set_ylabel('# of vertices/voxels')
plt.legend()

plt.show()

# easy to make, for example, V1 mask:
v1_mask = (mmp1 == atlas_mapping['V1'])
# or rh V1 mask
v1_mask_rh = (mmp1_rh == atlas_mapping_rh['V1'])

# you could:
# print(atlas_mapping)

####################################
# Get betas with a mask

# get the betas for a given subject, session and set of trials
v1_betas = nsda.read_betas(subject='subj01',
                           session_index=1,
                           trial_index=[],  # empty list as index means get all for this session
                           data_type='betas_fithrf_GLMdenoise_RR',
                           data_format='func1pt8mm',
                           mask=v1_mask)

print(betas.shape)

####################################

"""
Waiting for download - shows images
"""
imgs = nsda.read_images([569, 575], show=True)


### GOD ###
# from scipy.io import loadmat
#
# GOD = 'D:/Honours/Object Decoding Dataset/mat/'
# Subject1_Path = GOD + 'Subject1.mat'
# Subject1 = loadmat(Subject1_Path)

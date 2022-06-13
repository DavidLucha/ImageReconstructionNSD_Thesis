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

from nsd_access import NSDAccess



nsda = NSDAccess('D:/NSD/')

# get the betas for a given subject, session and set of trials
betas = nsda.read_betas(subject='subj01',
                        session_index=1,
                        trial_index=[], # empty list as index means get all for this session
                        data_type='betas_fithrf_GLMdenoise_RR',
                        data_format='func1pt8mm')

print(betas.shape)

### GOD ###
from scipy.io import loadmat

GOD = 'D:/Honours/Object Decoding Dataset/mat/'
Subject1_Path = GOD + 'Subject1.mat'
Subject1 = loadmat(Subject1_Path)

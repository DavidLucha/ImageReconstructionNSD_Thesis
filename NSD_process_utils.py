# Dependencies
import pandas as pd
import os.path
import os
import numpy as np
import h5py
from sklearn import preprocessing
import matplotlib.pyplot as plt
from PIL import Image


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


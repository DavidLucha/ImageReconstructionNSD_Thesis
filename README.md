# Reconstructing Seen Images From Natural Scenes Dataset Using Deep Learning

This is the repository for my Psychology honours thesis. This involves a rough technical replication of Ren et al.'s (2019) d-VAE/GAN using two datasets: the Generic Object Decoding (DOG) Dataset and the Natural Scenes Dataset (NSD). We look at whether ultra-high field fMRI can improve the capabilities of neural networks for natural image reconstruction. I also investigate the extent to which voxel resolution and training set size contribute to reconstruction quality. Finally, I demonstrate the relative importance of ROIs for reconstruction using the NSD. 

Please email me at dlucha@uqconnect.edu.au if you have any questions.


## Training Instructions

1. Download data and set training_config.data_root to directory
2. To pretrain, run pretrain.py. This will train a vae-gan model on image-only dataset (ImageNet 2011 Validation Set - download here: [LINK])
3. For each dataset (using arg --dataset) run train_stage1.py. 
4. For each subject, load the trained model from Stage 1 (using arg --prev_stage_trained), and run train_stage2.py and train_stage3.py sequentially (per subject). Make sure you update the --prev_stage_trained argument for Stage 3 (i.e., loading model from stage 2).

Note: X_config.py are important for detailing the directories for inputs and outputs.


### Requirements
As per requirements.txt, the following are needed to run the main training scripts:
- bdpy==0.18
- h5py==2.10.0
- matplotlib==3.3.4
- nibabel==3.2.2
- nilearn==0.9.1
- numpy==1.19.2
- pandas==1.1.5
- Pillow==9.2.0
- progressbar33==2.4
- pycocotools==2.0.4
- pycocotools_windows==2.0.0.2
- scikit_image==0.17.2
- scikit_learn==1.1.1
- scipy==1.5.2
- skimage==0.0
- tensorboardX==2.5.1
- tensorflow==2.9.1
- torch==1.10.2
- torchvision==0.11.3

Note: See /data/requirement_w_preprocess.txt for dependencies used in data preprocessing.


### Data
The data for training needs to be in a specific format for the network to train properly. Data is structured as a list of dictionaries ({"fMRI":"normalized voxel activity","image":"/path/to/seen/image"}). Code used for processing the raw data can be found in /data/.
#### Generic Object Decoding Dataset
- The raw fMRI data can be downloaded: [Deep Image Reconstruction@OpenNeuro](https://openneuro.org/datasets/ds001506)
#### Natural Scenes Dataset
- Instructions to download raw data ...
- Processed data can be downloaded here: 
#### ImageNet 2011 Validation Set (For Pretraining)
- The image set for pretrained network can be downloaded here:


### Model

- Trained models can be downloaded from here:
   [d-vaegan-model](https://drive.google.com/file/...)


### Results
#### Study 1
- The reconstructed images of Horikawa17 (GOD) dataset: 
 [Horikawa17](https://github.com/.../data4_imgs.pdf)
- The reconstructed images of NSD dataset (Allen et al., 2021): 
 [Horikawa17](https://github.com/.../data4_imgs.pdf)
- - The reconstructed images of non-pretrained NSD dataset (Allen et al., 2021): 
 [Horikawa17](https://github.com/.../data4_imgs.pdf)





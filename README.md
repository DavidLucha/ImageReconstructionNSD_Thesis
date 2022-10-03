# Reconstructing Seen Images From Natural Scenes Dataset Using Deep Learning

This is the repository for my Psychology honours thesis. This involves a rough technical replication of Ren et al.'s (2019) d-VAE/GAN using two datasets: the Generic Object Decoding (DOG) Dataset and the Natural Scenes Dataset (NSD). We look at whether ultra-high field fMRI can improve the capabilities of neural networks for natural image reconstruction. I also investigate the extent to which voxel resolution and training set size contribute to reconstruction quality. Finally, I demonstrate the relative importance of ROIs for reconstruction using the NSD. 

Please email me at dlucha@uqconnect.edu.au if you have any questions.


## Training Instructions

1. Set up data (see Data section below).
2. Set training_config.data_root to directory of your data.
3. To pretrain, run pretrain_WAE.py. This will train a wae-gan model on image-only dataset.
4. Then for each participant's data, load the trained model from Stage 1 (using args to define run name), and run train_stage2.py and train_stage3.py sequentially (per subject). Make sure you update the --prev_stage_trained argument for Stage 3 (i.e., loading model from stage 2).

Note: training_config.py and model_config.py are important for detailing the directories for inputs and outputs.


### Requirements
As per requirements.txt, the following are needed to run the main training scripts:
* matplotlib~=3.5.2
* numpy~=1.22.3
* pandas~=1.4.3
* Pillow~=9.0.1
* scipy~=1.7.3
* torchvision~=0.13.0
* scikit-learn~=1.1.1
* seaborn~=0.11.2
* pingouin~=0.5.2
* statsmodels~=0.13.2
* lpips~=0.1.4

### Data
The data for training needs to be in a specific format for the network to train properly. Data is structured as a list of dictionaries ({"fMRI":"normalized voxel activity","image":"/path/to/seen/image"}). Code used for processing the raw data can be found in NSD_preprocess.py and NSD_process_utils.py.

#### Natural Scenes Dataset
- To access the dataset please go to http://naturalscenesdataset.org/

## References
References to literature can be found in thesis. Link will be added soon.

## Repositories 
Maria Podguzova's d-WAE/GAN: [Link](https://github.com/MariaPdg/thesis-fmri-reconstruction)
Gaziv et al., Encoder-Decoder model, inspired the n-way comparison code: [Link](https://github.com/WeizmannVision/SelfSuperReconst)
VAE/GAN PyTorch: [Link](https://github.com/lucabergamini/VAEGAN-PYTORCH)
Ren's d-VAE/GAN: [Link](https://github.com/ziqiren/dvaeganImageRecon)
WAE Implementation: [Link](https://github.com/tolstikhin/wae)
SSIM: [Link](https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb) (Not used in Thesis)




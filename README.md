# Reconstructing Seen Images from fMRI Using Machine Learning
This is the official repository for my  thesis project. The full thesis can be read here: [Link](https://github.com/DavidLucha/ImageReconstructionNSD_Thesis/blob/main/NSD_ImageReconstruction_Thesis_Full.pdf)
## Abstract

Deep learning approaches have been increasingly successful in reconstructing seen images from brain data measured using functional magnetic resonance imaging (fMRI). This brain decoding research has been vital in understanding the how visual information is encoded in low- and high-order visual areas. However, much of the existing work is limited due to the lack of high-quality, large scale neuroimaging datasets. The Natural Scenes Dataset (NSD) was developed to address this limitation and is the most comprehensive fMRI dataset of its kind, providing greater spatial resolution (i.e., voxel resolution) than previous datasets. However, there is currently no available research investigating the viability of the dataset for image reconstruction tasks. This thesis aimed to evaluate whether machine learning approaches could successfully use the NSD to generate high fidelity reconstructions from fMRI. Additionally, the thesis aimed to explore the role of voxel resolution on the quality of reconstructions, and to use the NSD to explore the supplementary role of high-order visual areas in perception. Across three studies, a series of artificial neural networks were trained to reconstruct seen images from novel human brain data provided by the NSD. Reconstruction quality was assessed using three metrics of identification accuracy computed using two leading image similarity measures. In study one, it was found that neural networks generated reconstructions of the seen image with above chance accuracy. In study two, finer voxel resolutions resulted in higher quality reconstructions than coarser voxel resolutions. Finally, study three revealed that alone, low-order visual areas were most important for reconstructing seen images. However, high-order visual areas provided a significant boost to reconstruction quality. These findings suggest that the NSD is a promising tool for brain decoding research and sheds new light on how high-order visual areas provide contextual and semantic information required for visual perception.


### Contact
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

## Evaluation
Evaluation code can be found in the following.

#### net_evaluation_tabular.py
Loads each trained network and calculates the metrics between each reconstruction and each ground truth image in the test set. The result of this is used with all_net_eval.py below.

#### all_net_eval.py
main(): runs the n-way comparisons for each networks test reconstructions
study_1(): runs the permutation test for each participant on 1.8mm data. Calculates 95% CIs.
study_2n3(): for each study it combines the accuracy scores across all participants (pooled; N = 6,976) and calculates bootstrapped 95% CIs.

#### stats.py 
Runs the statistics for Study 2 and Study 2 (Wilcoxon and Friedman Tests)

## Models
Model details can be found in model_2.py and model_config_2.py.

## References
References to literature can be found in thesis. Link will be added soon.

## Repositories 
* Maria Podguzova's d-WAE/GAN: [Link](https://github.com/MariaPdg/thesis-fmri-reconstruction)
* Gaziv et al., Encoder-Decoder model, inspired the n-way comparison code: [Link](https://github.com/WeizmannVision/SelfSuperReconst)
* VAE/GAN PyTorch: [Link](https://github.com/lucabergamini/VAEGAN-PYTORCH)
* Ren's d-VAE/GAN: [Link](https://github.com/ziqiren/dvaeganImageRecon)
* WAE Implementation: [Link](https://github.com/tolstikhin/wae)
* SSIM: [Link](https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb) (Not used in Thesis)




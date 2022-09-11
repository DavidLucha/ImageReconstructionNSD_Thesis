"""
Utils for data loaders
Training utils and assessments
"""
import argparse
import os.path

import os
import math
import torch
import numpy as np
import random
import logging
import lpips as lpips
import statistics
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms
import time

from torch import nn, no_grad
from torch.autograd import Variable
from PIL import Image
from os import listdir

import torchvision

from typing import Any


"""
Data loaders
"""


class FmriDataloader(object):
    """
    Dataloader for ROI information in BOLD5000 dataset
    https://ndownloader.figshare.com/files/12965447
    """

    def __init__(self, dataset, root_path=None, transform=None, standardizer="none"):
        """
        The constructor to initialized paths to images and fmri data
        :param data_dir: directories to fmri and image data
        :param transform: list of transformations to be applied
        """
        self.dataset = dataset
        self.transform = transform
        self.root_path = root_path
        self.standardizer = standardizer
        self.path_to_rm = root_path + '/images/valid/'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        voxels = self.dataset[idx]['fmri']

        # Assuming image path in pickle file starts at 'images/'
        # Then add the set root directory to the start
        if self.root_path is not None and self.root_path not in self.dataset[idx]['image']:
            self.dataset[idx]['image'] = self.root_path + self.dataset[idx]['image']

        stimulus = Image.open(self.dataset[idx]['image']) # Needed to work with visions.transforms

        # Applies transformations only to image
        if self.transform:
            mod_stimulus = self.transform(stimulus) # Was sample = (sample)

        # Applies tensor conversion to voxels
        fmri_tensor = torch.FloatTensor(voxels)
        # Standardises the sample, mean of 0, std of 1.
        # comment the below if you have scaled this at the whole dataset stage
        if self.standardizer == "z":
            fmri_tensor = self.standardize(fmri_tensor)
        elif self.standardizer == "norm:":
            fmri_tensor = F.normalize((fmri_tensor))

        # TODO: Check this doesn't break everything
        # root path = args.data_root + dataset (NSD)

        path = self.dataset[idx]['image'].replace(self.path_to_rm, '')

        transformed_sample = {'fmri': fmri_tensor, 'image': mod_stimulus, 'path': path}

        return transformed_sample

    def standardize(self, fmri):
        mu = torch.mean(fmri)
        std = torch.std(fmri)
        return (fmri - mu)/std


class VQDataloader(object):

    def __init__(self, data_dir, transform=None, pickle=True):
        """
        The constructor to initialized paths to ImageNet images
        :param data_dir: directory to ImageNet images
        :param transform: image transformations
        :param pickle: True if names are stored in pickle file (deprecated)
        """
        self.transform = transform
        if not pickle:
            self.image_names = [os.path.join(data_dir, img) for img in listdir(data_dir) if os.path.join(data_dir, img)]
        else:
            self.image_names = data_dir

        # Get all data and calculate variance
        calc_var = False
        if calc_var:
            self.data: Any = []

            # Applies transformations to get 0-255 RGB values to run var calculation
            trans = torch.nn.Sequential(
                transforms.Resize(100),
                transforms.RandomCrop((100, 100)),
            )

            grey_colour = GreyToColor(100)

            img_count = 0
            # Grabs all images RGB values and places in array (C, H, W)
            for img in listdir(data_dir):
                # if img == 'ILSVRC2011_val_00000004.JPEG' or img == 'ILSVRC2011_val_00000040.JPEG':
                im_path = os.path.join(data_dir, img)
                im = Image.open(im_path)
                im = trans(im)
                im = transforms.PILToTensor()(im)
                im = grey_colour(im)
                self.data.append(im)
                img_count += 1
                if not img_count % 100:
                    print('{} images processed.'.format(img_count))

            # Rearranges the array to (N, H, W, C)
            self.data = np.vstack(self.data).reshape(-1, 3, 100, 100)
            self.data = self.data.transpose((0, 2, 3, 1))

            self.data_variance = np.var(self.data / 255.0)
            raise Exception('check')
        # self.data.append(mod_im)

        # vqVAEGAN code
        # self.data: Any = []

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image = Image.open(self.image_names[idx])

        if self.transform:
            image = self.transform(image)

        return image

"""
More Dataloaders
"""


class ImageNetDataloader(object):

    def __init__(self, data_dir, transform=None, pickle=True):
        """
        The constructor to initialized paths to ImageNet images
        :param data_dir: directory to ImageNet images
        :param transform: image transformations
        :param pickle: True if names are stored in pickle file (deprecated)
        """
        self.transform = transform
        if not pickle:
            self.image_names = [os.path.join(data_dir, img) for img in listdir(data_dir) if os.path.join(data_dir, img)]
        else:
            self.image_names = data_dir

        # vqVAEGAN code
        self.data: Any = []

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        image = Image.open(self.image_names[idx])

        if self.transform:
            image = self.transform(image)

        return image

"""
Evaluation Utilities
"""


class GreyToColor(object):
    """
    Converts grey tensor images to tensor with 3 channels
    """

    def __init__(self, size):
        """
        @param size: image size
        """
        self.image = torch.zeros([3, size, size])

    def __call__(self, image):
        """
        @param image: image as a torch.tensor
        @return: transformed image if it is grey scale, otherwise original image
        """

        out_image = self.image

        if image.shape[0] == 3:
            out_image = image
        else:
            out_image[0, :, :] = torch.unsqueeze(image, 0)
            out_image[1, :, :] = torch.unsqueeze(image, 0)
            out_image[2, :, :] = torch.unsqueeze(image, 0)

        return out_image


class PearsonCorrelation(nn.Module):

    """
    Calculates Pearson Correlation Coefficient
    """

    def __init__(self):
        super(PearsonCorrelation, self).__init__()

    def forward(self, y_pred, y_true):
        """
        @param y_pred: tensor [batch_size x channels x width x height]
            Predicted image
        @param y_true: tensor [batch_size x channels x width x height]
            Ground truth image
        @return: float
            Pearson Correlation Coefficient
        """

        vx = y_pred - torch.mean(y_pred)
        vy = y_true - torch.mean(y_true)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        loss = cost.mean()

        return loss


"""
Alternative Correlation
def average_correlation(samples, reference):
# from StYves https://github.com/styvesg/gan-decoding-supplementary/blob/master/gan_imaging_cifar-10.ipynb
# Paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8616183&casa_token=2JDy6AEq9x8AAAAA:l2VwkLcwO61sQIVsUk6Y1Ai7foFKSl2_o29ubXzpNiJY1tPFh1-2YfHUqObIWJ-ZpbbLcCr_7g
    ac = 0
    for s in samples:
        ac += np.corrcoef(s.flatten(), reference.flatten())[0,1]
    return ac / len(samples)
    
OR from Beliy voxels to pixels and back
def corr_percintiles(y,y_pred, per = [50,75,90]):
    num_voxels = y.shape[1]
    corr = np.zeros([num_voxels])

    for i in range(num_voxels):
        corr[i] = stat.pearsonr(y[:, i], y_pred[:, i])[0]
    corr = np.nan_to_num(corr)

    corr_per = []
    for p in per:
        corr_per.append(np.percentile(corr,p))
    return corr_per
"""


class StructuralSimilarity(nn.Module):

    """
    Structural Similarity Index Measure (mean of local SSIM)
    see Z. Wang "Image quality assessment: from error visibility to structural similarity"

    Calculates the SSIM between 2 images, the value is between -1 and 1:
     1: images are very similar;
    -1: images are very different

    Adapted from https://github.com/pranjaldatta/SSIM-PyTorch/blob/master/SSIM_notebook.ipynb
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(StructuralSimilarity, self).__init__()
        self.mean = mean
        self.std = std

    def gaussian(self, window_size, sigma):

        """
        Generates a list of Tensor values drawn from a gaussian distribution with standard
        diviation = sigma and sum of all elements = 1.

        @param window_size: 11 from the paper
        @param sigma: standard deviation of Gaussian distribution
        @return: list of values, length = window_size
        """

        gauss = torch.Tensor(
            [math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):

        """
        @param window_size: 11 from the paper
        @param channel: 3 for RGB images
        @return: 4D window with size [channels, 1, window_size, window_size]

        """
        # Generates an 1D tensor containing values sampled from a gaussian distribution.
        _1d_window = self.gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
        # Converts it to 2D
        _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
        # Adds extra dimensions to convert to 4D
        window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

        return window

    def forward(self, img1, img2, val_range=255, window_size=11, window=None, size_average=True, full=False):

        """
        Calculating Structural Similarity Index Measure

        @param img1: torch.tensor
        @param img2: torch.tensor
        @param val_range: 255 for RGB images
        @param window_size: 11 from the paper
        @param window: created with create_window function
        @param size_average: if True calculates the mean
        @param full: if true, return result and contrast_metric
        @return: value of SSIM
        """
        # try:
        #     # data validation
        #     if torch.min(img1) < 0.0 or torch.max(img1) > 1.0:  # if normalized with mean and std
        #         img1 = denormalize_image(img1, mean=self.mean, std=self.std).detach().clone()
        #         if torch.min(img1) < 0.0 or torch.max(img1) > 1.0:
        #             raise ValueError
        #
        #     if torch.min(img2) < 0.0 or torch.max(img2) > 1.0:  # if normalized with mean and std
        #         img2 = denormalize_image(img2, mean=self.mean, std=self.std).detach().clone()
        #         if torch.min(img2) < 0.0 or torch.max(img2) > 1.0:
        #             raise ValueError

        # except ValueError as error:
        #     print('Image values in SSIM must be between 0 and 1 or normalized with mean and std', error)

        L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

        pad = window_size // 2

        try:
            _, channels, height, width = img1.size()
        except:
            channels, height, width = img1.size()

        # if window is not provided, init one
        if window is None:
            real_size = min(window_size, height, width)  # window should be atleast 11x11
            window = self.create_window(real_size, channel=channels).to(img1.device)

        # calculating the mu parameter (locally) for both images using a gaussian filter
        # calculates the luminosity params
        mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
        mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        # now we calculate the sigma square parameter
        # Sigma deals with the contrast component
        sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

        # Some constants for stability
        C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
        C2 = (0.03) ** 2

        contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        contrast_metric = torch.mean(contrast_metric)

        numerator1 = 2 * mu12 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

        if size_average:
            result = ssim_score.mean()
        else:
            result = ssim_score.mean(1).mean(1).mean(1)

        if full:
            return result, contrast_metric

        return result


def save_out(model, dataloader, path=None, resize=200):
    """
    Save recon and real images individually

    @param model: network for evaluation
    @param dataloader: DataLoader object
    @param path: path to save images
    @param save: True if save images, otherwise False
    @param resize: the size of the image to save
    """

    real_path = os.path.join(path, 'real')
    recon_path = os.path.join(path, 'recon')

    for batch_idx, data_batch in enumerate(dataloader):
        model.eval()

        with no_grad():

            data_target = Variable(data_batch['image'], requires_grad=False).cpu().detach()

            out, _ = model(data_batch['fmri'])

            out = out.data.cpu()

            data_path = data_batch['path']
            
            # Make directory
            if not os.path.exists(real_path):
                os.makedirs(real_path)
            if not os.path.exists(recon_path):
                os.makedirs(recon_path)

            if resize is not None:
                out = F.interpolate(out, size=resize)
                data_target = F.interpolate(data_target, size=resize)
            for i, im in enumerate(out):
                torchvision.utils.save_image(im, fp=os.path.join(recon_path, '{}.png'.format(data_path[i])), normalize=True)
            for i, im in enumerate(data_target):
                torchvision.utils.save_image(im, fp=os.path.join(real_path, '{}.png'.format(data_path[i])), normalize=True)


def save_network_out(model, dataloader, path=None, save=False, resize=None):
    """
    Calculate metrics for the dataset specified with dataloader

    @param model: network for evaluation
    @param dataloader: DataLoader object
    @param norm: normalization
    @param mean: mean of the dataset
    @param std: standard deviation of the dataset
    @param path: path to save images
    @param save: True if save images, otherwise False
    @param resize: the size of the image to save
    @return: mean PCC, mean SSIM, MSE, mean IS (inception score)
    """
    real_path = path + '/real'
    recon_path = path + '/recon'

    for batch_idx, data_batch in enumerate(dataloader):
        model.eval()

        with no_grad():

            data_target = Variable(data_batch['image'], requires_grad=False).cpu().detach()
            data_path = data_batch['path']

            out, _ = model(data_batch['fmri'])

            out = out.data.cpu()
            if save:
                # Make directory
                if not os.path.exists(real_path):
                    os.makedirs(real_path)
                if not os.path.exists(recon_path):
                    os.makedirs(recon_path)

                if resize is not None:
                    out = F.interpolate(out, size=resize)
                    data_target = F.interpolate(data_target, size=resize)
                for i, im in enumerate(out):
                    # torchvision.utils.save_image(size_out,
                    #                              fp=save_path + '/' + str(idx) + str(data_path[i]) + '_recon.png',
                    #                              normalize=True)
                    torchvision.utils.save_image(im, fp=recon_path + '/' + str(data_path[i]), normalize=True)  # could add + str(i)
                for i, im in enumerate(data_target):
                    torchvision.utils.save_image(im, fp=real_path + '/' + str(data_path[i]), normalize=True)


def evaluate(model, dataloader, norm=True, mean=None, std=None, path=None, save=False, resize=None):
    """
    Calculate metrics for the dataset specified with dataloader

    @param model: network for evaluation
    @param dataloader: DataLoader object
    @param norm: normalization
    @param mean: mean of the dataset
    @param std: standard deviation of the dataset
    @param path: path to save images
    @param save: True if save images, otherwise False
    @param resize: the size of the image to save
    @return: mean PCC, mean SSIM, MSE, mean IS (inception score)
    """
    # import lpips

    pearson_correlation = PearsonCorrelation()
    structural_similarity = StructuralSimilarity()
    mse_loss = nn.MSELoss()
    # perceptual_similarity = lpips.LPIPS(net='alex')
    ssim = 0
    pcc = 0
    mse = 0
    # lpips = 0
    # is_mean = 0
    real_path = path + '/real'
    recon_path = path + '/recon'

    for batch_idx, data_batch in enumerate(dataloader):
        model.eval()

        with no_grad():

            data_target = Variable(data_batch['image'], requires_grad=False).cpu().detach()
            data_path = data_batch['path']

            out, _ = model(data_batch['fmri'])

            out = out.data.cpu()
            if save:
                # Make directory
                if not os.path.exists(real_path):
                    os.makedirs(real_path)
                if not os.path.exists(recon_path):
                    os.makedirs(recon_path)

                if resize is not None:
                    out = F.interpolate(out, size=resize)
                    data_target = F.interpolate(data_target, size=resize)
                for i, im in enumerate(out):
                    # torchvision.utils.save_image(size_out,
                    #                              fp=save_path + '/' + str(idx) + str(data_path[i]) + '_recon.png',
                    #                              normalize=True)
                    torchvision.utils.save_image(im, fp=recon_path + '/' + str(data_path[i]), normalize=True)  # could add + str(i)
                for i, im in enumerate(data_target):
                    torchvision.utils.save_image(im, fp=real_path + '/' + str(data_path[i]), normalize=True)
        if norm and mean is not None and std is not None:
            data_target = denormalize_image(data_target, mean=mean, std=std)
            out = denormalize_image(out, mean=mean, std=std)
        pcc += pearson_correlation(out, data_target)
        ssim += structural_similarity(out, data_target)
        mse += mse_loss(out, data_target)
        # TODO: Check this is working - does it not work per image?
        # lpips += perceptual_similarity(out, data_target)
        # is_mean += inception_score(out, resize=True)

    mean_pcc = pcc / (batch_idx+1)
    mean_ssim = ssim / (batch_idx+1)
    mse_loss = mse / (batch_idx+1)
    # mean_lpips = lpips / (batch_idx + 1)
    # is_mean = is_mean / (batch_idx+1)

    return mean_pcc, mean_ssim, mse_loss  # , mean_lpips

def denormalize_image(pred, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    denorm_img = pred.detach().clone()  # deep copy of tensor
    for i in range(3):
        denorm_img[:, i, :, :] = denorm_img[:, i, :, :] * std[i] + mean[i]

    return denorm_img


def objective_assessment(model, dataloader, top=5, save_path="D:/Lucha_Data/misc/", repeats=100, resize=200, save=True):
    """
    Calculates objective score of the predictions

    @param model: network for evaluation
    @param dataloader: DataLoader object
    @param top: n-top score: n=2,5,10
    @return: objective score - percentage of correct predictions
    """
    import lpips

    perceptual_similarity = lpips.LPIPS(net='alex')  # .to('cuda')
    pearson_correlation = PearsonCorrelation()
    structural_similarity = StructuralSimilarity()
    mse_loss = nn.MSELoss()

    # Make a results var to save individual image metrics
    results = dict(
        repeat=[],
        trial_id=[],
        image_path=[],
        pcc_im=[],
        ssim_im=[],
        lpips_im=[],
        mse_im=[]
    )

    objective_score = dict(
        repeat=[],
        pcc_score=[],
        ssim_score=[],
        lpips_score=[],
        mse_score=[]
    )

    for batch_idx, data_batch in enumerate(dataloader):
        model.eval()

        with no_grad():
            cpu = False
            if cpu:
                data_target = Variable(data_batch['image'], requires_grad=False).cpu().detach()

                out, _ = model(data_batch['fmri'])
                out = out.data.cpu()
            else:
                data_target = Variable(data_batch['image'], requires_grad=False).float().to('cuda')
                data_fmri = Variable(data_batch['fmri'], requires_grad=False).float().to('cuda')
                # data_path = Variable(data_batch['path'], requires_grad=False).float().to('cuda')
                data_path = data_batch['path']

                out, _ = model(data_fmri)
                # out_cpu = out.data.cpu()
                target_cpu = data_target.data.cpu()
                # out = out.data.cpu()

            # TODO Insert loop here for repeats
            for repeat in range(repeats):
                # [PCC, SSIM, PSM, MSE]
                true_positives = torch.tensor([0, 0, 0, 0])
                dataset_size = 0
                score_pcc = 0
                score_ssim = 0
                score_lpips = 0
                score_mse = 0

                for idx, image in enumerate(out):
                    numbers = list(range(0, len(out)))
                    numbers.remove(idx)
                    for i in range(top-1):
                        # Get random number not including ID of original
                        # Note: this is only a random choice of the mini-batch.
                        rand_idx = random.choice(numbers)
                        # PCC Metric
                        score_rand_pcc = pearson_correlation(image, data_target[rand_idx])
                        score_recon_pcc = pearson_correlation(image, data_target[idx])
                        if score_recon_pcc > score_rand_pcc:
                            score_pcc += 1

                        # SSIM
                        # TODO: check if the unsqueeze is needed
                        image_for_ssim = torch.unsqueeze(image, 0)
                        target_gt_for_ssim = torch.unsqueeze(data_target[idx], 0)
                        target_rand_for_ssim = torch.unsqueeze(data_target[rand_idx], 0)
                        score_rand_ssim = structural_similarity(image_for_ssim, target_rand_for_ssim)
                        score_recon_ssim = structural_similarity(image_for_ssim, target_gt_for_ssim)
                        if score_recon_ssim > score_rand_ssim:
                            score_ssim += 1

                        # Perceptual Similarity Metric - requires -1 to 1 normalization
                        # TODO: check if it's normalized
                        image_cpu = image.data.cpu()
                        score_rand_lpips = perceptual_similarity(image_cpu, target_cpu[rand_idx])
                        score_recon_lpips = perceptual_similarity(image_cpu, target_cpu[idx])
                        # Lower number means images are 'closer' together
                        if score_recon_lpips < score_rand_lpips:
                            score_lpips += 1

                        # MSE
                        score_rand_mse = mse_loss(image, data_target[rand_idx])
                        score_recon_mse = mse_loss(image, data_target[idx])
                        if score_recon_mse < score_rand_mse:
                            score_mse += 1

                    # Save individual scores per image
                    results['repeat'].append(repeat + 1)
                    results['trial_id'].append(dataset_size)
                    results['image_path'].append(data_path[idx])
                    results['pcc_im'].append(score_recon_pcc.item())
                    results['ssim_im'].append(score_recon_ssim.item())
                    results['lpips_im'].append(score_recon_lpips.item())
                    results['mse_im'].append(score_recon_mse.item())

                    if score_pcc == top - 1:
                        true_positives[0] += 1
                    if score_ssim == top - 1:
                        true_positives[1] += 1
                    if score_lpips == top - 1:
                        true_positives[2] += 1
                    if score_mse == top - 1:
                        true_positives[3] += 1

                    dataset_size += 1
                    score_pcc = 0
                    score_ssim = 0
                    score_lpips = 0
                    score_mse = 0

                # Get objective scores
                acc = true_positives.float() / dataset_size

                # Save all results
                objective_score['repeat'].append(repeat + 1)
                objective_score['pcc_score'].append(acc[0].item())
                objective_score['ssim_score'].append(acc[1].item())
                objective_score['lpips_score'].append(acc[2].item())
                objective_score['mse_score'].append(acc[3].item())
                print('PCC score at repeat {} is: {}'.format(repeat, acc[0].item()))
                print('SSIM score at repeat {} is: {}'.format(repeat, acc[1].item()))
                print('LPIPS score at repeat {} is: {}'.format(repeat, acc[2].item()))
                print('MSE score at repeat {} is: {}'.format(repeat, acc[3].item()))


    # averages = []
    # for key, values in results.items():
    #     if key in ('pcc_im', 'ssim_im', 'lpips_im', 'mse_im'):
    #         averages.append(sum(values)/float(len(values)))

    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(os.path.join(save_path, "trial_results.csv"), index=True)

    obj_score_to_save = pd.DataFrame(objective_score)
    obj_score_to_save.to_csv(os.path.join(save_path, "{}_way_obj_results.csv".format(top)), index=True)

    return objective_score # , objective_score[1], objective_score[2], objective_score[3]  # , averages


def parse_args(args):  # TODO: Add a second term here 'stage' and do conditional
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', default=3, type=int)
    parser.add_argument('--y', default=5, type=int)
    """
    parser.add_argument('--input', help="user path where the datasets are located", type=str)

    parser.add_argument('--batch_size', default=training_config.batch_size, help='batch size for dataloader', type=int)
    parser.add_argument('--epochs', default=training_config.n_epochs, help='number of epochs', type=int)
    parser.add_argument('--image_size', default=training_config.image_size, help='size to which image should '
                                                                                     'be scaled', type=int)
    parser.add_argument('--num_workers', '-nw', default=training_config.num_workers,
                        help='number of workers for dataloader',type=int)
    # Pretrained network components
    parser.add_argument('--pretrained_net', '-pretrain', default=training_config.pretrained_net,
                        help='pretrained network', type=str)
    parser.add_argument('--load_epoch', '-pretrain_epoch', default=training_config.load_epoch,
                        help='epoch of the pretrained model', type=int)
    parser.add_argument('--dataset', default='GOD', help='GOD, NSD', type=str)
    parser.add_argument('--subset', default='1.8mm', help='1.8mm, 3mm, 5S_Small, 8S_Small,'
                                                          '5S_Large, 8S_Large', type=str)
    parser.add_argument('--recon_level', default=training_config.recon_level,
                        help='reconstruction level in the discriminator',
                        type=int)  # NOT REALLY SURE WHAT RECON LEVEL DOES TBH - see VAE GAN implementation

    """
    return parser.parse_args(args)


def imgnet_dataloader(batch_size):
    # Define transform
    transform = transforms.Compose([
        transforms.Resize(100),
        transforms.CenterCrop(100),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Define dataset
    train_set = torchvision.datasets.CIFAR10(root='/data', train=True, download=True, transform=transform)

    # Initialize DataLoader
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return data_loader


def NLLNormal(pred, target):

    c = -0.5 * np.log(2 * np.pi)  # -0.5 * ~0.80 = -.40~
    multiplier = 1.0 / (2.0 * 1 ** 2)  # 0.5 (1/2)
    tmp = torch.square(pred - target)
    tmp *= -multiplier
    tmp += c
    # sum, then mean
    # tmp = torch.mean(tmp)

    return tmp


# Potentiation code
# xc = a * b c
def potentiation(start_lr, decay_lr, epochs):
    x_c = start_lr * (decay_lr ** epochs)
    return x_c


def objective_assessment_table(model, dataloader, save_path="D:/Lucha_Data/misc/"):
    """
    Calculates objective score of the predictions

    @param model: network for evaluation
    @param dataloader: DataLoader object
    @param top: n-top score: n=2,5,10
    @return: objective score - percentage of correct predictions
    """
    import lpips

    # CPU
    # perceptual_similarity = lpips.LPIPS(net='alex') # .to('cuda')
    perceptual_similarity_gpu = lpips.LPIPS(net='alex').cuda()
    pearson_correlation = PearsonCorrelation().cuda()
    structural_similarity = StructuralSimilarity().cuda()

    header = []

    table_pcc = np.zeros(shape=(872,872))
    table_ssim = np.zeros(shape=(872,872))
    table_lpips = np.zeros(shape=(872,872))

    for batch_idx, data_batch in enumerate(dataloader):
        model.eval()

        with no_grad():
            cpu = False
            if cpu:
                data_target = Variable(data_batch['image'], requires_grad=False).cpu().detach()

                out, _ = model(data_batch['fmri'])
                out = out.data.cpu()
            else:
                data_target = Variable(data_batch['image'], requires_grad=False).float().to('cuda')
                data_fmri = Variable(data_batch['fmri'], requires_grad=False).float().to('cuda')
                # data_path = Variable(data_batch['path'], requires_grad=False).float().to('cuda')
                data_path = data_batch['path']

                out, _ = model(data_fmri)
                # TODO: could add an argument for path, and spit this out with the recon

                # TODO: uncomment for cpu use
                # out_cpu = out.data.cpu()
                # target_cpu = Variable(data_batch['image'], requires_grad=False).cpu().detach()

                # out = out.data.cpu()

            for idx, recon in enumerate(out):
                print('Evaluating reconstruction:', idx)
                numbers = list(range(0, len(out)))

                row_pcc = []
                row_ssim = []
                row_lpips = []

                start = time.time()
                for i in numbers:
                    # if not idx % 20:
                    #     print(i)
                    if idx==0:
                        header.append(data_path[i])

                    # PCC Metric
                    # start = time.time()
                    score_pcc = pearson_correlation(recon, data_target[i])
                    # print('score between recon {} and real {} is {}'.format(idx, i, score_pcc))
                    row_pcc.append(score_pcc)
                    # end = time.time()
                    # print('time for pcc =', end - start)

                    # SSIM
                    # TODO: check if the unsqueeze is needed
                    # start = time.time()
                    recon_for_ssim = torch.unsqueeze(recon, 0)
                    target_for_ssim = torch.unsqueeze(data_target[i], 0)
                    score_ssim = structural_similarity(recon_for_ssim, target_for_ssim)
                    row_ssim.append(score_ssim)
                    # end = time.time()
                    # print('time for ssim =', end - start)

                    # Perceptual Similarity Metric - requires -1 to 1 normalization
                    # TODO: check if it's normalized
                    # start = time.time()
                    # CPU
                    # recon_cpu = recon.data.cpu()
                    # score_lpips = perceptual_similarity(out_cpu[idx], target_cpu[i])
                    # GPU
                    score_lpips = perceptual_similarity_gpu(recon, data_target[i])

                    row_lpips.append(score_lpips)
                    # end = time.time()
                    # print('time for lpips =', end - start)
                    # Lower number means images are 'closer' together

                    """if i == 5:
                        break"""

                    # if i == 50:
                    #     raise Exception('check')

                # Only thing is it might get very slow towards the end?

                table_pcc[idx] = row_pcc
                table_ssim[idx] = row_ssim
                table_lpips[idx] = row_lpips
                end = time.time()
                print('time for numbers loop =', end - start)

    table_pcc_pd = pd.DataFrame(table_pcc, columns=header, index=header)
    table_pcc_pd.to_excel(os.path.join(save_path, "pcc_table.xlsx"))

    table_ssim_pd = pd.DataFrame(table_ssim, columns=header, index=header)
    table_ssim_pd.to_excel(os.path.join(save_path, "ssim_table.xlsx"))

    table_lpips_pd = pd.DataFrame(table_lpips, columns=header, index=header)
    table_lpips_pd.to_excel(os.path.join(save_path, "lpips_table.xlsx"))

    return table_pcc_pd, table_ssim_pd, table_lpips_pd


def objective_assessment_table_batch(model, dataloader, save_path="D:/Lucha_Data/misc/"):
    """
    Calculates objective score of the predictions

    @param model: network for evaluation
    @param dataloader: DataLoader object
    @param top: n-top score: n=2,5,10
    @return: objective score - percentage of correct predictions
    """
    import lpips

    # CPU
    perceptual_similarity_gpu = lpips.LPIPS(net='alex').cuda()
    pearson_correlation = PearsonCorrelation().cuda()
    # structural_similarity = StructuralSimilarity().cuda()

    header = []

    table_pcc = np.zeros(shape=(872,872))
    # table_ssim = np.zeros(shape=(872,872))
    table_lpips = np.zeros(shape=(872,872))

    for batch_idx, data_batch in enumerate(dataloader):
        model.eval()

        with no_grad():
            cpu = False
            if cpu:
                data_target = Variable(data_batch['image'], requires_grad=False).cpu().detach()

                out, _ = model(data_batch['fmri'])
                out = out.data.cpu()
            else:
                data_target = Variable(data_batch['image'], requires_grad=False).float().to('cuda')
                data_fmri = Variable(data_batch['fmri'], requires_grad=False).float().to('cuda')
                # data_path = Variable(data_batch['path'], requires_grad=False).float().to('cuda')
                data_path = data_batch['path']

                out, _ = model(data_fmri)
                if batch_idx == 0:
                    out_comp = out.detach()
                    target_comp = data_target.detach()
                    path_comp = data_path
                    # print("out_comp size before stack: {}".format(out_comp.size()))
                else:
                    out_comp = torch.cat((out_comp, out))
                    target_comp = torch.cat((target_comp, data_target.detach()))
                    path_comp = path_comp + data_path
                    # print("out_comp size after stack: {}".format(out_comp.size()))

    for idx, recon in enumerate(out_comp):
        print('Evaluating reconstruction:', idx)
        numbers = list(range(0, len(out_comp)))

        row_pcc = []
        # row_ssim = []
        row_lpips = []

        start = time.time()
        for i in numbers:
            # if not idx % 20:
            #     print(i)
            if idx == 0:
                header.append(path_comp[i])

            # PCC Metric
            # start = time.time()

            score_pcc = pearson_correlation(recon, target_comp[i])
            row_pcc.append(score_pcc)

            # print('score between recon {} and real {} is {}'.format(idx, i, score_pcc))
            # end = time.time()
            # print('time for pcc =', end - start)

            # SSIM
            # start = time.time()

            # recon_for_ssim = torch.unsqueeze(recon, 0)
            # target_for_ssim = torch.unsqueeze(target_comp[i], 0)
            # score_ssim = structural_similarity(recon_for_ssim, target_for_ssim)
            # row_ssim.append(score_ssim)

            # end = time.time()
            # print('time for ssim =', end - start)

            # Perceptual Similarity Metric - requires -1 to 1 normalization
            score_lpips = perceptual_similarity_gpu(recon, target_comp[i])
            row_lpips.append(score_lpips)

            # end = time.time()
            # print('time for lpips =', end - start)
            # Lower number means images are 'closer' together

            """if i == 5:
                break"""

            # if i == 50:
            #     raise Exception('check')

        # Only thing is it might get very slow towards the end?

        table_pcc[idx] = row_pcc
        # table_ssim[idx] = row_ssim
        table_lpips[idx] = row_lpips
        end = time.time()
        print('time for numbers loop =', end - start)

    table_pcc_pd = pd.DataFrame(table_pcc, columns=header, index=header)
    table_pcc_pd.to_csv(os.path.join(save_path, "pcc_table.csv"))

    # table_ssim_pd = pd.DataFrame(table_ssim, columns=header, index=header)
    # table_ssim_pd.to_csv(os.path.join(save_path, "ssim_table.csv"))

    table_lpips_pd = pd.DataFrame(table_lpips, columns=header, index=header)
    table_lpips_pd.to_csv(os.path.join(save_path, "lpips_table.csv"))

    return table_pcc_pd, table_lpips_pd  # table_ssim_pd,


def nway_comp(data, n=5, repeats=10, metric="pcc"):
    import sys
    # To ensure reproducibility, and that every comparison gets the same 'random' batch
    # But need to work the how the assumption of independence works with our data
    # seed = random.randrange(sys.maxsize)
    # print('Seed was: ', seed)
    # random.seed(seed)
    # seed range as defined by .sample(random_state)
    # seed_list = random.sample(range(0, 2**32-1), repeats)

    # create container for full accuracy (cross repeats)
    accuracy_full = []

    results_net = []
    # results_net_rank = []
    count = 0

    for row in data:
        # results_recon_rank = []
        results_recon = []

        count += 1
        i = count - 1

        real_distance = [row[i]]

        repeat_count = 0
        for repeat in range(repeats):
            repeat_count += 1
            # if not repeat_count % 250:
            #     print(repeat_count)
            distractors = np.delete(row, i)
            distractor_distance = [row[ii] for ii in np.random.permutation(len(distractors))[:n - 1]]
            distances = real_distance + distractor_distance

            if metric == 'pcc':
                # print('PCC Evaluation...')
                # Here we include [::-1] to flip the order of the argsort becasue we want the highest PCC unlike LPIPS
                # results_recon_rank.append(np.argwhere(np.argsort(distances)[::-1] == 0).flatten()[0] / (len(distances) - 1))
                results_recon.append(np.argsort(distances)[::-1][0] == 0)
            else:
                # results_recon_rank.append(np.argwhere(np.argsort(distances) == 0).flatten()[0] / (len(distances) - 1))
                results_recon.append(np.argsort(distances)[0] == 0)

        # results_net_rank.append(np.mean(results_recon_rank))
        results_net.append(np.mean(results_recon))

    print('Overall accuracy is {}'.format(np.mean(results_net)))

    # Get accuracy per recon
    # accuracy = total_score / repeat_count * 100
    # print('Accuracy rate for repeat {} is {:.2f}%'.format(repeat_count, accuracy))

    # Concat this to full list for all recons
    # accuracy_full.append(accuracy)

    # print('Average n-way ({}) comparison accuracy: {:.2f} \n'
    #       'Standard deviation of n-way ({}) comparison accuracy: {:.2f}'.format(n, statistics.mean(accuracy_full),
    #                                                                n, statistics.stdev(accuracy_full)))

    return results_net  # , results_recon


def permutation(data, n=5, repeats=10, metric="pcc"):

    results_net = []
    # results_net_rank = []
    count = 0

    for row in data:
        # results_recon_rank = []
        results_recon = []

        count += 1
        i = count - 1

        # if not count % 100:
        #     print('Completed {} recons...'.format(count))

        real_distance = [row[i]]

        repeat_count = 0
        for repeat in range(repeats):
            repeat_count += 1
            # if not repeat_count % 10000:
            #     print('Completed {} repeats...'.format(repeat_count))
            # distractors = np.delete(row, i)
            distances = [row[ii] for ii in np.random.permutation(len(row))[:n]]
            # distances = real_distance + distractor_distance

            if metric == 'pcc':
                # print('PCC Evaluation...')
                # Here we include [::-1] to flip the order of the argsort becasue we want the highest PCC unlike LPIPS
                # results_recon_rank.append(np.argwhere(np.argsort(distances)[::-1] == 0).flatten()[0] / (len(distances) - 1))
                results_recon.append(np.argsort(distances)[::-1][0] == 0)
            else:
                # results_recon_rank.append(np.argwhere(np.argsort(distances) == 0).flatten()[0] / (len(distances) - 1))
                results_recon.append(np.argsort(distances)[0] == 0)

        # results_net_rank.append(np.mean(results_recon_rank))
        results_net.append(np.mean(results_recon))

    print('Overall accuracy is {}'.format(np.mean(results_net)))

    # Get accuracy per recon
    # accuracy = total_score / repeat_count * 100
    # print('Accuracy rate for repeat {} is {:.2f}%'.format(repeat_count, accuracy))

    # Concat this to full list for all recons
    # accuracy_full.append(accuracy)

    # print('Average n-way ({}) comparison accuracy: {:.2f} \n'
    #       'Standard deviation of n-way ({}) comparison accuracy: {:.2f}'.format(n, statistics.mean(accuracy_full),
    #                                                                n, statistics.stdev(accuracy_full)))

    return results_net  # , results_recon


def pairwise_comp(df, metric="pcc"):
    # create container
    accuracy_full = []

    # create counter (for each reconstruction)
    trials = 0

    # pull rows from full table
    for i, row in df.iterrows():
        trials += 1
        if not trials % 100:
            print('Completed {} evaluations...'.format(trials))
        # set score (count of wins for real vs each pairwise comp)
        score = 0

        # counts comparisons per row
        row_count = 0
        # print('Comparison {}'.format(trials))

        # convert to df
        row = row.to_frame()

        # grab real candidate value
        matched = row.loc[i]
        # print('Value of matched is: ', matched)

        # remove candidate from rest of row values
        new_row = row.drop([i])

        # pull value for each remaining comparison
        for idx, comparison in new_row.iterrows():
            row_count += 1
            if metric == "lpips":
                # because lpips lower is better
                if matched.item() < comparison.item():
                    score += 1
            else:
                # for pcc and ssim
                if matched.item() > comparison.item():
                    score += 1

        # calculate accuracy per reconstruction
        accuracy = score / row_count * 100
        # print('Recon of {} accuracy is {:.2f}'.format(i, accuracy))

        # adds accuracy to full list
        accuracy_full.append(accuracy)

    print('Average pairwise accuracy: {:.2f} \n'
          'Standard deviation of pairwise accuracy: {:.2f}'.format(statistics.mean(accuracy_full),
                                                                   statistics.stdev(accuracy_full)))

    return accuracy_full


def load_masters(master_root, comparison="nway"):
    out_dict = {}

    # Load data
    print('Reading Data')

    if comparison == "pairwise":
        pcc_dir = os.path.join(master_root, 'pcc_master_pairwise_out.csv')
        lpips_dir = os.path.join(master_root, 'lpips_master_pairwise_out.csv')

        pcc_master = pd.read_csv(pcc_dir, index_col=0)
        lpips_master = pd.read_csv(lpips_dir, index_col=0)

        out_dict['pcc'] = pcc_master
        out_dict['lpips'] = lpips_master
        # pcc_columns = pcc_master.loc[,]

    else:  # "nway"
        out_dict['pcc'] = {}
        out_dict['lpips'] = {}

        ns = [2, 5, 10]

        for n in ns:
            way_label = "{}-way_comparison".format(n)

            # Load PCC
            pcc_master = pd.read_excel(os.path.join(master_root, 'pcc_master_pairwise_out.xlsx'), sheet_name=way_label,
                                       engine='openpyxl', index_col=0, header=0)
            out_dict['pcc'][way_label] = pcc_master

            # Load LPIPS
            lpips_master = pd.read_excel(os.path.join(master_root, 'lpips_master_pairwise_out.xlsx'), sheet_name=way_label,
                                       engine='openpyxl', index_col=0, header=0)
            out_dict['lpips'][way_label] = lpips_master

    return out_dict


def eval_grab(master, networks, comparison='pairwise'):
    # For each name in list, grab each column from both metrics and each comparison. FUCK.
    # metrics = ['pcc', 'lpips']
    grab_out = {}

    if comparison == "pairwise":
        print('pairwise')
        # Grab columns from list for each
        pcc_grab = master['pcc'].loc[:, networks]
        lpips_grab = master['lpips'].loc[:, networks]

        if len(networks)==1:
            print('Loner.')
            grab_out['pcc'] = pcc_grab
            grab_out['lpips'] = lpips_grab

        else:
            # Stacks columns into one long column
            pcc_stack = pcc_grab.unstack().reset_index(drop=True)
            lpips_stack = lpips_grab.unstack().reset_index(drop=True)

            grab_out['pcc'] = pcc_stack
            grab_out['lpips'] = lpips_stack

        return grab_out

        # grab_out['pcc'] = master.loc[:, networks]

    else:
        grab_out['pcc'] = {}
        grab_out['lpips'] = {}

        # but also check for length
        if len(networks) > 1:
            single = False

        # for network in networks:
            # e.g. "Study1_SUBJ01_1pt8mm_VC_max"
            # TODO: I don't think I need to iterate through networks, I can select single or multiple with the list
        for metric in master:
            # key is pcc/lpips, value is another dictionary with n way as key, dataframe as value
            for nway, dataframe in master[metric].items():
                columns = dataframe[networks]
                # POGs in the chat :D

        return columns  # dict {PCC=list; LPIPS=list}
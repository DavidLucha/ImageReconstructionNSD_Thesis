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
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms as transforms

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
        path = self.dataset[idx]['image']

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
                    torchvision.utils.save_image(im, fp=recon_path + '/' + str(batch_idx * len(data_target) + i) + '.png', normalize=True)
                for i, im in enumerate(data_target):
                    torchvision.utils.save_image(im, fp=real_path + '/' + str(batch_idx * len(data_target) + i) + '.png', normalize=True)
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


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):

    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py
    """
    from torchvision.models.inception import inception_v3
    import numpy as np
    from scipy.stats import entropy

    N = len(imgs)

    # assert batch_size > 0
    # assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores)


def objective_assessment(model, dataloader, top=5, save_path="D:/Lucha_Data/misc/"):
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

    # [PCC, SSIM]
    true_positives = torch.tensor([0, 0, 0])
    dataset_size = 0
    score_pcc = 0
    score_ssim = 0
    score_lpips = 0
    # an
    lpips_sum = 0

    # Make a results var to save individual image metrics
    results = dict(
        trial_id=[],
        image_path=[],
        # mse_im=[],
        ssim_im=[],
        pcc_im=[],
        lpips_im=[]
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

                    # Perceptual Similarity Metric
                    # TODO: check if it's normalized
                    image_cpu = image.data.cpu()
                    score_rand_lpips = perceptual_similarity(image_cpu, target_cpu[rand_idx])
                    score_recon_lpips = perceptual_similarity(image_cpu, target_cpu[idx])
                    # Lower number means images are 'closer' together
                    if score_recon_lpips < score_rand_lpips:
                        score_lpips += 1

                    # results['mse_im'].append(score_recon.item())
                # Save individual scores per image
                results['trial_id'].append(idx)
                results['image_path'].append(data_path[idx])
                results['pcc_im'].append(score_recon_pcc.item())
                results['ssim_im'].append(score_recon_ssim.item())
                results['lpips_im'].append(score_recon_lpips.item())

                lpips_sum += score_recon_lpips

                if score_pcc == top - 1:
                    true_positives[0] += 1
                if score_ssim == top - 1:
                    true_positives[1] += 1
                if score_lpips == top - 1:
                    true_positives[2] += 1

                dataset_size += 1
                score_pcc = 0
                score_ssim = 0
                score_lpips = 0

    objective_score = true_positives.float() / dataset_size
    lpips_mean = lpips_sum / dataset_size

    results_to_save = pd.DataFrame(results)
    results_to_save.to_csv(os.path.join(save_path, "trial_results.csv"), index=True)

    return objective_score[0], objective_score[1], objective_score[2], lpips_mean


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




# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:23:16 2022

Created to try and solve the issue regarding multiple optimizers

So, this now works - but I'm rerunning the gen and discrim-
Each time I calculate a new error term and use the optimizer zero_grad

In this version, I have recalculated the required bits each time calculating new error.
In other words, this should work, but isn't ideal IMO

@author: david
"""

"""
IMPORTS
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random

from torchvision import datasets
from torchvision.utils import make_grid , save_image
from torch.autograd import Variable

# Set random seed for reproducibility
manualSeed = 123
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

"""
RESOLVE SSL Issue
"""

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


"""
Determine if any GPUs are available
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)



"""
GLOBAL PARAMS
"""
# Global constants
# filter_size : kernal w and h
# bs : batch size
# img_size : image h and w
# conv_count : number of 2-stride conv layers in Encoder
# conv_img_size : size of image dimensions after number of convs
# hl_dim : largest convolutional channel depth
# lin_neurons : input of first linear layer after convolutions.

filter_size = 5
img_channels = 3
img_size = 32
latent_dim = 128
hl_dim = 256

# Convolution calculations for visual encoder
conv_count = range(3) # Count of convolution layers with stride 2
conv_img_size = img_size # Input H/W at time=0
for i in conv_count:
  conv_img_size = (conv_img_size + 2 // 2) // 2 # Calculates enc_conv output and rounds to int after each conv.
print('Output size after final visual encoder convolution will be: ',conv_img_size)
lin_neurons = hl_dim * conv_img_size * conv_img_size # Calculates the number of outputs of final convolution for linear layer.

# Convolution calculations for discriminator
disc_conv_count = range(4)
disc_conv_img_size = img_size
for i in disc_conv_count:
  disc_conv_img_size = (disc_conv_img_size + 2 // 2) // 2 # Calculates enc_conv output and rounds to int
print('Output size after final disciminator convolution will be: ',disc_conv_img_size)
disc_lin_neurons = hl_dim * disc_conv_img_size * disc_conv_img_size

cog_vector_length = img_size**2 #**** INSERT HERE **** - FMRI Length | Currently just set to 32 x 32 img size for testing distillation


"""
METHODS
"""

def show_and_save(file_name,img):
    npimg = np.transpose(img.numpy(),(1,2,0)) # This might be the only thing we need to change | Tranpose flips the axes of array - 3d vector
    f = "./%s.png" % file_name
    fig = plt.figure(dpi=200)
    fig.suptitle(file_name, fontsize=14, fontweight='bold')
    plt.imshow(npimg)
    plt.imsave(f,npimg)

def plot_loss(loss_list):
    plt.figure(figsize=(10,5))
    plt.title("Loss During Training")
    plt.plot(loss_list,label="Loss")
    
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()






"""
MODELS
"""
"""
Weights Init function 

Currently not sure how this plays into things. Called in the VAE/GAN class in the apply function.
"""
def weights_init(self):
    classname = self.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(self.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(self.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bias.data, 0)

class Vis_Enc(nn.Module): #VAE
    def __init__(self):
        super(Vis_Enc, self).__init__()

        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1=nn.Conv2d(img_channels,64,filter_size,padding=2,stride=2)   #in_channels=3 (was one) and 30x30 images | padding is 2 to ensure the input matches the output before the compression, without padding image is 32->28 and then stride would take thaat to 14
        self.bn1=nn.BatchNorm2d(64,momentum=0.9)
        self.encConv2=nn.Conv2d(64,128,filter_size,padding=2,stride=2)
        self.bn2=nn.BatchNorm2d(128,momentum=0.9)
        self.encConv3=nn.Conv2d(128,256,filter_size,padding=2,stride=2)
        self.bn3=nn.BatchNorm2d(256,momentum=0.9)

        # Define relu
        self.relu=nn.LeakyReLU(0.2)

        # FC layers
        self.fc1=nn.Linear(lin_neurons,1024) # WAS 1024has to do about the size of the image given the convolutions
        self.bn4=nn.BatchNorm1d(1024,momentum=0.9)

        # Split variational layers for mean and sd
        self.fc_mean=nn.Linear(1024,latent_dim)
        self.fc_logvar=nn.Linear(1024,latent_dim)   #latent dim=128

    def forward(self, x):
        # print(x.size())
        out=self.relu(self.bn1(self.encConv1(x)))
        # print(out.size())
        out=self.relu(self.bn2(self.encConv2(out)))
        # print(out.size())
        out=self.relu(self.bn3(self.encConv3(out)))
        # print(out.size())
        out=out.view(batch_size,-1)
        # print(out.size())
        out=self.relu(self.bn4(self.fc1(out)))
        mean=self.fc_mean(out)
        logvar=self.fc_logvar(out)

        return mean,logvar
    
class Decoder(nn.Module): #GAN Reconstruction
    def __init__(self):
        super(Decoder,self).__init__()

        #Build Model
        self.de_fc1=nn.Linear(latent_dim, lin_neurons)
        self.de_bn1=nn.BatchNorm1d(lin_neurons,momentum=0.9)

        # Define relu
        self.de_relu=nn.LeakyReLU(0.2)

        # Deconvolutions
        self.deconv1=nn.ConvTranspose2d(256,128,6, stride=2, padding=2) 
        self.de_bn2=nn.BatchNorm2d(128,momentum=0.9)
        self.deconv2=nn.ConvTranspose2d(128,64,6, stride=2, padding=2) 
        self.de_bn3=nn.BatchNorm2d(64,momentum=0.9)
        self.deconv3=nn.ConvTranspose2d(64,32,6, stride=2, padding=2)
        self.de_bn4=nn.BatchNorm2d(32,momentum=0.9)
        self.deconv4=nn.ConvTranspose2d(32,img_channels,5, stride=1, padding=2) # This isn't part of Ren's design
        self.tanh=nn.Tanh()
        
    def forward(self,x):
        # print('before conv:', x.size())
        x=self.de_relu(self.de_bn1(self.de_fc1(x)))
        # print('after lin', x.size())
        x=x.view(-1,256,conv_img_size,conv_img_size)
        # print('view change', x.size())
        x=self.de_relu(self.de_bn2(self.deconv1(x)))
        # print('after conv 1', x.size())
        x=self.de_relu(self.de_bn3(self.deconv2(x)))
        # print('after conv 2', x.size())
        x=self.de_relu(self.de_bn4(self.deconv3(x)))
        # print('after conv 3', x.size())
        x=self.tanh(self.deconv4(x))
        # print('after conv 4', x.size())
        return x


class Discriminator(nn.Module): #GAN Discrimination
    def __init__(self):
        super(Discriminator,self).__init__()

        #Build model
        self.di_conv1=nn.Conv2d(img_channels,64, filter_size,padding=2,stride=2) #50*50*64
        self.di_bn1=nn.BatchNorm2d(64,momentum=0.9)
        self.di_conv2=nn.Conv2d(64,128,filter_size,padding=2,stride=2) #25*25*128
        self.di_bn2=nn.BatchNorm2d(128,momentum=0.9)
        self.di_conv3=nn.Conv2d(128,256,filter_size,padding=2,stride=2) #13*13*256
        self.di_bn3=nn.BatchNorm2d(256,momentum=0.9)
        self.di_conv4=nn.Conv2d(256,256,filter_size,padding=2,stride=2) #7*7*256
        self.di_bn4=nn.BatchNorm2d(256,momentum=0.9)

        # Define relu
        self.di_relu=nn.LeakyReLU(0.2)

        # Fully connected layers
        self.di_fc1=nn.Linear(disc_lin_neurons,256)
        self.di_bn5=nn.BatchNorm1d(256,momentum=0.9)
        self.di_fc2=nn.Linear(256,1)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        # print(x.size())
        x=self.di_relu(self.di_bn1(self.di_conv1(x)))
        # print('after conv 1', x.size())
        x=self.di_relu(self.di_bn2(self.di_conv2(x)))
        # print('after conv 2', x.size())
        x=self.di_relu(self.di_bn3(self.di_conv3(x)))
        # print('after conv 3', x.size())
        x=self.di_relu(self.di_bn4(self.di_conv4(x)))
        # print('after conv 4', x.size())
        x=x.view(-1,256*2*2)
        # print(x.size())
        x1=x;
        x=self.di_relu(self.di_bn5(self.di_fc1(x)))
        # print(x.size())
        x=self.sigmoid(self.di_fc2(x))
        # print(x.size())
        return x,x1

class VE_VAE_GAN(nn.Module): # Enc and Dec
  def __init__(self):
    super(VE_VAE_GAN,self).__init__()
    self.vis_enc=Vis_Enc()
    self.decoder=Decoder()
    self.vis_enc.apply(weights_init)
    self.decoder.apply(weights_init)

  def forward(self,x):
    z_mean,z_logvar=self.vis_enc(x) 
    
    # Reparameterise | See https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71
    std = torch.exp(z_logvar/2) # Converting logvar to std 
    epsilon = torch.randn_like(std) # Random numbers from normal distribution (0,1)
    z = z_mean+std*epsilon
    
    # Push Reparameterised variable z through decoder
    x_tilda=self.decoder(z) 
      
    return z_mean,z_logvar,x_tilda



"""
DATALOADER
Create dataloaders to feed data into the neural network
Default CIFAR10 dataset is used and standard train/test split is performed
"""

def dataloader(batch_size):
    # Set root location
    dataroot='/data'

    # Define transform
    transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    
    # Define dataset
    train_set = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform)

    # Initialize DataLoader
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    return data_loader

"""
Set up non-random dataloader
"""
def nr_dataloader(batch_size):
    # Set root location
    dataroot='/data'

    # Define transform
    transform=transforms.Compose([
                               transforms.Resize(img_size),
                               transforms.CenterCrop(img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    
    # Define dataset
    train_set = torchvision.datasets.CIFAR10(root=dataroot, train=True, download=True, transform=transform)

    # Initialize DataLoader
    nr_data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=8)
    return nr_data_loader

"""
Establish set dataloader for testing
"""
# Initiliaze non-random test batch
nr_data_loader = nr_dataloader(64)
test_batch = next(iter(nr_data_loader))
test_fixed=Variable(test_batch[0]).to(device)




"""
Training VAE-GAN for CIFAR10 Dataset

Adapted from  https://github.com/DavidLucha/VAE-GAN-PYTORCH/blob/master/models.py & 
              https://towardsdatascience.com/building-a-convolutional-vae-in-pytorch-a0f54c947f71 &
              https://realpython.com/generative-adversarial-networks/ &
              Ren et al., 2021

Set up to mimic Training Stage 1 of Ren dVAE/GAN implementation (without the dual encoder component).
"In the first stage, the visual stimuli are used as input to jointly train the [encoder], [generator] and [GAN discriminator]. 
In this stage, we perform a reconstruction task for learning the latent visual representations. 
Specifically, [encoder] is optimized to encode the visual stimulus X to the latent visual representation Z.
Then the [decoder] is trained to reconstruct X given Z. Meanwhile, [discriminator] is trained to discriminate real and reconstructed stimuli."
"""

##########################################################
torch.autograd.set_detect_anomaly(True)

""" Pull data from CIFAN10 with a batch size of n """
batch_size = 64
data_loader=dataloader(batch_size) # Was 64, changed to 16
# Note: The GitHub VAE-GAN implementation doesn't split the train and test data.

""" Initialize the networks """
gen=VE_VAE_GAN().to(device)
discrim=Discriminator().to(device)
discrim.apply(weights_init)

""" Load the real batch and visualise """ # I think.
real_batch = next(iter(data_loader))
show_and_save("training" ,make_grid((real_batch[0]*0.5+0.5).cpu(),8))

""" Setting Hyperparameters """
epochs=10 # They do 400 epochs
lr=3e-3 # As defined by learning rate for Stage 1 (Ren)
decay_rate=0.98 # Unclear whether this is all stages or just third.

loss_gamma=15 # See below
"""
Regarding Gamma:
"Since" Dec receives an error signal from both LDislllike and LGAN, 
we use a parameter γ [Gamma] to weight the ability to reconstruct vs. fooling the discriminator. 
This can also be interpreted as weighting style and content. 
"""

""" Initialize the loss function, optimizers and fixed variables """
loss_function=nn.BCELoss().to(device) # Was BCELoss This is establishing the loss function, we would need to change this to whatever we think is suitable

# They set beta of Adam optimizers to 0.9 - where do we define this?
optimizer_vis_encoder=torch.optim.Adam(gen.vis_enc.parameters(), lr=lr) 
optimizer_decoder=torch.optim.Adam(gen.decoder.parameters(), lr=lr)
optimizer_discriminator=torch.optim.Adam(discrim.parameters(), lr=lr) 


# Initialize schedulers to update learning rate each epoch
scheduler_VE = torch.optim.lr_scheduler.ExponentialLR(optimizer_vis_encoder, gamma=decay_rate)
scheduler_de = torch.optim.lr_scheduler.ExponentialLR(optimizer_decoder, gamma=decay_rate)
scheduler_di = torch.optim.lr_scheduler.ExponentialLR(optimizer_discriminator, gamma=decay_rate)
# print('Starting learing rate is: ', scheduler_VE.get_last_lr())


# Fixed variables are only used at the end of for loop
z_fixed=Variable(torch.randn((batch_size,128))).to(device) # Z refers to latent space
x_fixed=Variable(real_batch[0]).to(device) # X refers to the input


for epoch in range(epochs):
  # Initialise empty arrays for loss variables
  prior_loss_list,gan_loss_list,reconstruction_loss_list=[],[],[]
  discriminator_real_list,discriminator_fake_list,discriminator_prior_list=[],[],[]

  # Print learning rate at start of epoch
  # print('Learning rate at epoch', epoch, 'is: ', scheduler_VE.get_last_lr())

  for i, (data,_) in enumerate(data_loader, 0):
    # Set batch size
    bs=data.size()[0] 
    
    # Set up data variable
    datav = Variable(data).to(device) # Pulls the data variable from the for loop to push into network

    """
    RUN MODEL
    """
    # Feeding a batch of images into the network to obtain the output, mean and logvar
    mean, logvar, rec_enc = gen(datav) 
    
    # Run Discriminators
    disc_x_real, disc_x1_real = discrim(datav) # Real (x,x1)
    disc_x_tilda, disc_x1_tilda = discrim(rec_enc) # Reconstruction
    
    # Empty label generation for discriminator
    ones_label=Variable(torch.ones(bs,1)).to(device)
    zeros_label=Variable(torch.zeros(bs,1)).to(device)
    
    """
    LOSS FUNCTIONS
    """
    """
    ENCODER
    """
    # L_prior_v
    # Regularize encoder using KL divergence between z|x and prior (0,I)
    # Updates Vis_Enc
    loss_KL = -0.5* torch.mean( 1.0 + logvar - mean.pow(2.0) - logvar.exp() )
    L_prior_v = loss_KL
    prior_loss_list.append(L_prior_v.item())
    # GOOD
    
    # L_gan_rec_v (i.e., reconstruction loss from GAN discriminator)
    # Guassian observation mdoel from Larsen et al. (2016)
    # Updates all parameters
    disc_x1_real = discrim(datav)[1]
    disc_x1_tilda = discrim(rec_enc)[1]
    rec_loss = ((disc_x1_tilda - disc_x1_real) ** 2).mean()
    L_gan_rec_v = rec_loss
    reconstruction_loss_list.append(rec_loss.item())
    # GOOD
    
    # FIX FOR ENC_ERR ##############
    Enc_Err = L_prior_v + L_gan_rec_v
    # Update encoder parameters
    optimizer_vis_encoder.zero_grad() # Set gradients to zero
    Enc_Err.backward(retain_graph=True) # Calculate derivative
    optimizer_vis_encoder.step() # Update the optimizer
    
    """
    DECODER
    """
    # FIX - RECALCULATE L_GAN_REC_V ###############
    mean, logvar, rec_enc = gen(datav) 
    disc_x1_real = discrim(datav)[1]
    disc_x1_tilda = discrim(rec_enc)[1]
    # L_gan_rec_v
    rec_loss = ((disc_x1_tilda - disc_x1_real) ** 2).mean()
    L_gan_rec_v = rec_loss
    reconstruction_loss_list.append(rec_loss.item())
    # FIX UPDATE DECODER ###########
    Dec_Err = L_gan_rec_v
    # Update decoder parameters
    optimizer_decoder.zero_grad()
    Dec_Err.backward(retain_graph=True) 
    optimizer_decoder.step() 
    
    """
    DISCRIMINATOR
    """
    mean, logvar, rec_enc = gen(datav) 
    # L_gan
    disc_x_real = discrim(datav)[0]
    Disc_Err_Real = loss_function(disc_x_real, ones_label) 
    discriminator_real_list.append(Disc_Err_Real.item())
    
    disc_x_tilda = discrim(rec_enc)[0]
    # Updates discriminator
    Disc_Err_Rec = loss_function(disc_x_tilda, zeros_label)
    discriminator_fake_list.append(Disc_Err_Rec.item())

    L_gan = Disc_Err_Real + Disc_Err_Rec
    gan_loss_list.append(L_gan.item())
    
    # FIX DISCRIMINATOR #########
    Disc_Err = L_gan
    # Update discriminator parameters
    optimizer_discriminator.zero_grad() 
    Disc_Err.backward(retain_graph=True) 
    optimizer_discriminator.step() 
    

    ########
    """
    # Update parameters
    # Calculate combined errors
    # Note: In Larsen paper, L_gan_rec_v always has a weighting (either gamma:15, or 5)
    Enc_Err = L_prior_v + L_gan_rec_v
    print('13')
    # Even here, it's unclear how the 'jointly update' applies.
    Dec_Err = L_gan_rec_v
    print('14')
    Disc_Err = L_gan
    print('15')
    # Text makes it seem like Disc_Err is just L_gan
    # But tripple criterion makes it seem like L_gan_rec_v is involved too 
    
    
    # Update encoder parameters
    optimizer_vis_encoder.zero_grad() # Set gradients to zero
    print('16')
    Enc_Err.backward(retain_graph=True) # Calculate derivative
    print('17')
    optimizer_vis_encoder.step() # Update the optimizer
    print('18')
    
    # Update decoder parameters
    optimizer_decoder.zero_grad()
    print('19')
    Dec_Err.backward(retain_graph=True) 
    print('20')
    optimizer_decoder.step() 
    print('21')
    
    # Update discriminator parameters
    optimizer_discriminator.zero_grad() 
    print('22')
    Disc_Err.backward(retain_graph=True) 
    print('23')
    optimizer_discriminator.step() 
    print('24')
    """
    
    """ Print loss Statistics """
    if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_gan: %.4f\tLoss_prior: %.4f\tReconstruction_loss: %.4f\tdis_real_loss: %0.4f\tdis_fake_loss: %.4f'
                  % (epoch,epochs, i, len(data_loader),
                     L_gan.item(), L_prior_v.item(),rec_loss.item(),Disc_Err_Real.item(),Disc_Err_Rec.item()))

  
   
   
   
   
    
    
  


  """
  Note re: detach() below:
  tensor.detach() creates a tensor that shares storage with tensor that does not require grad. 
  It detaches the output from the computational graph. 
  So, no gradient will be backpropagated along this variable.
  From: https://stackoverflow.com/questions/56816241/difference-between-detach-and-with-torch-nograd-in-pytorch
  """  

  b=gen(x_fixed)[2]
  b=b.detach()
  c=gen.decoder(z_fixed)
  c=c.detach()
  scheduler_VE.step()
  scheduler_de.step()
  scheduler_di.step()
  show_and_save('CIFAR10rec_noise_epoch_%d.png' % epoch ,make_grid((c*0.5+0.5).cpu(),8))
  show_and_save('CIFA10rec_epoch_%d.png' % epoch ,make_grid((b*0.5+0.5).cpu(),8))

plot_loss(prior_loss_list)
plot_loss(reconstruction_loss_list)
plot_loss(gan_loss_list)



"""
PRINT STATES
"""
"""
print('after training, gen state is: ', gen.state_dict())
print('after training, discrim (should be trained) state is: ', discrim.state_dict())
print('after training, gen.discrim (shouldnt be trained) state is: ', gen.discriminator.state_dict())
print('after training, gen decoder state is: ', gen.decoder.state_dict())
print('after training, gen encoder state is: ', gen.vis_enc.state_dict())
"""


"""
TEST THE MODEL
"""
# Initiliaze non-random test batch
# nr_data_loader = nr_dataloader(64)
# test_batch = next(iter(nr_data_loader))
# test_fixed=Variable(test_batch[0]).to(device)

# Run Test
test_gen=gen(test_fixed)[2]
test_gen=test_gen.detach()
test_rec_GAN=discrim(test_gen)
test_real_GAN=discrim(test_fixed)

# show_and_save("training" ,make_grid((real_batch[0]*0.5+0.5).cpu(),8))
show_and_save('Training Real.png',make_grid((test_fixed*0.5+0.5).cpu(),8))
show_and_save('Complete training reconstruction.png',make_grid((test_gen*0.5+0.5).cpu(),8))
print('GAN Output for Reconstruction: ',test_rec_GAN)
print('GAN Output for Real: ',test_real_GAN)


"""
CLEAR MEM
import gc

gc.collect()

torch.cuda.empty_cache()
"""


"""
RESET NETWORK
"""
"""
import collections
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d) or isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.BatchNorm2d):
        m.reset_parameters()

gen.apply(weight_reset)
discrim.apply(weight_reset)
gen.decoder.apply(weight_reset)

optimizer_vis_encoder.state = collections.defaultdict(dict)
optimizer_decoder.state = collections.defaultdict(dict)
optimizer_discriminator.state = collections.defaultdict(dict)
"""







"""
SAVE NETWORK
"""

PATH = "CIFAR_VAEGAN_Stage_I.pt"

torch.save({
            'gen_visenc_state_dict': gen.vis_enc.state_dict(),
            'gen_decoder_state_dict': gen.decoder.state_dict(),
            'discrim_state_dict': discrim.state_dict(),
            'optimizer_visenc_state_dict': optimizer_vis_encoder.state_dict(),
            'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
            'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
            }, PATH)
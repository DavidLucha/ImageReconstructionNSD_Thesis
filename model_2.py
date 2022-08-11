import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
import training_config
import model_config_2 as config

from torch.autograd import Variable
from utils_2 import NLLNormal


class EncoderBlock(nn.Module):
    """
    Encoder block used in encoder and discriminator
    """
    def __init__(self, channel_in, channel_out):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=config.kernel_size,
                              padding=config.padding, stride=config.stride,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

    def forward(self, ten, out=False, t=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = F.relu(ten, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.relu(ten, True)
            return ten


class DecoderBlock(nn.Module):
    """
    Decoder block used in decoder
    """
    def __init__(self, channel_in, channel_out, out=False):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        if out:
            self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=config.kernel_size, padding=config.padding,
                                           stride=config.stride, output_padding=1,
                                           bias=False)
        else:
            self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=config.kernel_size,
                                           padding=config.padding,
                                           stride=config.stride,
                                           bias=False)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        return ten


class Encoder(nn.Module):
    """
    Visual encoder with parameters from config.
    Used to transform images to visual latent features in training in Stage I
    """
    def __init__(self, channel_in=3, z_size=128):
        super(Encoder, self).__init__()
        self.size = channel_in
        layers_list = []
        # the first time 3->64, for every other double the channel size
        for i in range(3):  # [0, 1, 2]
            layers_list.append(EncoderBlock(channel_in=self.size, channel_out=config.encoder_channels[i]))
            self.size = config.encoder_channels[i]

        # final shape Bx256x8x8
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(nn.Linear(in_features=config.fc_input * config.fc_input * self.size,
                                          out_features=config.fc_output, bias=False),
                                nn.BatchNorm1d(num_features=config.fc_output, momentum=0.9),
                                nn.ReLU(True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=config.fc_output, out_features=z_size)
        self.l_var = nn.Linear(in_features=config.fc_output, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        logvar = self.l_var(ten)
        return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class Decoder(nn.Module):
    """
    Decoder with parameters from config.
    Used to transform latent features (visual or cognitive) to images
    """
    def __init__(self, z_size, size):
        super(Decoder, self).__init__()
        # Code to swap between Maria channel config, and Ren paper
        channels = config.maria_decoder_channels # Can change the config here 'decoder_channels'/'maria_decoder_channels'
        self.size = size  # 256

        # start from B*z_size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size, out_features=config.fc_input * config.fc_input * size, bias=False),
                                nn.BatchNorm1d(num_features=config.fc_input * config.fc_input * size, momentum=0.9),
                                nn.ReLU(True))
        layers_list = []
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=channels[0], out=config.output_pad_dec[0]))
        self.size = channels[0]
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=channels[1], out=config.output_pad_dec[1]))
        self.size = channels[1]
        layers_list.append(DecoderBlock(channel_in=self.size, channel_out=channels[2], out=config.output_pad_dec[2]))
        self.size = channels[2]
        # final conv to get 3 channels and tanh layer
        # TODO: Look into this. Interesting.
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=channels[3], kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, config.fc_input, config.fc_input)
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


class Discriminator(nn.Module):
    """
    Discriminator with parameters from config. Used in VAE/GAN
    """
    def __init__(self, channel_in=3, recon_level=3):
        super(Discriminator, self).__init__()
        self.size = channel_in
        self.recon_levl = recon_level
        channels = config.maria_discrim_channels  # Can change channel config here
        # module list because we need need to extract an intermediate output
        self.conv = nn.ModuleList()
        self.conv.append(nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=5, stride=config.stride_gan, padding=2),
            nn.ReLU(inplace=True)))
        self.size = channels[0]
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=channels[1]))
        self.size = channels[1]
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=channels[2]))
        self.size = channels[2]
        self.conv.append(EncoderBlock(channel_in=self.size, channel_out=channels[3]))
        # final fc to get the score (real or fake)
        self.fc = nn.Sequential(
            nn.Linear(in_features=config.fc_input_gan * config.fc_input_gan * self.size,
                      out_features=config.fc_output_gan, bias=False),
            nn.BatchNorm1d(num_features=config.fc_output_gan, momentum=0.9),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=config.fc_output_gan, out_features=1),
        )

    def forward(self, ten_orig, ten_predicted, ten_sampled, mode='REC'):
        if mode == "REC":
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                # we take the 9th layer as one of the outputs
                # print('I am layer {}: {} ... of the discriminator'.format(i, lay))
                if i == self.recon_levl:
                    ten, layer_ten = lay(ten, True)
                    # we need the layer representations just for the original and reconstructed,
                    # flatten, because it's a convolutional shape
                    # layer_ten = layer_ten.view(len(layer_ten), -1)
                    # We've removed this because Ren just takes the conv.
                    # TODO: Check this
                    return layer_ten
                else:
                    ten = lay(ten)
        else:
            ten = torch.cat((ten_orig, ten_predicted, ten_sampled), 0)
            for i, lay in enumerate(self.conv):
                ten = lay(ten)

            ten = ten.view(len(ten), -1)
            ten = self.fc(ten)
            # removing the sigmoid and using the BCElogits loss (more stable)
            return ten
            # return ten
        # raise Exception('testing discrimintator')

    def __call__(self, *args, **kwargs):
        return super(Discriminator, self).__call__(*args, **kwargs)


class CognitiveEncoder(nn.Module):
    """
    Cognitive encoder to transform fMRI to cognitive latent representations.
    Used in training on Stage II and III
    """
    def __init__(self, input_size, z_size=128, channel_in=3, lin_size=1024, lin_layers=1):
        super(CognitiveEncoder, self).__init__()
        self.size = channel_in
        self.lin_layers = lin_layers

        if self.lin_layers == 1:
            self.fc1 = nn.Sequential(nn.Linear(in_features=input_size, out_features=lin_size, bias=False),
                                    nn.BatchNorm1d(num_features=lin_size, momentum=0.9),
                                    nn.ReLU(True))
            # self.fc2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512, bias=False),
            #                         nn.BatchNorm1d(num_features=512, momentum=0.9),
            #                         nn.LeakyReLU(True))
            # two linear to get the mu vector and the diagonal of the log_variance
            self.l_mu = nn.Linear(in_features=lin_size, out_features=z_size)
            self.l_var = nn.Linear(in_features=lin_size, out_features=z_size)
        else:
            self.fc1 = nn.Sequential(nn.Linear(in_features=input_size, out_features=lin_size, bias=False),
                                     nn.BatchNorm1d(num_features=lin_size, momentum=0.9),
                                     nn.ReLU(True))
            self.fc2 = nn.Sequential(nn.Linear(in_features=lin_size, out_features=int(lin_size/2), bias=False),
                                     nn.BatchNorm1d(num_features=int(lin_size/2), momentum=0.9),
                                     nn.ReLU(True))
                                     # was nn.LeakyReLU(True))
            # two linear to get the mu vector and the diagonal of the log_variance
            self.l_mu = nn.Linear(in_features=int(lin_size/2), out_features=z_size)
            self.l_var = nn.Linear(in_features=int(lin_size/2), out_features=z_size)
    #     self.init_parameters()
    #
    # def init_parameters(self):
    #     # just explore the network, find every weight and bias matrix and fill it
    #     for m in self.modules():
    #         if isinstance(m, (nn.Linear)):
    #             if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
    #                 # init as original implementation
    #                 # scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
    #                 # scale /= numpy.sqrt(3)
    #                 nn.init.xavier_normal_(m.weight, 1)
    #                 # nn.init.constant(m.weight, 0.005)
    #                 # nn.init.uniform_(m.weight, -scale, scale)
    #             if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
    #                 nn.init.constant_(m.bias, 0.0)

    def forward(self, ten):
        if self.lin_layers == 1:
            ten = self.fc1(ten)
            # ten = self.fc2(ten)
            mu = self.l_mu(ten)
            logvar = self.l_var(ten)
            return mu, logvar
        else:
            ten = self.fc1(ten)
            ten = self.fc2(ten)
            mu = self.l_mu(ten)
            logvar = self.l_var(ten)
            return mu, logvar

    def __call__(self, *args, **kwargs):
        return super(CognitiveEncoder, self).__call__(*args, **kwargs)


class VaeGan(nn.Module):
    """
    VAE/GAN model, which is used in training on Stage I: image-to-image translation.
    Modified from https://github.com/lucabergamini/VAEGAN-PYTORCH
    """
    def __init__(self, device, z_size=128, recon_level=3):
        super(VaeGan, self).__init__()
        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size).to(device)
        # self.encoder = ResNetEncoder(z_size=self.z_size).to(device)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size).to(device)
        self.discriminator = Discriminator(channel_in=3, recon_level=recon_level).to(device)
        # self-defined function to init the parameters
        self.init_parameters()
        self.device = device

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation # TODO: Check this
                    scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /= numpy.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x, gen_size=10):
        if x is not None:
            x = Variable(x).to(self.device)
            if self.training:
                mus, log_variances = self.encoder(x)
                z = self.reparameterize(mus, log_variances)
                x_tilde = self.decoder(z)

                z_p = Variable(torch.randn(len(x), self.z_size).to(self.device), requires_grad=True)
                x_p = self.decoder(z_p)

                disc_layer = self.discriminator(x, x_tilde, x_p, "REC")  # discriminator for reconstruction
                disc_class = self.discriminator(x, x_tilde, x_p, "GAN")

                return x_tilde, disc_class, disc_layer, mus, log_variances
            else:
                mus, log_variances = self.encoder(x)
                z = self.reparameterize(mus, log_variances)
                x_tilde = self.decoder(z)
                return x_tilde
        else:
            z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)  # just sample and decode
            x_p = self.decoder(z_p)
            return x_p


    def __call__(self, *args, **kwargs):
        return super(VaeGan, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(x, x_tilde, hid_dis_real, hid_dis_pred, hid_dis_sampled, fin_dis_real, fin_dis_pred, fin_dis_sampled, mus, variances):

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5 * (x.view(len(x), -1) - x_tilde.view(len(x_tilde), -1)) ** 2

        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)
        # print("axis then sum", kl, kl.size())
        # print("then, sum that", torch.sum(kl), torch.sum(kl).size())
        # kl_meth_2 = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1)
        # print("sum with no axis", kl_meth_2, kl_meth_2.size())

        # loss_KL = -0.5* torch.mean( 1.0 + logvar - mean.pow(2.0) - logvar.exp() )

        # mse between intermediate layers
        # mse_1 = torch.sum((hid_dis_real - hid_dis_pred) ** 2)
        # The above didn't work as well. Not sure why though.
        # Likely, because things are summed, everything is really finely tuned.
        mse_1 = torch.sum(0.5 * (hid_dis_real - hid_dis_pred) ** 2, 1)
        mse_2 = torch.sum(0.5 * (hid_dis_real - hid_dis_sampled) ** 2, 1)  # NEW

        # bce for decoder and discriminator for original and reconstructed
        bce_dis_original = -torch.log(fin_dis_real + 1e-5) # 1e-3
        bce_dis_predicted = -torch.log(1 - fin_dis_pred + 1e-5)
        bce_dis_sampled = -torch.log(1 - fin_dis_sampled + 1e-5)

        bce_gen_sampled = -torch.log(fin_dis_sampled + 1e-3)  # NEW
        bce_gen_recon = -torch.log(fin_dis_pred + 1e-3)  # NEW
        '''


        bce_gen_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.ones_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_gen_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
        bce_dis_original = nn.BCEWithLogitsLoss(size_average=False)(labels_original,
                                        Variable(torch.ones_like(labels_original.data).cuda(), requires_grad=False))
        bce_dis_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.zeros_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_dis_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.zeros_like(labels_sampled.data).cuda(), requires_grad=False))
        '''
        return nle, kl, mse_1, mse_2, bce_dis_original, bce_dis_predicted, bce_dis_sampled, \
               bce_gen_recon, bce_gen_sampled

    @staticmethod
    def ren_loss(x, x_tilde, mus, log_variances, hid_dis_real, hid_dis_pred, hid_dis_sampled, fin_dis_real, fin_dis_pred,
                 fin_dis_sampled, hid_dis_cog=None, fin_dis_cog=None, stage=1, device='cuda', d_scale=0.25, g_scale=0.625):
        # set Ren params
        # TODO: Switch scale factors on
        # TODO: Test without
        # d_scale_factor = 0.25
        # g_scale_factor = 1 - 0.75 / 2  # 0.625
        d_scale_factor = d_scale
        g_scale_factor = g_scale
        # d_scale_factor = 0
        # g_scale_factor = 0
        BCE = nn.BCELoss(reduction='none').to(device)
        # MSE = nn.MSELoss().to(device)
        # GNLLLoss = torch.nn.GaussianNLLLoss(full=True, reduction='none')

        # NEED NLE, KL, BCE_DIS_ORIGINAL, BCE_DIS_PREDICTED
        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5 * (x.view(len(x), -1) - x_tilde.view(len(x_tilde), -1)) ** 2

        # kl-divergence
        kl = -0.5 * torch.sum(1.0 + log_variances - mus.pow(2.0) - log_variances.exp(), 1) # SUM ISSUE
        # kl_avg should be ...
        # kl_ori = -0.5 * torch.sum(1.0 + log_variances - mus.pow(2.0) - log_variances.exp())
        # kl_1_step = -0.5 * torch.mean(1.0 + log_variances - mus.pow(2.0) - log_variances.exp())
        # kl_latent = (1.0 + log_variances - mus.pow(2.0) - log_variances.exp())
        # print("kl latent:", kl_latent.size())
        # kl = -0.5 * torch.mean(1.0 + log_variances - mus.pow(2.0) - log_variances.exp())

        # bce for decoder and discriminator for original and reconstructed
        bce_dis_original = -torch.log(fin_dis_real + 1e-3)
        bce_dis_predicted = -torch.log(1 - fin_dis_pred + 1e-3)
        bce_dis_sampled = -torch.log(1 - fin_dis_sampled + 1e-3)

        """
        What do we need for Ren:

        z_mean, z_sigma (encoder output) = mus, log_variances | CHECK

        # We have these, but might readjust how these come from the model
        1_x_tilde, De_pro_tilde (hidden, final disc output for vis recon) = hid_dis_pred, fin_dis_pred
        v_x_tilde, G_pro_logits (hidden, final disc output for cog recon) = hid_dis_cog, fin_dis_cog
        1_x, D_pro_logits (hidden, final disc output for real image) = hid_dis_real, fin_dis_real

        KL_Loss | CHECK

        DISCRIMINATOR | Following could be used with:
            nn.BCEWithLogitsLoss(size_average=False)(labels_predicted, Variable(torch.zeros_like(
            labels_predicted.data).cuda(), requires_grad=False))
        D_fake_loss (Cross entropy between 0 labels, and G_pro_logits[fin_dis_cog]) | dis_fake_cog_loss
            nn.BCEWithLogitsLoss(size_average=False)(fin_dis_cog, Variable(torch.zeros_like(
            fin_dis_cog.data).cuda(), requires_grad=False))
        D_real_loss (CE between Label = .75, and D_pro_logits[fin_dis_real] | dis_real_loss
        D_tilde_loss (CE between 0 and De_pro_tilde[fin_dis_pred] | dis_fake_pred_loss
        D2_tilde_loss (CE between L=.75, and De_pro_tilde[fin_dis_pred] | dis_real_pred_loss ... dis2?

        DECODER | Use same BCE with Logits Loss
        G_fake_loss (CE between L=.375(1-G_scale) and G_pro_logits[fin_dis_cog] | dec_fake_cog_loss
        G_tilde_loss (ce between L=.375 and De_pro_tilde[fin_dis_pred]) | dec_fake_pred_loss

        FEATURE LOSS
        LL_loss (NLL of 1_x_tilde[hid_dis_pred] and 1_x[hid_dis_real]) | feature_loss_pred
        LL_loss_2 (NLL of v_x_tilde[hid_dis_cog] and 1_x_tilde[hid_dis_pred] | feature_loss_pred_cog
        LL_loss_v (NLL of v_x_tilde[hid_dis_cog] and 1_x[hid_dis_real]) | feature_loss_cog
        
        REDEFINITIONS:
        bce_dis_original, ->
        bce_dis_predicted, 
        bce_dis_sampled, 
        bce_gen_recon, 
        bce_gen_sampled

        """

        # Stage 1 Loss
        if stage == 1:
            dis_real_loss = BCE(fin_dis_real,
                                Variable((torch.ones_like(fin_dis_real.data) - d_scale_factor).cuda()))
            dis_fake_pred_loss = BCE(fin_dis_pred, Variable(torch.zeros_like(fin_dis_pred.data).cuda()))
            dis_fake_sampled_loss = BCE(fin_dis_sampled, Variable(torch.zeros_like(fin_dis_sampled.data).cuda()))
            dec_fake_pred_loss = BCE(fin_dis_pred,
                                     Variable((torch.ones_like(fin_dis_pred.data) - g_scale_factor).cuda()))

            # Hidden vis recon vs hidden real
            # NLL_out = NLLNormal(hid_dis_pred, hid_dis_real)
            # print("NLL output is:", NLL_out, NLL_out.size())
            # feature_loss_sum = torch.sum(NLL_out, [1, 2, 3])
            # print("feature loss sum:", feature_loss_sum, feature_loss_sum.size())
            # feature_loss_pred = torch.mean(feature_loss_sum)
            feature_loss_pred = torch.mean(torch.sum(NLLNormal(hid_dis_pred, hid_dis_real), [1, 2, 3]))
            mse = torch.sum(0.5 * (hid_dis_real - hid_dis_pred) ** 2, 1)
            # print("feature loss:", feature_loss_pred, feature_loss_pred.size())
            # Testing Gaussian loss thing (doesn't work)
            # var = torch.ones_like(hid_dis_pred)
            # feature_loss_pred_torch = -GNLLLoss(hid_dis_pred, hid_dis_real, var)
            # print('Are NLL loss calculations the same?', torch.all(feature_loss_pred_ren.eq(feature_loss_pred_GNLL)))
            # if torch.all(feature_loss_pred_ren.eq(feature_loss_pred_torch)):
            #     print('NLL Tensors match.')
            #     feature_loss_pred = feature_loss_pred_torch
            # else:
            #     raise Exception('NLL loss does not match')
            # feature_loss_pred = NLLNormal(hid_dis_pred, hid_dis_real)
            # feature_loss_pred = torch.mean(torch.sum(NLLNormal(hid_dis_pred, hid_dis_real)))
            # feature_loss_pred = MSE(hid_dis_pred, hid_dis_real) # As calculated by Maria | Not using the NLL

            # loss_encoder = (kl / (training_config.latent_dim * training_config.batch_size)) - (feature_loss_pred / (
            #         4 * 4 * 64))  # 1024
            # loss_encoder = torch.mean(loss_encoder)
            # TODO: Change the above to changeable parameters based on model out size
            # loss_encoder = kl + feature_loss_pred
            # loss_decoder = dec_fake_pred_loss - training_config.lambda_mse * feature_loss_pred
            # loss_discriminator = dis_fake_pred_loss + dis_real_loss

            return bce_dis_original, bce_dis_predicted, nle, kl, mse, feature_loss_pred, dis_real_loss, dis_fake_pred_loss, dis_fake_sampled_loss, dec_fake_pred_loss
            # return nle, kl, bce_dis_original, bce_dis_predicted, loss_encoder, loss_decoder, loss_discriminator, feature_loss_pred

        # Stage 2 Loss
        elif stage == 2:
            dis_fake_cog_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_cog,
                                                                         Variable(
                                                                             torch.zeros_like(fin_dis_cog.data).cuda(),
                                                                             requires_grad=False))
            dis_real_pred_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_pred,
                                                                         Variable((torch.ones_like(
                                                                             fin_dis_pred.data) - d_scale_factor).cuda(),
                                                                                  requires_grad=False))
            # feature_loss_pred_cog = torch.mean(torch.sum(NLLNormal(hid_dis_cog, hid_dis_pred), [1, 2, 3]))
            feature_loss_pred_cog = NLL(hid_dis_cog, hid_dis_pred)

            encoder_loss_2 = kl_2 / (training_config.latent_dim * training_config.batch_size) - feature_loss_pred_cog / (
                    4 * 4 * 64)
            D2_loss = dis_fake_cog_loss + dis_real_pred_loss

            return encoder_loss_2, D2_loss

        # Stage 3 Loss
        else:
            dis_real_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_real,
                                                                     Variable((torch.ones_like(
                                                                         fin_dis_real.data) - d_scale_factor).cuda(),
                                                                              requires_grad=False))
            dis_fake_cog_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_cog,
                                                                         Variable(
                                                                             torch.zeros_like(fin_dis_cog.data).cuda(),
                                                                             requires_grad=False))
            dec_fake_cog_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_cog,
                                                                         Variable((torch.ones_like(
                                                                             fin_dis_cog.data) - g_scale_factor).cuda(),
                                                                                  requires_grad=False))

            # feature_loss_cog = torch.mean(torch.sum(NLLNormal(hid_dis_cog, hid_dis_real), [1, 2, 3]))
            feature_loss_cog = NLL(hid_dis_cog, hid_dis_real)

            G3_loss = dec_fake_cog_loss - 1e-6 * feature_loss_cog
            D3_loss = dis_fake_cog_loss + dis_real_loss

            return G3_loss, D3_loss

        # Feature Loss
        # TODO: Look into GuassianNLLL and NLLL loss
        # TODO: Look into square function


class VaeGanCognitive(nn.Module):

    """
    Dual-VAE/GAN model, which is trained with 3-stage training procedure.
    This model used in training on Stage II and III.
    On Stage II we do knowledge distillation from teacher network (trained on Stage I):
    fix decoder weights and use images generated on Stage I as real for discriminator
    Stage III is fine-tuning with the fixed cognitive encoder.

    """

    def __init__(self, device, encoder, decoder, discriminator, z_size=128, recon_level=3, teacher_net=None, stage=1,
                 mode='vae'):
        super(VaeGanCognitive, self).__init__()
        # latent space size
        self.device = device
        self.z_size = z_size
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.teacher_net = teacher_net
        self.stage = stage
        self.mode = mode

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, sample, gen_size=10):

        if sample is not None:
            x = Variable(sample['fmri'], requires_grad=False).to(self.device)
            gt_x = Variable(sample['image'], requires_grad=False).to(self.device)

            if self.training:
                    mus, log_variances = self.encoder(x)
                    z = self.reparameterize(mus, log_variances)
                    x_tilde = self.decoder(z)

                    if self.teacher_net is not None and self.stage == 2:

                        for param in self.teacher_net.encoder.parameters():
                            param.requires_grad = False

                        # Inter-modality knowledge distillation
                        mu_teacher, logvar_teacher = self.teacher_net.encoder(gt_x)
                        # Re-parametrization trick
                        z_teacher = self.reparameterize(mu_teacher, logvar_teacher)
                        # Reconstruct gt by the teacher net
                        gt_x = self.decoder(z_teacher)

                    z_p = Variable(torch.randn(len(x), self.z_size).to(self.device), requires_grad=True)
                    x_p = self.decoder(z_p)

                    disc_layer = self.discriminator(gt_x, x_tilde, x_p, "REC")  # discriminator for reconstruction
                    disc_class = self.discriminator(gt_x, x_tilde, x_p, "GAN")  # gt_x acts as real

                    return gt_x, x_tilde, disc_class, disc_layer, mus, log_variances
            else:
                # Use fmri only for evaluation
                mus, log_variances = self.encoder(x)
                z = self.reparameterize(mus, log_variances)
                x_tilde = self.decoder(z)

                # Also grab visual encoder reconstruction for comparison
                if self.teacher_net is not None and self.stage == 2:

                    for param in self.teacher_net.encoder.parameters():
                        param.requires_grad = False

                    # Inter-modality knowledge distillation
                    mu_teacher, logvar_teacher = self.teacher_net.encoder(gt_x)
                    # Re-parametrization trick
                    z_teacher = self.reparameterize(mu_teacher, logvar_teacher)
                    # Reconstruct gt by the teacher net
                    gt_x = self.decoder(z_teacher)
                    return x_tilde, gt_x

                return x_tilde
        else:
            z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)
            x_p = self.decoder(z_p)
            return x_p

    def __call__(self, *args, **kwargs):
        return super(VaeGanCognitive, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(gt_x, x_tilde, hid_dis_real, hid_dis_pred, fin_dis_real, fin_dis_pred, fin_dis_sampled, mus, variances):

        # (gt_x, x_tilde, hid_dis_real, hid_dis_pred, hid_dis_sampled, fin_dis_real,
        #              fin_dis_pred, fin_dis_sampled, mus, variances):

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5 * (gt_x.view(len(gt_x), -1) - x_tilde.view(len(x_tilde), -1)) ** 2
        # mse_loss = nn.MSELoss()
        # nle = mse_loss(gt_x, x_tilde)

        # kl-divergence
        kld_weight = 1
        kld = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1) * kld_weight

        # mse between intermediate layers
        mse = torch.sum(0.5 * (hid_dis_real - hid_dis_pred) ** 2, 1)

        # bce for decoder and discriminator for original and reconstructed
        # the disc output without anything is the equivalent of BCELoss(x, 1s)
        # i.e., the the loss between real labels and the real images
        # the disc output of 1 - x is the opposite BCELoss (x, 0s)
        bce_dis_original = -torch.log(fin_dis_real + 1e-3)
        bce_dis_predicted = -torch.log(1 - fin_dis_pred + 1e-3)
        bce_dis_sampled = -torch.log(1 - fin_dis_sampled + 1e-3)

        return nle, kld, mse, bce_dis_original, bce_dis_predicted, bce_dis_sampled

    @staticmethod
    def ren_loss(x_gt, x_tilde, mus, log_variances, hid_dis_real, hid_dis_pred, fin_dis_real, fin_dis_pred,
                 stage=2, device='cuda'):
        # set Ren params
        d_scale_factor = 0.25
        g_scale_factor = 1 - 0.75 / 2  # 0.625
        BCE = nn.BCELoss().to(device)
        # MSE = nn.MSELoss().to(device)

        # NEED NLE, KL, BCE_DIS_ORIGINAL, BCE_DIS_PREDICTED
        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5 * (x_gt.view(len(x_gt), -1) - x_tilde.view(len(x_tilde), -1)) ** 2

        # kl-divergence
        kl = -0.5 * torch.sum(-log_variances.exp() - torch.pow(mus, 2) + log_variances + 1, 1)

        # Stage 2 Loss
        if stage == 2:
            """# dis_fake_cog_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_cog,
            #                                                              Variable(
            #                                                                  torch.zeros_like(fin_dis_cog.data).cuda(),
            #                                                                  requires_grad=False))
            # dis_real_pred_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_pred,
            #                                                               Variable((torch.ones_like(
            #                                                                   fin_dis_pred.data) - d_scale_factor).cuda(),
            #                                                                        requires_grad=False))
            # feature_loss_pred_cog = torch.mean(torch.sum(NLLNormal(hid_dis_cog, hid_dis_pred), [1, 2, 3]))"""
            dis_fake_cog_loss = BCE(fin_dis_pred, Variable(torch.zeros_like(fin_dis_pred.data).cuda()))
            dis_real_pred_loss = BCE(fin_dis_real, Variable((torch.ones_like(fin_dis_real.data) - d_scale_factor).cuda()))

            feature_loss_vis_cog = torch.mean(torch.sum(NLLNormal(hid_dis_pred, hid_dis_real)))

            loss_encoder = kl / (
                        training_config.latent_dim * training_config.batch_size) - feature_loss_vis_cog / (
                                     4 * 4 * 64)
            loss_discriminator = dis_fake_cog_loss + dis_real_pred_loss

            dec_fake_pred_loss = BCE(fin_dis_pred,
                                     Variable((torch.ones_like(fin_dis_pred.data) - g_scale_factor).cuda()))
            loss_decoder = dec_fake_pred_loss - 1e-6 * feature_loss_vis_cog

            return nle, loss_encoder, loss_decoder, loss_discriminator

        # Stage 3 Loss
        if stage == 3:
            dis_real_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_real,
                                                                     Variable((torch.ones_like(
                                                                         fin_dis_real.data) - d_scale_factor).cuda(),
                                                                              requires_grad=False))
            dis_fake_cog_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_cog,
                                                                         Variable(
                                                                             torch.zeros_like(fin_dis_cog.data).cuda(),
                                                                             requires_grad=False))
            dec_fake_cog_loss = nn.BCEWithLogitsLoss(size_average=False)(fin_dis_cog,
                                                                         Variable((torch.ones_like(
                                                                             fin_dis_cog.data) - g_scale_factor).cuda(),
                                                                                  requires_grad=False))

            # feature_loss_cog = torch.mean(torch.sum(NLLNormal(hid_dis_cog, hid_dis_real), [1, 2, 3]))
            feature_loss_cog = NLL(hid_dis_cog, hid_dis_real)

            G3_loss = dec_fake_cog_loss - 1e-6 * feature_loss_cog
            D3_loss = dis_fake_cog_loss + dis_real_loss

            return G3_loss, D3_loss

        # Feature Loss
        # TODO: Look into GuassianNLLL and NLLL loss
        # TODO: Look into square function


class VAE(nn.Module):
    """
    Just VAE
    """
    def __init__(self, device, z_size=128):
        super(VAE, self).__init__()
        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size).to(device)
        # self.encoder = ResNetEncoder(z_size=self.z_size).to(device)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size).to(device)
        # self-defined function to init the parameters
        self.init_parameters()
        self.device = device

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation # TODO: Check this
                    scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /= numpy.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform_(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant_(m.bias, 0.0)

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x, gen_size=10):
        x = Variable(x).to(self.device)
        if self.training:
            mus, log_variances = self.encoder(x)
            z = self.reparameterize(mus, log_variances)
            x_tilde = self.decoder(z)

            return x_tilde, mus, log_variances
        else:
            mus, log_variances = self.encoder(x)
            z = self.reparameterize(mus, log_variances)
            x_tilde = self.decoder(z)
            return x_tilde

    def __call__(self, *args, **kwargs):
        return super(VAE, self).__call__(*args, **kwargs)

    @staticmethod
    def loss(x, x_tilde, hid_dis_real, hid_dis_pred, hid_dis_sampled, fin_dis_real, fin_dis_pred, fin_dis_sampled, mus, variances):

        # reconstruction error, not used for the loss but useful to evaluate quality
        nle = 0.5 * (x.view(len(x), -1) - x_tilde.view(len(x_tilde), -1)) ** 2

        # kl-divergence
        kl = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1, 1)
        # print("axis then sum", kl, kl.size())
        # print("then, sum that", torch.sum(kl), torch.sum(kl).size())
        # kl_meth_2 = -0.5 * torch.sum(-variances.exp() - torch.pow(mus, 2) + variances + 1)
        # print("sum with no axis", kl_meth_2, kl_meth_2.size())

        # loss_KL = -0.5* torch.mean( 1.0 + logvar - mean.pow(2.0) - logvar.exp() )

        # mse between intermediate layers
        # mse_1 = torch.sum((hid_dis_real - hid_dis_pred) ** 2)
        # The above didn't work as well. Not sure why though.
        # Likely, because things are summed, everything is really finely tuned.
        mse_1 = torch.sum(0.5 * (hid_dis_real - hid_dis_pred) ** 2, 1)
        mse_2 = torch.sum(0.5 * (hid_dis_real - hid_dis_sampled) ** 2, 1)  # NEW

        # bce for decoder and discriminator for original and reconstructed
        bce_dis_original = -torch.log(fin_dis_real + 1e-5) # 1e-3
        bce_dis_predicted = -torch.log(1 - fin_dis_pred + 1e-5)
        bce_dis_sampled = -torch.log(1 - fin_dis_sampled + 1e-5)

        bce_gen_sampled = -torch.log(fin_dis_sampled + 1e-3)  # NEW
        bce_gen_recon = -torch.log(fin_dis_pred + 1e-3)  # NEW
        '''


        bce_gen_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.ones_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_gen_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.ones_like(labels_sampled.data).cuda(), requires_grad=False))
        bce_dis_original = nn.BCEWithLogitsLoss(size_average=False)(labels_original,
                                        Variable(torch.ones_like(labels_original.data).cuda(), requires_grad=False))
        bce_dis_predicted = nn.BCEWithLogitsLoss(size_average=False)(labels_predicted,
                                         Variable(torch.zeros_like(labels_predicted.data).cuda(), requires_grad=False))
        bce_dis_sampled = nn.BCEWithLogitsLoss(size_average=False)(labels_sampled,
                                       Variable(torch.zeros_like(labels_sampled.data).cuda(), requires_grad=False))
        '''
        return nle, kl, mse_1, mse_2, bce_dis_original, bce_dis_predicted, bce_dis_sampled, \
               bce_gen_recon, bce_gen_sampled


class WaeGan(nn.Module):
    """
    WAE with GAN-based penalty. WAE discriminator distinguishes real and fake latent distributions
    rather than images such as for standard GANs.
    """
    def __init__(self, device, z_size=128):
        super(WaeGan, self).__init__()
        # latent space size
        self.z_size = z_size
        self.encoder = Encoder(z_size=self.z_size).to(device)
        self.decoder = Decoder(z_size=self.z_size, size=self.encoder.size).to(device)
        # self.decoder = WaeDecoder(z_size=self.z_size, size=self.encoder.size).to(device)
        self.discriminator = WaeDiscriminator(z_size=self.z_size).to(device)
        # self-defined function to init the parameters
        self.init_parameters()
        self.device = device

    def init_parameters(self):
        # just explore the network, find every weight and bias matrix and fill it
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                if hasattr(m, "weight") and m.weight is not None and m.weight.requires_grad:
                    # init as original implementation
                    scale = 1.0 / numpy.sqrt(numpy.prod(m.weight.shape[1:]))
                    scale /= numpy.sqrt(3)
                    # nn.init.xavier_normal(m.weight,1)
                    # nn.init.constant(m.weight,0.005)
                    nn.init.uniform(m.weight, -scale, scale)
                if hasattr(m, "bias") and m.bias is not None and m.bias.requires_grad:
                    nn.init.constant(m.bias, 0.0)

    def forward(self, x, gen_size=10):

        if x is not None:
            x = Variable(x).to(self.device)

        if self.training:

            # TODO: currently not used

            mus, log_variances = self.encoder(x)
            x_tilde = self.decoder(mus)

            z_p = Variable(torch.randn(len(mus), self.z_size).to(self.device), requires_grad=True)
            x_p = self.decoder(z_p)

            disc_class = self.discriminator(mus, x_p, "GAN")  # encoder distribution

            return x_tilde, disc_class, mus, log_variances
        else:
            if x is None:
                # z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)  # just sample and decode
                z_p = Variable(torch.randn_like(x).to(self.device), requires_grad=False)
                x_p = self.decoder(z_p)
                return x_p
            else:
                mus, log_variances = self.encoder(x)
                x_tilde = self.decoder(mus)

                # logits = self.discriminator(mus)
                return x_tilde, mus

    def __call__(self, *args, **kwargs):
        return super(WaeGan, self).__call__(*args, **kwargs)


class WaeGanCognitive(nn.Module):

    """
    WAE/GAN model for training in Stage II and III
    """
    def __init__(self, device, encoder, decoder, discriminator=None, z_size=128):
        super(WaeGanCognitive, self).__init__()
        # latent space size
        self.z_size = z_size
        self.encoder = encoder
        if discriminator is not None:
            print('using defined discriminator...')
            self.discriminator = discriminator
        else:
            print('creating new discriminator...')
            self.discriminator = WaeDiscriminator(z_size=self.z_size).to(device)
        self.device = device
        self.decoder = decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, logvar):
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        return eps.mul(logvar).add_(mu)

    def forward(self, x, gen_size=10):

        if x is not None:
            x = Variable(x).to(self.device)

        if self.training:
            mus, log_variances = self.encoder(x)
            z = self.reparameterize(mus, log_variances)
            x_tilde = self.decoder(mus)

            z_p = Variable(torch.randn(len(mus), self.z_size).to(self.device), requires_grad=True)
            x_p = self.decoder(z_p)

            disc_class = self.discriminator(mus, x_p, "GAN")  # encoder distribution

            return x_tilde, disc_class, mus, log_variances
        else:
            if x is None:
                # z_p = Variable(torch.randn(gen_size, self.z_size).to(self.device), requires_grad=False)  # just sample and decode
                z_p = Variable(torch.randn_like(x).to(self.device), requires_grad=False)
                x_p = self.decoder(z_p)
                return x_p
            else:
                mus, log_variances = self.encoder(x)
                x_tilde = self.decoder(mus)

                logits = self.discriminator(mus)
                return x_tilde, logits


class WaeDiscriminator(nn.Module):

    """
    Discriminator for the latent space to distinguish real and fake latent representations
    """

    def __init__(self, z_size=128, dim_h=512):
        super(WaeDiscriminator, self).__init__()
        self.n_z = z_size
        self.dim_h = dim_h

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, self.dim_h),
            nn.ReLU(True),
            nn.Linear(self.dim_h, 1),
            # Removing sigmoid to use the BCELogits loss
            # nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0099999)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.main(x)
        return x



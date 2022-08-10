"""_______________________Config for model parameters__________________"""

kernel_size = 5
stride = 2
padding = 2
dropout = 0.7

encoder_channels = [64, 128, 256] # GOOD
# decoder_channels = [256, 128, 64, 32, 3]
decoder_channels = [128, 64, 32, 3]
discrim_channels = [64, 128, 256, 256]  # [0] was 32

maria_decoder_channels = [256, 128, 64, 3]
maria_discrim_channels = [32, 128, 256, 256]

# paper settings
image_size = 100
fc_input = 13  # 8/13/14/16/28 image_size = 64/100/112/128/224
fc_output = 2048  # Was 1024
fc_input_gan = 7
fc_output_gan = 256
stride_gan = 2
latent_dim = 512 # NOT CORRECT: 128
output_pad_dec = [False, True, True]
# decoder_channels = [256, 128, 64, 3] # NOT CORRECT [...64, 32, 3]

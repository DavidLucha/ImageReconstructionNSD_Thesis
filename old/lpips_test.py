import lpips
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

transform_blur_1 = transforms.Compose([transforms.Resize((100, 100)),
                                   transforms.CenterCrop((100, 100)),
                                   transforms.ToTensor(),
                                   transforms.GaussianBlur((99,99),sigma=(0.9,10.0))
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ])

transform_blur_2 = transforms.Compose([transforms.Resize((100, 100)),
                                   transforms.CenterCrop((100, 100)),
                                   transforms.ToTensor(),
                                   transforms.GaussianBlur((9,9))
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ])

transform_norm = transforms.Compose([transforms.Resize((100, 100)),
                                    transforms.CenterCrop((100, 100)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])

image_dir = "D:/Lucha_Data/datasets/NSD/images/faces/faces/shared_nsd03729.png"

target = Image.open(image_dir)
target = transform_norm(target)
target_1 = transform_blur_1(target)
target_2 = transform_blur_2(target)
# target_np = target_trans.numpy()

plt.imshow(target_1.cpu().detach().permute(1,2,0))
plt.show()
plt.imshow(target_2.cpu().detach().permute(1,2,0))
plt.show()

vgg_distance = lpips.LPIPS(net='vgg')
vgg_1 = vgg_distance(target, target_1)
vgg_2 = vgg_distance(target, target_2)

img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.ones(1,3,64,64)
d = vgg_distance(img0, img1)

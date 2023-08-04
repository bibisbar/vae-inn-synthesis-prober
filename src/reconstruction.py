from numpy.core.fromnumeric import var
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
from diffusers.models import AutoencoderKL
import matplotlib.pyplot as plt
import numpy as np
import INN

from model import VaeInnModel

BATCH_SIZE = 20 # create BATCH_SIZE images in total

model = VaeInnModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()
model.cuda()


transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = datasets.CIFAR10(root='data', train=True,
                              download=True, transform=transform)
val_set = datasets.CIFAR10(root='data', train=False,
                              download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=20, shuffle=True)
val_loader = DataLoader(val_set, batch_size=20)


encoder = model.VariationalEncoder()
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

#get a batch of data to test
features, targets = next(iter(train_loader))
grid_img = torchvision.utils.make_grid(features.cpu(), nrow=5)
plt.imshow(grid_img.permute(1, 2, 0))

# image_demo = features[0]
# image_demo.unsqueeze(0)
# tensor_to_image(image_demo)
#encode
# latent_space = encoder(features)
# #decode
# decoded_output = vae.tiled_decode(latent_space)
# image_rec = decoded_output.sample[0].unsqueeze(0)
# image_rec = image_rec.squeeze(0)
# tensor_to_image(image_rec)
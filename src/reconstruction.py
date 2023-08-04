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
model.load_state_dict(torch.load("/export/home/ra35tiy/vae-inn-synthesis-prober/ckpt/model_mnist_vae.pt"))
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



#get a batch of data to test
features, targets = next(iter(train_loader))
grid_img = torchvision.utils.make_grid(features.cpu(), nrow=5)
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig("original.png")

inn_output,x1,x2 = model(features.cuda())
print(inn_output.shape)
samples = inn_output.reshape((20, 4, 4, 4))
decoded_img = model.encoder.sd_vae.tiled_decode(samples)
grid_img = torchvision.utils.make_grid(decoded_img.sample.cpu(), nrow=5)
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig("reconstructed.png")


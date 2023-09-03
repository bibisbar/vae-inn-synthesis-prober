from numpy.core.fromnumeric import var
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda, Resize, ConvertImageDtype, Grayscale, ToPILImage
from model_mnist import VaeInnModel
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
import matplotlib.pyplot as plt
import torchvision
import warnings


import json
from diffusers.image_processor import VaeImageProcessor

from fid_score_cal import calculate_fid
warnings.filterwarnings('ignore')
#transfrom mnist from grayscale to rgb

transform = Compose(
    [ToTensor(), Normalize((0.5,), (0.5,)),
     Resize(32),Lambda(lambda x: x.repeat(3, 1, 1) )
])
# mnist dataset 
train_set = datasets.MNIST(root='data', train=False,
                              download=True,transform=transform)
#initialize mnist dataloader
val_loader = DataLoader(train_set, batch_size=1, num_workers=32)

#save all the images in the dataloader
for i, (images, labels) in enumerate(val_loader):
    torchvision.utils.save_image(images, f"/export/home/ra35tiy/vae-inn-synthesis-prober/src/data/fashion_mnist_fid/eval_{i}.png")
    
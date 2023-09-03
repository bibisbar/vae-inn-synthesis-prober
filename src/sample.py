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

model = VaeInnModel()
model.load_state_dict(torch.load("model_best.pt"))
model.eval()
model.cuda()


BATCH_SIZE = 25
latents = model.innmodule.inn.inverse(torch.randn((BATCH_SIZE, 4*16*16)).cuda()).detach()
latents = latents.reshape((BATCH_SIZE, 4, 16, 16))
imgs = model.encoder.sd_vae.tiled_decode(latents).sample
grid_img = torchvision.utils.make_grid(imgs, nrow=5)
torchvision.utils.save_image(grid_img, f"/export/home/ra35tiy/vae-inn-synthesis-prober/results/outputs_ffhq/eval.png")

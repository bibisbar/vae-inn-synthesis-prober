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
import numpy as np
import INN

#add tensorboard logger
logger = TensorBoardLogger("tb_logs", name="my_model")

#add ckpt
# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename='sample-cifar10-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
    save_last=True
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = Compose(
    [ToTensor(),
     Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = datasets.CIFAR10(root='data', train=True,
                              download=True, transform=transform)
val_set = datasets.CIFAR10(root='data', train=False,
                              download=True, transform=transform)

class VariationalEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.sd_vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

        # Freeze sd-vae
        for param in self.sd_vae.parameters():
            param.requires_grad = False
        
        # self.sd_vae.eval()
    def forward(self,x):
        latent_output = self.sd_vae.tiled_encode(x)
        return latent_output.latent_dist.sample()

class InnModel(pl.LightningModule):
    def __init__(self) :
        super().__init__()
        self.latent_dim = 2
        self.mu = INN.Sequential(INN.Nonlinear(64, 'RealNVP', k=self.latent_dim), INN.Nonlinear(64, 'RealNVP', k=self.latent_dim))
        self.sigma = INN.Sequential(INN.Nonlinear(64, 'RealNVP', k=self.latent_dim), INN.Nonlinear(64, 'RealNVP', k=self.latent_dim))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self,x):
        mean_out, _, _ = self.mu(x)
        var_out, _, _ = self.sigma(x)
        return self.relu(mean_out) , self.sig(var_out)

class VaeInnModel(pl.LightningModule):
    def __init__(self) :
        super().__init__()
        
        self.BATCH_SIZE = 8
        #Magic number
        self.scale_factor = 0.18215
        # Initialize three parts
        self.encoder = VariationalEncoder()
        self.flatten = nn.Flatten()
        self.innmodule = InnModel()

        # Initialize a 'target' normal distribution for KL divergence
        self.norm = torch.distributions.Normal(0, 1)
    def forward(self, x):
        z_sample = self.encoder(x)
        z_flatten = self.flatten(z_sample*self.scale_factor)
        mean_flatten,var_flatten = self.innmodule(z_flatten)
        mean_sample = mean_flatten.reshape(z_sample.shape)
        var_sample = var_flatten.reshape(z_sample.shape)

        #Get sample from norm distribution
        norm_sample = self.norm.sample(mean_sample.shape).to(DEVICE)

        #Reparalization
        inn_sample =  var_sample * norm_sample + mean_sample

        inn_sample_reverse = inn_sample / self.scale_factor  #TODO double check if this is correct

        return inn_sample_reverse, mean_sample, var_sample

    def training_step(self, batch, batch_idx):
        image, label = batch
        latent_output, mean , var = self(image)


        #image_rec = sd_vae.tiled_decode(latent_output).sample

        #Compute kl loss
        kl_loss = (0.5 * (var ** 2) + 0.5 * (mean ** 2) - torch.log(var) - 0.5).float().sum()
        self.log("train_kl_loss", kl_loss)
        #Compute reconstruction loss
        #rec_loss = ((image - image_rec)**2).sum()
        #self.log("train_rec_loss", rec_loss)
        #loss = kl_loss + rec_loss
        loss = kl_loss  #remove rec loss
        #Logs
        
        self.log('train_total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        latent_output, mean , var = self(image)

        #image_rec = sd_vae.tiled_decode(latent_output).sample

        #Compute kl loss
        kl_loss = (0.5 * (var ** 2) + 0.5 * (mean ** 2) - torch.log(var) - 0.5).sum()
        self.log("val_kl_loss", kl_loss)
        #Compute reconstruction loss
        #rec_loss = ((image - image_rec)**2).sum()
        #self.log("val_rec_loss", rec_loss)
        #loss = kl_loss + rec_loss
        loss = kl_loss  #remove rec loss
        #Logs
        self.log("val_total_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        image, label = batch
        latent_output, mean , var = self(image)

        #image_rec = sd_vae.tiled_decode(latent_output).sample

        #Compute kl loss
        kl_loss = (0.5 * (var ** 2) + 0.5 * (mean ** 2) - torch.log(var) - 0.5).sum()
        self.log("test_kl_loss", kl_loss)
        #Compute reconstruction loss
        #rec_loss = ((image - image_rec)**2).sum()
        #self.log("test_rec_loss", rec_loss)
        #loss = kl_loss + rec_loss
        loss = kl_loss  #remove rec loss
        #Logs
        self.log("test_total_loss", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=0.001)

        # Only optimize the parameters that are requires_grad
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.001)
        return optimizer

# Example usage
# Assuming you have train_dataset and val_dataset as torch.utils.data.Dataset objects

model = VaeInnModel()
train_loader = DataLoader(train_set, batch_size=model.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=model.BATCH_SIZE)

if DEVICE == 'cpu':
  devices = 0
else:
  devices = 1
# trainer = pl.Trainer(max_epochs=10, devices=devices,logger=logger,callbacks=[checkpoint_callback])  # Set devices=1 for GPU training
trainer = pl.Trainer(max_epochs=10, devices=devices,logger=logger)  # Set devices=1 for GPU training

# Start training
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


#TODO
# 1. freeze sd-vae
# 2. reverse the latent space
 
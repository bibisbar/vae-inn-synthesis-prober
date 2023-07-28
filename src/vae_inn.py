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
    filename='sample-cifar100-{epoch:02d}-{val_loss:.2f}',
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
        self.hidden_layers = 2
        self.inn = INN.Sequential(INN.Nonlinear(64, 'RealNVP', k=self.latent_dim), INN.Nonlinear(64, 'RealNVP', k=self.hidden_layers))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
    def forward(self,x):
        y, logp, logdet = self.inn(x) 
        return y, logp, logdet

class VaeInnModel(pl.LightningModule):
    def __init__(self) :
        super().__init__()
        
        self.BATCH_SIZE = 64
        #Magic number
        self.scale_factor = 0.18215
        # Initialize three parts
        self.encoder = VariationalEncoder()
        self.flatten = nn.Flatten()
        self.innmodule = InnModel()

        # Initialize a 'target' normal distribution for the NLL loss
        self.n = INN.utilities.NormalDistribution()
    def forward(self, x):
        z_sample = self.encoder(x)
        z_flatten = self.flatten(z_sample*self.scale_factor)
        y, logp, logdet = self.innmodule(z_flatten)
        return y, logp, logdet

    def training_step(self, batch, batch_idx):
        image, label = batch
        y, logp, logdet = self(image)
        py = self.n.logp(y)

        loss = py + logp + logdet
        loss = -1 * loss.mean()
        
        self.log('train_total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        y, logp, logdet = self(image)
        py = self.n.logp(y)

        loss = py + logp + logdet
        loss = -1 * loss.mean()
        
        self.log("val_total_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        image, label = batch
        y, logp, logdet = self(image)
        py = self.n.logp(y)

        loss = py + logp + logdet
        loss = -1 * loss.mean()
        
        self.log("test_total_loss", loss)
        return loss

    def configure_optimizers(self):
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
trainer = pl.Trainer(max_epochs=30, devices=devices, logger=logger)  # Set devices=1 for GPU training

# Start training
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

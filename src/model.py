import pytorch_lightning as pl
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
import INN
import torch.optim as optim


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
        self.dims = 256
        self.inn = INN.Sequential(INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims),
                                  INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims),
                                  INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims),
                                  INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims),
                                  INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims),
                                  INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims))
    def forward(self,x):
        y, logp, logdet = self.inn(x) 
        return y, logp, logdet
    
class ConvInnModel(pl.LightningModule):
    def __init__(self) :
        super().__init__()
        self.n = INN.utilities.NormalDistribution()
        self.inn = INN.Sequential(INN.Conv2d(channels=1, kernel_size=3),
                                  INN.Conv2d(channels=1, kernel_size=3),
                                  INN.Conv2d(channels=1, kernel_size=3),
                                  INN.Conv2d(channels=1, kernel_size=3),
                                  INN.Conv2d(channels=1, kernel_size=3),
                                  INN.Conv2d(channels=1, kernel_size=3))
    def forward(self,x):
        y, logp, logdet = self.inn(x) 
        return y, logp, logdet
    
    def training_step(self, batch, batch_idx):
        image, label = batch
        y, logp, logdet = self(image)
        py = self.n.logp(y)

        loss = py + logp + logdet
        loss = -1 * loss.mean()
        
        self.log('train_total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        # Only optimize the parameters that are requires_grad
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0005)
        return optimizer
    
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
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0005)
        return optimizer
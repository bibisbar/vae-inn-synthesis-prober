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

# For learning with the vae
class InnModel(pl.LightningModule):
    def __init__(self, latent_dims: list) :
        super().__init__()
        self.dims = latent_dims[0] * latent_dims[1] * latent_dims[2]
        self.layers = []
        for i in range(0, 5):
            self.layers.append(INN.BatchNorm1d(self.dims))
            self.layers.append(INN.Nonlinear(self.dims, 'RealNVP'))
            self.layers.append(INN.JacobianLinear(self.dims))
        self.inn = INN.Sequential(*self.layers)
    def forward(self,x):
        y, logp, logdet = self.inn(x) 
        return y, logp, logdet
    
class ConvINN(pl.LightningModule):
    def __init__(self, latent_dims: list) :
        super().__init__()
        self.dims = latent_dims[0] * latent_dims[1] * latent_dims[2]
        self.layers = []
        for i in range(0, 5):
            self.layers.append(INN.BatchNorm1d(self.dims))
            self.layers.append(INN.Nonlinear(self.dims, 'RealNVP'))
            self.layers.append(INN.JacobianLinear(self.dims))
        self.inn = INN.Sequential(*self.layers)
        
    def forward(self,x):
        y, logp, logdet = self.inn(x) 
        return y, logp, logdet
    
class VaeInnModel(pl.LightningModule):
    def __init__(self, latent_dims: list) :
        super().__init__()
        
        self.BATCH_SIZE = 32
        #Magic number
        self.scale_factor = 0.18215
        # Initialize three parts
        self.encoder = VariationalEncoder()
        self.flatten = nn.Flatten()
        self.innmodule = InnModel(latent_dims=latent_dims) # change and test with ConvINN

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

        loss = py + logdet
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
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=8e-5)
        return optimizer

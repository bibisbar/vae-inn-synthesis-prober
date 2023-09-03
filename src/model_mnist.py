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
    def __init__(self) :
        super().__init__()
        self.dims = 4*4*4
        self.inn = INN.Sequential(INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims),
                                  INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims),
                                  INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims),
                                  INN.BatchNorm1d(self.dims), INN.Nonlinear(self.dims, 'RealNVP'), INN.JacobianLinear(self.dims)
                                  )
    def forward(self,x):
        y, logp, logdet = self.inn(x) 
        return y, logp, logdet
    

    
class VaeInnModel(pl.LightningModule):
    def __init__(self) :
        super().__init__()
        
        # self.BATCH_SIZE = 2048
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
        #print('zsample',z_sample.shape)
        # z_flatten = self.flatten(z_sample*self.scale_factor)
        z_flatten = self.flatten(z_sample)
        #print('zflatten',z_flatten.shape)
        y, logp, logdet = self.innmodule(z_flatten)
        return y, logp, logdet

    def training_step(self, batch, batch_idx):
        image,label= batch
        print(image.shape)
        y, logp, logdet = self(image)
        py = self.n.logp(y)

        loss = py + logdet
        loss = -1 * loss.mean()
        
        self.log('train_total_loss', loss, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image,label= batch
        print(image.shape)
        y, logp, logdet = self(image)
        py = self.n.logp(y)

        loss = py + logdet
        loss = -1 * loss.mean()
        
        self.log("val_total_loss", loss,on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        image,label= batch
        y, logp, logdet = self(image)
        py = self.n.logp(y)

        loss = py + logdet
        loss = -1 * loss.mean()
        
        self.log("test_total_loss", loss,sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Only optimize the parameters that are requires_grad
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=0.0001)
        return optimizer
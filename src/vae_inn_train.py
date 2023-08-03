from numpy.core.fromnumeric import var
import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda, Resize, ConvertImageDtype, Grayscale, ToPILImage
from model import VaeInnModel
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
import matplotlib.pyplot as plt
import torchvision
from diffusers.image_processor import VaeImageProcessor

#add tensorboard logger
logger = TensorBoardLogger("tb_logs", name="my_model")
print(torch.cuda.is_available())

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def normalize(x):
   return x * 2. - 1.
def normalize_cifar10(x):
   return x / 255.
def make_rgb(x):
   return torch.cat([x,x,x]) # For single-channel images
def denormalize(x):
   return (x + 1.) / 2.

transform = Compose(
    [ToTensor(), normalize])

train_set = datasets.CIFAR10(root='data', train=True,
                              download=True, transform=transform)
print(train_set)
#val_set = datasets.CIFAR10(root='data', train=False,
#                              download=True, transform=transform)

# Example usage
# Assuming you have train_dataset and val_dataset as torch.utils.data.Dataset objects

model = VaeInnModel()
pl.seed_everything(42)
train_loader = DataLoader(train_set, batch_size=model.BATCH_SIZE, shuffle=True)
#val_loader = DataLoader(val_set, batch_size=model.BATCH_SIZE)

if DEVICE == 'cpu':
  devices = 0
else:
  devices = 1

class ModelSaveCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % 5 == 0:
          torch.save(model.state_dict(), "model.pt")
          BATCH_SIZE = 30
          latents = model.innmodule.inn.inverse(torch.randn((BATCH_SIZE, 4*4*4)).cuda()).detach()
          latents = latents.reshape((BATCH_SIZE, 4, 4, 4))
          imgs = model.encoder.sd_vae.tiled_decode(latents * (1./model.scale_factor)).sample
          #transform = Grayscale(1) # for single-channel datasets
          #imgs = transform(imgs)
          grid_img = torchvision.utils.make_grid(imgs, nrow=5)
          torchvision.utils.save_image(grid_img, f"vae-inn-synthesis-prober/src/outputs_vae_cifar10/epoch_{trainer.current_epoch}.png")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


# trainer = pl.Trainer(max_epochs=10, devices=devices,logger=logger,callbacks=[checkpoint_callback])  # Set devices=1 for GPU training
trainer = pl.Trainer(max_epochs=10000, devices=devices, logger=logger, callbacks=[ModelSaveCallback()])  # Set devices=1 for GPU training

# Start training
trainer.fit(model, train_dataloaders=train_loader)

# Save the model for later inference.
torch.save(model.state_dict(), "model_cifar10_vae.pt")
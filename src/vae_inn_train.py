from numpy.core.fromnumeric import var
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda, Resize, ConvertImageDtype, Grayscale
from model import VaeInnModel
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
import matplotlib.pyplot as plt
import torchvision

#add tensorboard logger
logger = TensorBoardLogger("tb_logs", name="my_model")

#add ckpt
# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
# TODO: https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-and-loading-models
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    filename='sample-cifar100-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
    save_last=True,
    every_n_epochs=20
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = Compose(
    [Resize((32,32)), Grayscale(3), ToTensor(), Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST(root='data', train=True,
                              download=True, transform=transform)
val_set = datasets.CIFAR10(root='data', train=False,
                              download=True, transform=transform)

# Example usage
# Assuming you have train_dataset and val_dataset as torch.utils.data.Dataset objects

model = VaeInnModel()
pl.seed_everything(42)
train_loader = DataLoader(train_set, batch_size=model.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=model.BATCH_SIZE)

if DEVICE == 'cpu':
  devices = 0
else:
  devices = 1

class ModelSaveCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % 2 == 0:
          torch.save(model.state_dict(), "model.pt")
          BATCH_SIZE = 10
          samples = model.innmodule.inn.inverse(torch.randn((BATCH_SIZE, 64))).detach()
          samples = samples.reshape((BATCH_SIZE, 4, 4, 4))
          decoded_img = model.encoder.sd_vae.tiled_decode(samples)
          grid_img = torchvision.utils.make_grid(decoded_img.sample, nrow=5)
          torchvision.utils.save_image(grid_img, f"vae-inn-synthesis-prober/src/outputs_mnist/epoch_{trainer.current_epoch}.png")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


# trainer = pl.Trainer(max_epochs=10, devices=devices,logger=logger,callbacks=[checkpoint_callback])  # Set devices=1 for GPU training
trainer = pl.Trainer(max_epochs=200, devices=devices, logger=logger, callbacks=[ModelSaveCallback()])  # Set devices=1 for GPU training

# Start training
trainer.fit(model, train_dataloaders=train_loader)

# Save the model for later inference.
torch.save(model.state_dict(), "model.pt")
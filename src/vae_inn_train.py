from numpy.core.fromnumeric import var
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda, Resize
from model import VaeInnModel

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
    every_n_epochs=10
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = Compose(
    [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = datasets.CIFAR10(root='data', train=True,
                              download=True, transform=transform)
val_set = datasets.CIFAR10(root='data', train=False,
                              download=True, transform=transform)

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
trainer = pl.Trainer(max_epochs=100, devices=devices, logger=logger)  # Set devices=1 for GPU training

# Start training
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Save the model for later inference.
torch.save(model.state_dict(), "model.pt")
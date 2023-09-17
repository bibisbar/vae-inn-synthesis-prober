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
from pytorch_fid import fid_score
from fid_score_cal import calculate_fid
import torch_fidelity as tfid
from torchvision.io import read_image

#add tensorboard logger
dataset_alias = "ffhq" # TODO: change this, when you are training with another dataset
logger = TensorBoardLogger("tb_logs", name=f"my_model_{dataset_alias}_test")
fid_scores_logger = []
cifar10_latent_dims = [4,4,4]
fashion_mnist = [4,4,4]
celeb_latent_dims = [4,8,8]
ffhq_latent_dims = [4,16,16]
pl.seed_everything(42)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def concat(x):
    return torch.cat([x,x,x],0)

transform = Compose([Resize(32), ToTensor()])

class CelebHQDataset(Dataset):
   def __init__(self) -> None:
      super().__init__()

   def __len__(self):
      return 30000
   
   def __getitem__(self, index):
      img = (read_image(path=f"/export/home/ru27juh/vae-inn-synthesis-prober/src/celeb/{index}.jpg") / 255.) * 2. - 1.
      transform = Resize(64)
      img = transform(img)
      return img, img

class FFHQDataset(Dataset):
   def __init__(self):
      super().__init__()

   def __len__(self):
      return 69999

   def __getitem__(self, index):
      img = (read_image(path=f"/export/home/ru27juh/vae-inn-synthesis-prober/src/ffhq/train/{index}.png") / 255.) * 2. - 1.
      #transform = Resize(32)
      #img = transform(img)
      return img, img

#train_set = datasets.FashionMNIST(root='data', train=True,
#                              download=True, transform=transform)
#print(train_set)
#val_set = datasets.CIFAR10(root='data', train=False,
#                              download=True, transform=transform)
train_set = FFHQDataset()

latent_dims = ffhq_latent_dims # TODO: change it, if the dataset is changed
model = VaeInnModel(latent_dims=latent_dims)
train_loader = DataLoader(train_set, batch_size=model.BATCH_SIZE, shuffle=True, num_workers=36)
#val_loader = DataLoader(val_set, batch_size=model.BATCH_SIZE, num_workers=36)

if DEVICE == 'cpu':
  devices = 0
else:
  devices = 1

class ModelSaveCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        #os.system(f"")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.current_epoch % 20 == 0:
          pass
          '''ori_data_path =f"/export/home/ru27juh/vae-inn-synthesis-prober/src/{dataset_alias}/train"
          gene_data_path = f"/export/home/ru27juh/vae-inn-synthesis-prober/src/{dataset_alias}_generated"
          #remove previously generated images
          os.system(f"rm -r {gene_data_path}")
          
          os.system(f"mkdir {gene_data_path}")
          #generate images
          batchsize_generate = 500

          latents = model.innmodule.inn.inverse(torch.randn((batchsize_generate, latent_dims[0]*latent_dims[1]*latent_dims[2])).cuda()).detach()
          latents = latents.reshape((batchsize_generate, latent_dims[0], latent_dims[1], latent_dims[2]))
          imgs = model.encoder.sd_vae.tiled_decode(latents * (1./model.scale_factor)).sample
          for i in range(batchsize_generate):
            torchvision.utils.save_image(imgs[i], f"{gene_data_path}/{i}.png")
          #calculate fid score
          #fid = {"fid_score": calculate_fid(ori_data_path, gene_data_path)}
          logger.log_metrics(metrics=tfid.calculate_metrics(input1=gene_data_path, input2=ori_data_path, fid=True), step=trainer.current_epoch)'''

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


# trainer = pl.Trainer(max_epochs=10, devices=devices,logger=logger,callbacks=[checkpoint_callback])  # Set devices=1 for GPU training
trainer = pl.Trainer(max_epochs=100, devices=4, logger=logger, callbacks=[ModelSaveCallback()])  # Set devices=1 for GPU training

# Start training
trainer.fit(model, train_dataloaders=train_loader)

# Save the model for later inference.
torch.save(model.state_dict(), f"/export/home/ru27juh/model_{dataset_alias}_vae.pt")

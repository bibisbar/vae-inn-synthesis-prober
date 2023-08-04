from numpy.core.fromnumeric import var
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
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
import deeplake
from diffusers.image_processor import VaeImageProcessor

from dataloaders.dataloader_ffhq import ClassificationDataset
from fid_score_cal import calculate_fid


#initialize dataloader
ds = deeplake.load("hub://activeloop/ffhq")
ds.summary()
transform = Compose(
    [ToTensor()])
ffhq_pytorch = ClassificationDataset(ds, transform = transform)

train_loader= DataLoader(ffhq_pytorch, batch_size = 48, num_workers = 2, shuffle = True)
#add tensorboard logger
logger = TensorBoardLogger("tb_logs", name="my_model")
print(torch.cuda.is_available())

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#initialize model

model = VaeInnModel()
pl.seed_everything(42)


class ModelSaveCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
      if trainer.current_epoch % 20 == 0:
         torch.save(model.state_dict(), "model.pt")
         BATCH_SIZE = 5
         latents = model.innmodule.inn.inverse(torch.randn((BATCH_SIZE, 4*4*4)).cuda()).detach()
         latents = latents.reshape((BATCH_SIZE, 4, 4, 4))
         #imgs = model.encoder.sd_vae.tiled_decode(latents * (1./model.scale_factor)).sample
         imgs = model.encoder.sd_vae.tiled_decode(latents).sample
         grid_img = torchvision.utils.make_grid(imgs, nrow=5)
         torchvision.utils.save_image(grid_img, f"/export/home/ra35tiy/vae-inn-synthesis-prober/results/outputs_ffhq/epoch_{trainer.current_epoch}.png")

         ori_data_path = "/export/home/ra35tiy/vae-inn-synthesis-prober/src/data/ffhq_images"
         gene_data_path = "/export/home/ra35tiy/vae-inn-synthesis-prober/results/sample_per_epoch"
         #generate images
         batchsize_generate = 10000
         latents = model.innmodule.inn.inverse(torch.randn((batchsize_generate, 4*4*4)).cuda()).detach()
         latents = latents.reshape((batchsize_generate, 4, 4, 4))
         #imgs = model.encoder.sd_vae.tiled_decode(latents * (1./model.scale_factor)).sample
         imgs = model.encoder.sd_vae.tiled_decode(latents).sample
         for i in range(10000):
            torchvision.utils.save_image(imgs[i], gene_data_path+"_"+str(trainer.current_epoch)+"_"+str(i)+".png")
         #calculate fid score
         trainer.logger(experiment=trainer.logger.experiment).log({"fid_score": calculate_fid(ori_data_path, gene_data_path)})
         #remove generated images
         os.system("rm -rf /export/home/ra35tiy/vae-inn-synthesis-prober/results/sample_per_epoch/*")
    def on_train_end(self, trainer, pl_module):
        print("Training is ending")


# trainer = pl.Trainer(max_epochs=10, devices=devices,logger=logger,callbacks=[checkpoint_callback])  # Set devices=1 for GPU training
trainer = pl.Trainer(max_epochs=400, logger=logger, callbacks=[ModelSaveCallback()],accelerator="gpu", devices=4)  # Set devices=1 for GPU training

# Start training
trainer.fit(model, train_dataloaders=train_loader)

# Save the model for later inference.
torch.save(model.state_dict(), "model_ffhq.pt")
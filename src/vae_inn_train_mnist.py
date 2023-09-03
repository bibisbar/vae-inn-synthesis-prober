from numpy.core.fromnumeric import var
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda, Resize, ConvertImageDtype, Grayscale, ToPILImage
from model_mnist import VaeInnModel
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer
import matplotlib.pyplot as plt
import torchvision
import warnings


import json
from diffusers.image_processor import VaeImageProcessor

from fid_score_cal import calculate_fid
warnings.filterwarnings('ignore')
#transfrom mnist from grayscale to rgb

transform = Compose(
    [ToTensor(), Normalize((0.5,), (0.5,)),
     Resize(32),Lambda(lambda x: x.repeat(3, 1, 1) )
])
# mnist dataset 
train_set = datasets.MNIST(root='data', train=True,
                              download=True,transform=transform)
#initialize mnist dataloader
train_loader = DataLoader(train_set, batch_size=2048, shuffle=True,num_workers=32)

print('initial dataloader, length = ',len(train_loader))
print('shape of one iteration:', next(iter(train_loader))[0].shape)


#add tensorboard logger
logger = TensorBoardLogger("tb_logs", name="model_mnist")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#initialize model

model = VaeInnModel()
model.load_state_dict(torch.load("/export/home/ra35tiy/vae-inn-synthesis-prober/src/model_390.pt"))


model.cuda()
pl.seed_everything(42)


class ModelSaveCallback(Callback):
    def __init__(self):
        super().__init__()
        self.fid_dict = {}
        self.best_fid = 1000000
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_epoch_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        print("Training epoch is starting")
        if trainer.current_epoch % 2 == 0:
                if trainer.local_rank == 0:
                    
                    print('rank:', trainer.local_rank)
                    BATCH_SIZE = 5
                    latents = model.innmodule.inn.inverse(torch.randn((BATCH_SIZE, 4*4*4)).cuda()).detach()
                    latents = latents.reshape((BATCH_SIZE, 4, 4, 4))
                    #imgs = model.encoder.sd_vae.tiled_decode(latents * (1./model.scale_factor)).sample
                    imgs = model.encoder.sd_vae.tiled_decode(latents).sample
                    grid_img = torchvision.utils.make_grid(imgs, nrow=5)
                    torchvision.utils.save_image(grid_img, f"/export/home/ra35tiy/vae-inn-synthesis-prober/results/outputs_mnist/epoch_{trainer.current_epoch}.png")
                    
                    

        if trainer.current_epoch % 10 == 0:
            if trainer.local_rank == 0:
                ori_data_path = "/export/home/ra35tiy/vae-inn-synthesis-prober/src/data/mnist_fid"
                gene_data_path = "/export/home/ra35tiy/vae-inn-synthesis-prober/results/sample_per_epoch_mnist"
                #generate images
                batchsize_generate = 10000
                for i in range(batchsize_generate):
                    latents = model.innmodule.inn.inverse(torch.randn((1, 4*4*4)).cuda()).detach()
                    latents = latents.reshape((1, 4, 4, 4))
                    #imgs = model.encoder.sd_vae.tiled_decode(latents * (1./model.scale_factor)).sample
                    imgs = model.encoder.sd_vae.tiled_decode(latents).sample
                    torchvision.utils.save_image(imgs[0], gene_data_path+"/epoch"+"_"+str(trainer.current_epoch)+"_"+str(i)+".png")
                
                #calculate fid score
                print('epoch:', trainer.current_epoch)
                #self.log('fid_score', calculate_fid(ori_data_path, gene_data_path),sync_dist=True)
                metricfid = calculate_fid(ori_data_path, gene_data_path)
                self.fid_dict[trainer.current_epoch] = metricfid
                if metricfid < self.best_fid:
                    self.best_fid = metricfid
                    print("best fid score:", self.best_fid)
                    torch.save(model.state_dict(), f"model_best.pt")
                print(self.fid_dict)

                
                #remove generated images
                os.system("rm -rf /export/home/ra35tiy/vae-inn-synthesis-prober/results/sample_per_epoch_mnist/*.png")
        if trainer.current_epoch % 10 == 0:
            if trainer.local_rank == 0:
                torch.save(model.state_dict(), f"model_{trainer.current_epoch}.pt")
                print("model saved")
    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
        fid_dict = json.dumps(self.fid_dict)
        with open("fid_dict.json", "w") as f:
            f.write(fid_dict)
        



trainer = pl.Trainer(max_epochs=610, logger=logger, callbacks=[ModelSaveCallback()],accelerator="gpu", devices=2)  # Set devices=1 for GPU training


# Start training
trainer.fit(model, train_dataloaders=train_loader)

# Save the model for later inference.
torch.save(model.state_dict(), "model_mnist.pt")
import deeplake
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda, Resize, ConvertImageDtype, Grayscale, ToPILImage
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np

ds = deeplake.load("hub://activeloop/ffhq")


ds.summary()
transform = Compose(
    [ToTensor()])

class ClassificationDataset(Dataset):
    def __init__(self, ds, transform = None):
        self.ds = ds
        self.transform = transform
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        image = self.ds['images_1024/image'][idx].numpy()

        if self.transform is not None:
            image = self.transform(image)
        return image
    
ffhq_pytorch = ClassificationDataset(ds, transform = transform)

dataloader_pytorch = DataLoader(ffhq_pytorch, batch_size = 100, num_workers = 2, shuffle = True)

for epoch in range(100):
    images = next(iter(dataloader_pytorch))
    print(epoch)
    for i in range(100):
        torchvision.utils.save_image(images[i], "/export/home/ra35tiy/vae-inn-synthesis-prober/src/data/ffhq_images/ffhq_"+str(epoch)+"_"+str(i)+".png")
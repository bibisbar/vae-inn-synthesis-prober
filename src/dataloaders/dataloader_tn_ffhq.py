import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda, Resize, ConvertImageDtype, Grayscale, ToPILImage
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np



data_dir = '/export/home/ra35tiy/dataset'

transform = transforms.Compose([
                                transforms.ToTensor()
                               ])
dataset = datasets.ImageFolder(data_dir, transform=transform)
lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

image ,label= next(iter(train_dataloader))
print(image.shape)
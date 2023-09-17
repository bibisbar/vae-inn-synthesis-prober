#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import os

transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.FashionMNIST(root='data', train=True,
                              download=True, transform=transform)
val_set = datasets.FashionMNIST(root='data', train=False,
                              download=True, transform=transform)


# In[2]:
dataset_name = "fashion_mnist"

os.system(f'mkdir {dataset_name}')
os.system(f'mkdir {dataset_name}/train')
os.system(f'mkdir {dataset_name}/test')


# In[5]:


for i in range(0, len(train_set)):
    img, _ = train_set.__getitem__(i)
    print(i)
    torchvision.utils.save_image(img, f"{dataset_name}/train/{i}.png")

for i in range(0, len(val_set)):
    img, _ = val_set.__getitem__(i)
    print(i)
    torchvision.utils.save_image(img, f"{dataset_name}/test/{i}.png")


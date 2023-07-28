import torch
import os
from torchvision import transforms
 


def tensor_to_image(tensor_image):
    toPIL = transforms.ToPILImage()
    image = toPIL(tensor_image)
    if os.path.exists('demo.png'):
        image.save('rec.png')
    else:
        image.save('demo.png')
    return

def test_rec():
    pass
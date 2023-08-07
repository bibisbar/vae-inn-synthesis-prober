import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score


real_images_folder = 'path_to_real_images'
# generated_images_folder = './FID_app3'
generated_images_folder = 'path_to_generated_images'

def calculate_fid(real_images_folder, generated_images_folder):
    print('Calculating FID...')
    inception_model = torchvision.models.inception_v3(pretrained=True)


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], device='cuda', dims=2048, batch_size=64)
    #fid_value = 1.0
    print('FID value:', fid_value)
    return fid_value
    
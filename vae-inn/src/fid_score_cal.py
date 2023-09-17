import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score


real_images_folder = 'path_to_real_images'
# generated_images_folder = './FID_app3'
generated_images_folder = 'path_to_generated_images'

def calculate_fid(real_images_folder, generated_images_folder):

    fid_value = fid_score.calculate_fid_given_paths(paths=[real_images_folder, generated_images_folder],
                                                    device='cuda:0', batch_size=512, dims=2048, num_workers=32)
    print('FID value:', fid_value)
    return fid_value
    
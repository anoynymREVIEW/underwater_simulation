import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
import cv2
from io import BytesIO
import random

import numpy as np
from scipy import io
import h5py
import cv2
import glob

from numpy import load


class Make3DDataset(Dataset):
    def __init__(self, transforms=None):
        self.transforms = transforms
        train_images = glob.glob('/sanssauvegarde/homes/t20monda/Make3D/Train400Img/*.jpg')
        train_depth = glob.glob('/sanssauvegarde/homes/t20monda/Make3D/Train400Depth/*.mat')

        a_mat_arr = load('/homes/t20monda/DenseDepth_1/A_Mat_Make_3D.npy')
        beta_mat_arr = load('/homes/t20monda/DenseDepth_1/Beta_Mat_Make_3D.npy')

        self.train_images = sorted(train_images, key=lambda p: p.split('/')[-1].split('img-')[-1])
        self.train_depth = sorted(train_depth, key=lambda p: p.split('/')[-1].split('depth_sph_corr-')[-1])

        self.beta_mat_arr = beta_mat_arr
        self.a_mat_arr = a_mat_arr

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, idx):

        haze_image_name = "/sanssauvegarde/homes/t20monda/all_data_make_3D/" + str(idx) + "haze_image" + ".npy"
        complex_haze_image_name =  "/sanssauvegarde/homes/t20monda/all_data_make_3D/" + str(idx) + "complex_haze_image_name" + ".npy"

        haze_image = load(haze_image_name)  # m * n * 3
        complex_noisy_img = load(complex_haze_image_name)  # m * n * 3

        m = haze_image.shape[0]
        n = complex_noisy_img.shape[1]

        beta_mat = self.beta_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)  # m * n * 3

        a_mat = self.a_mat_arr[idx]
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)  # m * n * 3

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = create_reorganize_dimension_custom(unit_mat, m, n)  # m * n * 3

        image = cv2.imread(self.train_images[idx])
        depth = io.loadmat(self.train_depth[idx])['Position3DGrid'][:, :, 3]

        image = cv2.resize(image, (460, 345), interpolation=cv2.INTER_LINEAR)  # 3 * m * n
        depth = cv2.resize(depth, (460, 345), interpolation=cv2.INTER_LINEAR)  # m * n

        depth_half = cv2.resize(depth, (230, 173), interpolation=cv2.INTER_LINEAR)
        image_half = cv2.resize(image, (230, 173), interpolation=cv2.INTER_LINEAR)
        del depth
        
        # another_simple_image_save(image_half, "before_trans_image_half_numpy.png")
        # another_simple_image_save(depth_half, "before_trans_depth_half_numpy.png")
        # another_simple_image_save(haze_image, "before_trans_haze_image_numpy.png")
        # another_simple_image_save(complex_noisy_img, "before_trans_complex_noisy_image_numpy.png")

        if self.transforms is not None:
            image, depth_half_1 = self.transforms(image, depth_half)
            image_half, depth_half = self.transforms(image_half, depth_half)

            haze_image, complex_noisy_img = self.transforms(haze_image, complex_noisy_img)
        
        # another_simple_image_save(image_half, "after_trans_image_half_numpy.png")
        # another_simple_image_save(depth_half, "after_trans_depth_half_numpy.png")
        # another_simple_image_save(haze_image, "after_trans_haze_image_numpy.png")
        # another_simple_image_save(complex_noisy_img, "after_trans_complex_noisy_image_numpy.png")
        
        del depth_half_1
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255
        image_half = torch.from_numpy(image_half).permute(2, 0, 1).float() / 255
        depth_half = torch.from_numpy(depth_half).float() / 80

        depth_half.requires_grad = False
        image.requires_grad = False
        image_half.requires_grad = False

        image_numpy = np.array(image)
        image_numpy = np.swapaxes(image_numpy, 0, 2)  # making m * n * 3
        image_numpy = np.swapaxes(image_numpy, 0, 1)  # making m * n * 3

        image_half_numpy = np.array(image_half)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)  # making m * n * 3
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)  # making m * n * 3

        depth_half = torch.unsqueeze(depth_half, 0)
        depth_half = torch.tile(depth_half, [1, 1, 1])

        del a_mat, image, image_half
        del beta_mat

        a_mat_mod = self.only_reorganize_dimension(a_mat_mod)  # reorganize the dimension as 3 * m * n
        beta_mat_mod = self.only_reorganize_dimension(beta_mat_mod)  # reorganize the dimension as 3 * m * n
        unit_mat = self.only_reorganize_dimension(unit_mat)  # reorganize the dimension as 3 * m * n
        complex_noisy_img = self.only_reorganize_dimension(complex_noisy_img)  # reorganize the dimension as 3 * m * n

        image_numpy = self.only_reorganize_dimension(image_numpy)  # reorganize the dimension as 3 * m * n
        image_half_numpy = self.only_reorganize_dimension(image_half_numpy)  # reorganize the dimension as 3 * m * n
        haze_image = self.only_reorganize_dimension(haze_image)  # reorganize the dimension as 3 * m * n

        image_tensor = torch.from_numpy(image_numpy)  # convert into tensor
        image_half_tensor = torch.from_numpy(image_half_numpy)  # convert into tensor
        haze_image_tensor = torch.from_numpy(haze_image)  # convert into tensor

        a_mat_mod = torch.from_numpy(a_mat_mod)  # convert into tensor
        beta_mat_mod = torch.from_numpy(beta_mat_mod)  # convert into tensor
        unit_mat = torch.from_numpy(unit_mat)  # convert into tensor
        complex_image_tensor = torch.from_numpy(complex_noisy_img)  # convert into tensor
        
        del complex_noisy_img, image_numpy, haze_image, image_half_numpy

        return {'image_full': image_tensor, 'image_half': image_half_tensor, 'depth_half': depth_half,
                'haze_image': haze_image_tensor, 'beta': beta_mat_mod,
                'a_val': a_mat_mod, 'unit_mat': unit_mat, 'complex_noise_img': complex_image_tensor}

    def only_reorganize_dimension(self, data):
        data = np.swapaxes(data, 0, 2)  # making m * n *3
        data = np.swapaxes(data, 1, 2)  # making m * n *3
        return data


def another_simple_image_save(image, path):
    image = image.astype(np.float32)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = image.astype(np.uint8)
    cv2.imwrite(path, image)  

def create_reorganize_dimension_custom(data, m, n):
    data = np.reshape(data, [3, 1, 1])
    data = np.tile(data, [1, m, n])
    data = np.swapaxes(data, 0, 2)  # making m * n *3
    data = np.swapaxes(data, 0, 1)  # making m * n *3
    return data
   
def getTrainingTestingData(batch_size, transforms):
    
    transformed_images = Make3DDataset(transforms)

    return DataLoader(transformed_images, batch_size, shuffle=True,  num_workers=8, drop_last=True)


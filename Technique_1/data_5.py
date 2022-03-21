import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image

import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import cv2
from io import BytesIO
from sklearn.utils import shuffle
import random

from numpy import asarray
from numpy import save
from numpy import load

from numpy import savez_compressed

from ricardo_underwater_image import compute_complex_noise

beta_val_r = [0.0192, 0.0182, 0.0263, 0.0253]
beta_val_g = [0.0398, 0.0598, 0.0460, 0.0661]
beta_val_b = [0.357, 0.416, 0.528, 0.364, 0.423, 0.523]

# Here the code is very similar as the one in data_2.py but the only difference here is that, we have used here saved  
# "index_haze_image.npz", "index_complex_haze_image.npz" and "index_complex_depth_half_3d.npz" files, during training 


loader = transforms.Compose([
    transforms.ToTensor()])

un_loader = transforms.ToPILImage()


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = un_loader(image)
    return image

class RandomGamma(object):
    """
    Apply Random Gamma Correction to the images
    """
    def __init__(self, gamma=0):
        self.gamma = gamma

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if self.gamma == 0:
            return {'image': image, 'depth': depth}
        else:
            gamma_ratio = random.uniform(1 / self.gamma, self.gamma)
            return {'image': TF.adjust_gamma(image, gamma_ratio, gain=1),
                    'depth': depth}

class RandomHorizontalFlipCustom(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}


class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}


def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n')
                       if len(row) > 0))

    nyu2_train = shuffle(nyu2_train, random_state=0)

    #if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train

def loadZipToMemTest(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}

    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n')
                      if len(row) > 0))

    nyu2_test = shuffle(nyu2_test, random_state=0)

    print('Loaded ({0}).'.format(len(nyu2_test)))
    return data, nyu2_test


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, beta_mat_arr, a_mat_arr, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train

        self.beta_mat_arr = beta_mat_arr
        self.a_mat_arr = a_mat_arr
        self.transform = transform

    def __getitem__(self, idx):
        # idx = 49652  # **************************************

        haze_image_name = "/sanssauvegarde/homes/t20monda/save_water_data_test/" + str(idx) + "haze_image" + ".npy"
        complex_haze_image_name =  "/sanssauvegarde/homes/t20monda/save_water_data_test/" + str(idx) + "complex_haze_image_name" + ".npy"
        # orig_depth_image_name =  "/data/zenith/user/tmondal/save_data_water/all_data/" +  str(idx) + "orig_depth_image" + ".npy"
        # orig_image_name =  "/data/zenith/user/tmondal/save_data_water/all_data/" +  str(idx) + "orig_image" + ".npy"

        haze_image = load(haze_image_name)
        complex_noisy_img = load(complex_haze_image_name)

        # image = load(orig_image_name)
        # image = Image.fromarray(image) # converting into PIL image

        # depth = load(orig_depth_image_name)
        # depth = Image.fromarray(depth) # converting into PIL image

        sample = self.nyu_dataset[idx]
        image = Image.open(BytesIO(self.data[sample[0]]))
        depth = Image.open(BytesIO(self.data[sample[1]]))

        sample = {'image': image, 'depth': depth}
        if self.transform:
            sample = self.transform(sample)
        
        # all the image matrices are normalized here within 0-1
        image_full, image_half, depth_half_10_1000, depth_half_0_1 = sample['image_norm'], \
            sample['image_half_norm'], sample['depth_half_norm_10_1000'], sample['depth_half_norm_0_1']

        # self.create_image_to_verify(image_half, depth_half_0_1)
        # depth_half_0_1 = depth_half_0_1*10
        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]

        # depth_half_0_1_numpy = np.array(depth_half_0_1)
        del depth_half_0_1

        # need to modify the dimension ordering to use in opencv
        
        # depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 2)
        # depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 1)
        # depth_half_0_1_3d = cv2.cvtColor(depth_half_0_1_numpy, cv2.COLOR_GRAY2RGB)  # transformed into 3 channels
        # del depth_half_0_1_numpy
        # tx1 = np.exp(-np.multiply(beta_mat_mod, depth_half_0_1_3d))
        
        beta_mat = self.beta_mat_arr[idx]
        beta_mat_mod = self.create_reorganize_dimension(beta_mat, m, n)

        a_mat = self.a_mat_arr[idx]
        a_mat_mod = self.create_reorganize_dimension(a_mat, m, n)

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = self.create_reorganize_dimension(unit_mat, m, n)

        image_half_numpy = np.array(image_half)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)  # making m * n *3
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)  # making m * n *3
        
        del a_mat
        del beta_mat

        image_half_numpy = self.only_reorganize_dimension(image_half_numpy)  # reorganize the dimension as m*n*3
        haze_image = self.only_reorganize_dimension(haze_image)  # reorganize the dimension as m*n*3

        # depth_half_0_1_3d = self.only_reorganize_dimension(depth_half_0_1_3d)  # reorganize the dimension as m*n*3

        a_mat_mod = self.only_reorganize_dimension(a_mat_mod)  # reorganize the dimension as m*n*3
        beta_mat_mod = self.only_reorganize_dimension(beta_mat_mod)  # reorganize the dimension as m*n*3
        unit_mat = self.only_reorganize_dimension(unit_mat)  # reorganize the dimension as m*n*3
        complex_noisy_img = self.only_reorganize_dimension(complex_noisy_img)  # reorganize the dimension as m*n*3

        image_half_tensor = torch.from_numpy(image_half_numpy)  # convert into tensor
        haze_image_tensor = torch.from_numpy(haze_image)  # convert into tensor

        # depth_half_0_1_3d = torch.from_numpy(depth_half_0_1_3d)  # convert into tensor
        
        a_mat_mod = torch.from_numpy(a_mat_mod)  # convert into tensor
        beta_mat_mod = torch.from_numpy(beta_mat_mod)  # convert into tensor
        unit_mat = torch.from_numpy(unit_mat)  # convert into tensor
        complex_image_tensor = torch.from_numpy(complex_noisy_img)  # convert into tensor

        del complex_noisy_img

        return {'image': image_full, 'image_half': image_half_tensor, 'depth': depth_half_10_1000,
                'haze_image': haze_image_tensor, 'beta': beta_mat_mod,
                'a_val': a_mat_mod, 'unit_mat': unit_mat, 'complex_noise_img': complex_image_tensor}
                  

    def __len__(self):
        # print("The dataset length is :", len(self.nyu_dataset))
        return len(self.nyu_dataset)

    def create_image_to_verify(self, image, depth):
        image = (image - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255
        depth = (depth - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255

        image_numpy = np.array(image)
        image_numpy = np.swapaxes(image_numpy, 0, 2)
        image_numpy = np.swapaxes(image_numpy, 0, 1)

        depth_numpy = np.array(depth)
        depth_numpy = np.swapaxes(depth_numpy, 0, 2)
        depth_numpy = np.swapaxes(depth_numpy, 0, 1)

        depth_numpy = depth_numpy[:, :, 0]

        imgD_Norm = depth_numpy
        imgD_Norm = imgD_Norm + 1
        imgD_Norm = (imgD_Norm / np.min([np.max(imgD_Norm), 255])) * 255  # values are > 0 & <= 255

        imgRGB = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        fname_orig_imag = '/home/tmondal/Documents/Dataset/Dehazing/NYU_Depth_V2/test_orig_imag_NN.png'
        imgRGB = imgRGB.astype(np.uint8)
        cv2.imwrite(fname_orig_imag, imgRGB)

        # imgD_Norm = cv2.cvtColor(imgD_Norm, cv2.COLOR_BGR2RGB)
        imgD_Norm = cv2.normalize(imgD_Norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        imgD_Norm = imgD_Norm.astype(np.uint8)
        fname_orig_depth = '/home/tmondal/Documents/Dataset/Dehazing/NYU_Depth_V2/test_orig_depth_NN.png'
        cv2.imwrite(fname_orig_depth, imgD_Norm)

    def create_reorganize_dimension(self, data, m, n):
        data = np.reshape(data, [3, 1, 1])
        data = np.tile(data, [1, m, n])
        data = np.swapaxes(data, 0, 2)  # making m * n *3
        data = np.swapaxes(data, 0, 1)  # making m * n *3
        return data

    def only_reorganize_dimension(self, data):
        data = np.swapaxes(data, 0, 2)  # making m * n *3
        data = np.swapaxes(data, 1, 2)  # making m * n *3
        return data

    def simple_image_save(self, image, path):
        image = (image - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255
        # image = np.swapaxes(image, 0, 2)
        # image = np.swapaxes(image, 1, 2)
        image = image.astype(np.float32)
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bringing back in the range of 0-255
        imgRGB = imgRGB.astype(np.uint8)
        cv2.imwrite(path, imgRGB)

    def another_simple_image_save(self, image, path):
        image = image.astype(np.float32)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = image.astype(np.uint8)
        cv2.imwrite(path, image)    


class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image_half = image.resize((320, 240))  # now it is PIL image, hence we can resize the image
        depth_half = depth.resize((320, 240))

        image_half_norm_0_1 = self.to_tensor(image_half).float()
        image_norm_0_1 = self.to_tensor(image).float()
        depth_half_0_1 = self.to_tensor(depth_half).float()

        if self.is_test:
            depth_half_norm_10_1000 = self.to_tensor(depth_half).float() / 1000
        else:
            depth_half_norm_10_1000 = self.to_tensor(depth_half).float() * 1000

        # put in expected range
        depth_half_norm_10_1000 = torch.clamp(depth_half_norm_10_1000, 10, 1000)
        return {'image_norm': image_norm_0_1, 'image_half_norm': image_half_norm_0_1,
                'depth_half_norm_10_1000': depth_half_norm_10_1000, 'depth_half_norm_0_1': depth_half_0_1}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


def ToTensorCustom(sample):
    image, depth = sample['image'], sample['depth']

    image_half = image.resize((320, 240))  # now it is PIL image, hence we can resize the image
    depth_half = depth.resize((320, 240))

    image_half_norm_0_1 = to_tensor_custom(image_half).float()
    image_norm_0_1 = to_tensor_custom(image).float()
    depth_half_0_1 = to_tensor_custom(depth_half).float()

    depth_half_norm_10_1000 = to_tensor_custom(depth_half).float() * 1000

    # put in expected range
    depth_half_norm_10_1000 = torch.clamp(depth_half_norm_10_1000, 10, 1000)
    return {'image_norm': image_norm_0_1, 'image_half_norm': image_half_norm_0_1,
            'depth_half_norm_10_1000': depth_half_norm_10_1000, 'depth_half_norm_0_1': depth_half_0_1}


def to_tensor_custom(pic):
    if not (_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError(
            'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))

        return img.float().div(255)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(
            torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)

    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def create_reorganize_dimension_custom(data, m, n):
    data = np.reshape(data, [3, 1, 1])
    data = np.tile(data, [1, m, n])
    data = np.swapaxes(data, 0, 2)  # making m * n *3
    data = np.swapaxes(data, 0, 1)  # making m * n *3
    return data


def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor()
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlipCustom(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])
   

def getTrainingTestingData(batch_size):
    data, nyu2_train = loadZipToMem('/users/local/t20monda/nyu_data.zip')

    beta_mat_arr = load('/homes/t20monda/DenseDepth_1/Beta_Mat.npy')
    a_mat_arr = load('/homes/t20monda/DenseDepth_1/A_Mat.npy')

    transformed_training = depthDatasetMemory(data, nyu2_train, beta_mat_arr, a_mat_arr,
                                              transform=getDefaultTrainTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True, drop_last=True)

def getTestingDataOnly(batch_size):
    data, nyu2_test = loadZipToMemTest('/users/local/t20monda/nyu_data.zip')

    beta_mat_arr = load('/homes/t20monda/DenseDepth_1/Beta_Mat.npy')
    a_mat_arr = load('/homes/t20monda/DenseDepth_1/A_Mat.npy')
    
    transformed_testing = depthDatasetMemory(data, nyu2_test, beta_mat_arr, a_mat_arr,
                                              transform=getNoTransform())

    return DataLoader(transformed_testing, batch_size, shuffle=True, drop_last=True)

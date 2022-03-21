import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.utils import save_image

import torch.nn.functional as F
from PIL import Image

from io import BytesIO
from sklearn.utils import shuffle
import random

from numpy import asarray
from numpy import save
from numpy import load

from scipy import io

from numpy import savez_compressed

import cv2
import glob

from ricardo_underwater_image import compute_complex_noise

beta_val_r = [0.0192, 0.0182, 0.0263, 0.0253]
beta_val_g = [0.0398, 0.0598, 0.0460, 0.0661]
beta_val_b = [0.357, 0.416, 0.528, 0.364, 0.423, 0.523]

# loader uses the transforms function that comes with torchvision
# here we will pre-compute the images and will save them from before. Later, we will just load the images inside the get_item() function
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


def generate_and_save_atmosphere_light_beta():
    data, nyu2_train = loadZipToMem('/home/tmondal/Python_Projects/water_correction/nyu_data.zip')
    data_len = len(nyu2_train)
    del data, nyu2_train

    beta_arr = []
    a_mat_arr = []
    for x in range(data_len):

        rand_indx_r = random.randint(0, len(beta_val_r) - 1)
        rand_indx_g = random.randint(0, len(beta_val_g) - 1)
        rand_indx_b = random.randint(0, len(beta_val_b) - 1)

        beta_mat = [beta_val_r[rand_indx_r], beta_val_g[rand_indx_g], beta_val_b[rand_indx_b]]
        beta_arr.append(beta_mat)

        # a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        rand_val_a = random.uniform(0, 1)
        a_mat = [rand_val_a, rand_val_a, rand_val_a]

        a_mat_arr.append(a_mat)

    save('A_Mat.npy', a_mat_arr)
    save('Beta_Mat.npy', beta_arr)


def generate_and_save_atmosphere_light_beta_make_3D_Test():

    train_images = glob.glob('/sanssauvegarde/homes/t20monda/Make3D/Test134/*.jpg')
    data_len = len(train_images)
    del train_images

    beta_arr = []
    a_mat_arr = []
    for x in range(data_len):

        rand_indx_r = random.randint(0, len(beta_val_r) - 1)
        rand_indx_g = random.randint(0, len(beta_val_g) - 1)
        rand_indx_b = random.randint(0, len(beta_val_b) - 1)

        beta_mat = [beta_val_r[rand_indx_r], beta_val_g[rand_indx_g], beta_val_b[rand_indx_b]]
        beta_arr.append(beta_mat)

        # a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        rand_val_a = random.uniform(0, 1)
        a_mat = [rand_val_a, rand_val_a, rand_val_a]

        a_mat_arr.append(a_mat)

    save('/homes/t20monda/DenseDepth_1/A_Mat_Make_3D_Test.npy', a_mat_arr)
    save('/homes/t20monda/DenseDepth_1/Beta_Mat_Make_3D_Test.npy', beta_arr)


def generate_and_save_atmosphere_light_beta_make_3D():

    train_images = glob.glob('/sanssauvegarde/homes/t20monda/Make3D/Train400Img/*.jpg')
    data_len = len(train_images)
    del train_images

    beta_arr = []
    a_mat_arr = []
    for x in range(data_len):

        rand_indx_r = random.randint(0, len(beta_val_r) - 1)
        rand_indx_g = random.randint(0, len(beta_val_g) - 1)
        rand_indx_b = random.randint(0, len(beta_val_b) - 1)

        beta_mat = [beta_val_r[rand_indx_r], beta_val_g[rand_indx_g], beta_val_b[rand_indx_b]]
        beta_arr.append(beta_mat)

        # a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]

        rand_val_a = random.uniform(0, 1)
        a_mat = [rand_val_a, rand_val_a, rand_val_a]

        a_mat_arr.append(a_mat)

    save('/homes/t20monda/DenseDepth_1/A_Mat_Make_3D.npy', a_mat_arr)
    save('/homes/t20monda/DenseDepth_1/Beta_Mat_Make_3D.npy', beta_arr)


def generate_and_save_haze_image_test(stIndex, endIndex):
    data, nyu_dataset = loadZipToMemTest('/users/local/t20monda/nyu_data.zip')
    a_mat_arr = load('/homes/t20monda/DenseDepth_1/A_Mat.npy')
    beta_mat_arr = load('/homes/t20monda/DenseDepth_1/Beta_Mat.npy')

    # keep_all_haze_image = []
    # keep_all_complex_haze_image = []
    # keep_all_depth_half_3d = []
    # for idx in range(len(nyu_dataset)):
    for idx in range(stIndex, endIndex):
        sample = nyu_dataset[idx]
        image = Image.open(BytesIO(data[sample[0]]))
        depth = Image.open(BytesIO(data[sample[1]]))
        sample_duo = {'image': image, 'depth': depth}

        sample = ToTensorCustom(sample_duo, False)

        # all the image matrices are normalized here within 0-1
        image_full, image_half, depth_half_10_1000, depth_half_0_1 = sample['image_norm'], \
            sample['image_half_norm'], sample['depth_half_norm_10_1000'], sample['depth_half_norm_0_1']

        depth_half_0_1 = depth_half_0_1*10
        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]

        depth_half_0_1_numpy = np.array(depth_half_0_1)
        del depth_half_0_1

        # need to modify the dimension ordering to use in opencv
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 2)
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 1)
        depth_half_0_1_3d = cv2.cvtColor(depth_half_0_1_numpy, cv2.COLOR_GRAY2RGB)  # transformed into 3 channels

        del depth_half_0_1_numpy

        beta_mat = beta_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)

        a_mat = a_mat_arr[idx]
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)

        tx1 = np.exp(-np.multiply(beta_mat_mod, depth_half_0_1_3d))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = create_reorganize_dimension_custom(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))

        image_half_numpy = np.array(image_half)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)  # making m * n *3
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)  # making m * n *3
        haze_image = np.add((np.multiply(image_half_numpy, tx1)), second_term)

        complex_noisy_img = compute_complex_noise(image_half_numpy, depth_half_0_1_3d[:, :, 0]/10, beta_mat,
                                                  a_mat)
        complex_noisy_img = complex_noisy_img/255

        del a_mat
        del beta_mat

        # asarray will help to convert the input into an array
        # keep_all_haze_image.append(asarray(haze_image))
        # keep_all_complex_haze_image.append(asarray(complex_noisy_img))
        # keep_all_depth_half_3d.append(asarray(depth_half_0_1_3d))

        asarray(haze_image)  # values are between 0-1
        asarray(complex_noisy_img)  # values are between 0-1
        asarray(depth)
        asarray(image)
        
        haze_image_name = "/sanssauvegarde/homes/t20monda/save_water_data_test/" + str(idx) + "haze_image" + ".npy"
        complex_haze_image_name =  "/sanssauvegarde/homes/t20monda/save_water_data_test/" + str(idx) + "complex_haze_image_name" + ".npy"
        # orig_depth_image_name =  "/sanssauvegarde/homes/t20monda/save_water_data_test/" +  str(idx) + "orig_depth_image" + ".npy"
        # orig_image_name =  "/sanssauvegarde/homes/t20monda/save_water_data_test/" +  str(idx) + "orig_image" + ".npy"

        save(haze_image_name, haze_image)
        save(complex_haze_image_name, complex_noisy_img)


def generate_and_save_ricardo_image_make_3D_Test(stIndex, endIndex):
    
    train_images = glob.glob('/sanssauvegarde/homes/t20monda/Make3D/Test134/*.jpg')
    train_depth = glob.glob('/sanssauvegarde/homes/t20monda/Make3D/Test134Depth/Gridlaserdata/*.mat')

    train_images = sorted(train_images, key=lambda p: p.split('/')[-1].split('img-')[-1])
    train_depth = sorted(train_depth, key=lambda p: p.split('/')[-1].split('depth_sph_corr-')[-1])

    a_mat_arr = load('/homes/t20monda/DenseDepth_1/A_Mat_Make_3D_Test.npy')
    beta_mat_arr = load('/homes/t20monda/DenseDepth_1/Beta_Mat_Make_3D_Test.npy')

    for idx in range(stIndex, endIndex):

        image = cv2.imread(train_images[idx])
        depth = io.loadmat(train_depth[idx])['Position3DGrid'][:, :, 3]

        image = cv2.resize(image, (460, 345), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (460, 345), interpolation=cv2.INTER_LINEAR)

        image_half = cv2.resize(image, (230, 173), interpolation=cv2.INTER_LINEAR)
        depth_half = cv2.resize(depth, (230, 173), interpolation=cv2.INTER_LINEAR)

        del image, depth

        image_half = torch.from_numpy(image_half).permute(2, 0, 1).float() / 255
        depth_half = torch.from_numpy(depth_half).float() / 80

        depth_half = torch.unsqueeze(depth_half, 0)
        depth_half = torch.tile(depth_half, [3, 1, 1])

        depth_half.requires_grad = False
        image_half.requires_grad = False

        depth_half_0_1 = depth_half
        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]

        # save_image(image_half, 'img_half.png')
        # save_image(depth_half, 'depth_half.png')

        image_half = image_half.numpy() # converting into numpy array
        image_half = np.swapaxes(image_half, 0, 2)  # making m * n *3
        image_half = np.swapaxes(image_half, 0, 1)  # making m * n *3

        depth_half = depth_half.numpy() # converting into numpy array
        depth_half = np.swapaxes(depth_half, 0, 2)  # making m * n *3
        depth_half = np.swapaxes(depth_half, 0, 1)  # making m * n *3

        # another_simple_image_save(image_half, "image_numpy.png")
        # another_simple_image_save(depth_half, "depth_numpy.png")

        beta_mat = beta_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)

        a_mat = a_mat_arr[idx]
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)

        tx1 = np.exp(-np.multiply(beta_mat_mod, depth_half))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = create_reorganize_dimension_custom(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))
        haze_image = np.add((np.multiply(image_half, tx1)), second_term)

        complex_noisy_img = compute_complex_noise(image_half, depth_half[:, :, 0], beta_mat,
                                                  a_mat)

        # another_simple_image_save(haze_image, "formed_haze_numpy.png")
        # another_simple_image_save(complex_noisy_img, "formed_complex_numpy.png")

        complex_noisy_img = complex_noisy_img/255

        del a_mat
        del beta_mat

        asarray(haze_image)  # values are between 0-1
        asarray(complex_noisy_img)  # values are between 0-1

        haze_image_name = "/sanssauvegarde/homes/t20monda/all_data_make_3D_test/" + str(idx) + "haze_image" + ".npy"
        complex_haze_image_name =  "/sanssauvegarde/homes/t20monda/all_data_make_3D_test/" + str(idx) + "complex_haze_image_name" + ".npy"
        
        save(haze_image_name, haze_image)
        save(complex_haze_image_name, complex_noisy_img)


def generate_and_save_ricardo_image_make_3D(stIndex, endIndex):
    
    train_images = glob.glob('/sanssauvegarde/homes/t20monda/Make3D/Train400Img/*.jpg')
    train_depth = glob.glob('/sanssauvegarde/homes/t20monda/Make3D/Train400Depth/*.mat')

    train_images = sorted(train_images, key=lambda p: p.split('/')[-1].split('img-')[-1])
    train_depth = sorted(train_depth, key=lambda p: p.split('/')[-1].split('depth_sph_corr-')[-1])

    a_mat_arr = load('/homes/t20monda/DenseDepth_1/A_Mat_Make_3D.npy')
    beta_mat_arr = load('/homes/t20monda/DenseDepth_1/Beta_Mat_Make_3D.npy')

    for idx in range(stIndex, endIndex):

        image = cv2.imread(train_images[idx])
        depth = io.loadmat(train_depth[idx])['Position3DGrid'][:, :, 3]

        image = cv2.resize(image, (460, 345), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (460, 345), interpolation=cv2.INTER_LINEAR)

        image_half = cv2.resize(image, (230, 173), interpolation=cv2.INTER_LINEAR)
        depth_half = cv2.resize(depth, (230, 173), interpolation=cv2.INTER_LINEAR)

        del image, depth

        image_half = torch.from_numpy(image_half).permute(2, 0, 1).float() / 255
        depth_half = torch.from_numpy(depth_half).float() / 80

        depth_half = torch.unsqueeze(depth_half, 0)
        depth_half = torch.tile(depth_half, [3, 1, 1])

        depth_half.requires_grad = False
        image_half.requires_grad = False

        depth_half_0_1 = depth_half
        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]

        # save_image(image_half, 'img_half.png')
        # save_image(depth_half, 'depth_half.png')

        image_half = image_half.numpy() # converting into numpy array
        image_half = np.swapaxes(image_half, 0, 2)  # making m * n *3
        image_half = np.swapaxes(image_half, 0, 1)  # making m * n *3

        depth_half = depth_half.numpy() # converting into numpy array
        depth_half = np.swapaxes(depth_half, 0, 2)  # making m * n *3
        depth_half = np.swapaxes(depth_half, 0, 1)  # making m * n *3

        # another_simple_image_save(image_half, "image_numpy.png")
        # another_simple_image_save(depth_half, "depth_numpy.png")

        beta_mat = beta_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)

        a_mat = a_mat_arr[idx]
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)

        tx1 = np.exp(-np.multiply(beta_mat_mod, depth_half))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = create_reorganize_dimension_custom(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))
        haze_image = np.add((np.multiply(image_half, tx1)), second_term)

        complex_noisy_img = compute_complex_noise(image_half, depth_half[:, :, 0], beta_mat,
                                                  a_mat)

        # another_simple_image_save(haze_image, "formed_haze_numpy.png")
        # another_simple_image_save(complex_noisy_img, "formed_complex_numpy.png")

        complex_noisy_img = complex_noisy_img/255

        del a_mat
        del beta_mat

        asarray(haze_image)  # values are between 0-1
        asarray(complex_noisy_img)  # values are between 0-1

        haze_image_name = "/sanssauvegarde/homes/t20monda/all_data_make_3D/" + str(idx) + "haze_image" + ".npy"
        complex_haze_image_name =  "/sanssauvegarde/homes/t20monda/all_data_make_3D/" + str(idx) + "complex_haze_image_name" + ".npy"
        
        save(haze_image_name, haze_image)
        save(complex_haze_image_name, complex_noisy_img)


def another_simple_image_save(image, path):
    image = image.astype(np.float32)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = image.astype(np.uint8)
    cv2.imwrite(path, image)   


def generate_and_save_haze_image(stIndex, endIndex):
    data, nyu_dataset = loadZipToMem('/users/local/t20monda/nyu_data.zip')
    a_mat_arr = load('/homes/t20monda/DenseDepth_1/A_Mat.npy')
    beta_mat_arr = load('/homes/t20monda/DenseDepth_1/Beta_Mat.npy')

    # keep_all_haze_image = []
    # keep_all_complex_haze_image = []
    # keep_all_depth_half_3d = []
    # for idx in range(len(nyu_dataset)):
    for idx in range(stIndex, endIndex):
        sample = nyu_dataset[idx]
        image = Image.open(BytesIO(data[sample[0]]))
        depth = Image.open(BytesIO(data[sample[1]]))
        sample_duo = {'image': image, 'depth': depth}

        sample = ToTensorCustom(sample_duo, False)

        # all the image matrices are normalized here within 0-1
        image_full, image_half, depth_half_10_1000, depth_half_0_1 = sample['image_norm'], \
            sample['image_half_norm'], sample['depth_half_norm_10_1000'], sample['depth_half_norm_0_1']

        depth_half_0_1 = depth_half_0_1*10
        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]

        depth_half_0_1_numpy = np.array(depth_half_0_1)
        del depth_half_0_1

        # need to modify the dimension ordering to use in opencv
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 2)
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 1)
        depth_half_0_1_3d = cv2.cvtColor(depth_half_0_1_numpy, cv2.COLOR_GRAY2RGB)  # transformed into 3 channels

        del depth_half_0_1_numpy

        beta_mat = beta_mat_arr[idx]
        beta_mat_mod = create_reorganize_dimension_custom(beta_mat, m, n)

        a_mat = a_mat_arr[idx]
        a_mat_mod = create_reorganize_dimension_custom(a_mat, m, n)

        tx1 = np.exp(-np.multiply(beta_mat_mod, depth_half_0_1_3d))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = create_reorganize_dimension_custom(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))

        image_half_numpy = np.array(image_half)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)  # making m * n *3
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)  # making m * n *3
        haze_image = np.add((np.multiply(image_half_numpy, tx1)), second_term)

        complex_noisy_img = compute_complex_noise(image_half_numpy, depth_half_0_1_3d[:, :, 0]/10, beta_mat,
                                                  a_mat)
        complex_noisy_img = complex_noisy_img/255

        del a_mat
        del beta_mat

        # asarray will help to convert the input into an array
        # keep_all_haze_image.append(asarray(haze_image))
        # keep_all_complex_haze_image.append(asarray(complex_noisy_img))
        # keep_all_depth_half_3d.append(asarray(depth_half_0_1_3d))

        asarray(haze_image)  # values are between 0-1
        asarray(complex_noisy_img)  # values are between 0-1
        asarray(depth)
        asarray(image)

        haze_image_name = "/data/zenith/user/tmondal/save_data_water/all_data/" + str(idx) + "haze_image" + ".npy"
        complex_haze_image_name =  "/data/zenith/user/tmondal/save_data_water/all_data/" + str(idx) + "complex_haze_image_name" + ".npy"
        orig_depth_image_name =  "/data/zenith/user/tmondal/save_data_water/all_data/" +  str(idx) + "orig_depth_image" + ".npy"
        orig_image_name =  "/data/zenith/user/tmondal/save_data_water/all_data/" +  str(idx) + "orig_image" + ".npy"

        save(haze_image_name, haze_image)
        save(complex_haze_image_name, complex_noisy_img)
        save(orig_depth_image_name, depth)
        save(orig_image_name, image)

        # print("I am done with image no : ", idx)

    # save to npy file
    # savez_compressed('all_haze_image.npz', keep_all_haze_image)
    # savez_compressed('all_complex_haze_image.npz', keep_all_complex_haze_image)
    # savez_compressed('all_complex_depth_half_3d.npz', keep_all_depth_half_3d)


def testing_image_saving_code(self, idx):
    # idx = 49652  # **************************************

    haze_image_name = "/data/zenith/user/tmondal/save_data_water/all_data/" + str(idx) + "haze_image" + ".npy"
    complex_haze_image_name =  "/data/zenith/user/tmondal/save_data_water/all_data/" + str(idx) + "complex_haze_image_name" + ".npy"
    orig_depth_image_name =  "/data/zenith/user/tmondal/save_data_water/all_data/" +  str(idx) + "orig_depth_image" + ".npy"
    orig_image_name =  "/data/zenith/user/tmondal/save_data_water/all_data/" +  str(idx) + "orig_image" + ".npy"

    haze_image = load(haze_image_name)
    complex_noisy_img = load(complex_haze_image_name)
    image = load(orig_image_name)
    depth = load(orig_depth_image_name)

    # sample = self.nyu_dataset[idx]
    # image = Image.open(BytesIO(self.data[sample[0]]))
    # depth = Image.open(BytesIO(self.data[sample[1]]))
    sample = {'image': image, 'depth': depth}
    if self.transform:
        sample = self.transform(sample)
    
    # all the image matrices are normalized here within 0-1
    image_full, image_half, depth_half_10_1000, depth_half_0_1 = sample['image_norm'], \
        sample['image_half_norm'], sample['depth_half_norm_10_1000'], sample['depth_half_norm_0_1']

    # self.create_image_to_verify(image_half, depth_half_0_1)
    depth_half_0_1 = depth_half_0_1*10
    m = depth_half_0_1.shape[1]
    n = depth_half_0_1.shape[2]

    depth_half_0_1_numpy = np.array(depth_half_0_1)
    del depth_half_0_1

    # need to modify the dimension ordering to use in opencv
    depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 2)
    depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 1)
    depth_half_0_1_3d = cv2.cvtColor(depth_half_0_1_numpy, cv2.COLOR_GRAY2RGB)  # transformed into 3 channels
    del depth_half_0_1_numpy

    beta_mat = self.beta_mat_arr[idx]
    beta_mat_mod = self.create_reorganize_dimension(beta_mat, m, n)

    a_mat = self.a_mat_arr[idx]
    a_mat_mod = self.create_reorganize_dimension(a_mat, m, n)

    tx1 = np.exp(-np.multiply(beta_mat_mod, depth_half_0_1_3d))

    # generate 3D unit matrix
    unit_mat = [1.0, 1.0, 1.0]
    unit_mat = self.create_reorganize_dimension(unit_mat, m, n)

    second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))

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
            'depth_norm_simple': depth_half_0_1_3d, 'haze_image': haze_image_tensor, 'beta': beta_mat_mod,
            'a_val': a_mat_mod, 'unit_mat': unit_mat, 'complex_noise_img': complex_image_tensor}


class depthDatasetMemory(Dataset):
    def __init__(self, data, nyu2_train, transform=None):
        self.data, self.nyu_dataset = data, nyu2_train
        # beta_arr = []
        # a_mat_arr = []
        # for x in range(len(nyu2_train)):
        #     rand_indx_r = random.randint(0, len(beta_val_r) - 1)
        #     rand_indx_g = random.randint(0, len(beta_val_g) - 1)
        #     rand_indx_b = random.randint(0, len(beta_val_b) - 1)
        #     beta_mat = [beta_val_r[rand_indx_r], beta_val_g[rand_indx_g], beta_val_b[rand_indx_b]]
        #     beta_arr.append(beta_mat)

        #     a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        #     a_mat_arr.append(a_mat)

        self.beta_mat_arr = load('Beta_Mat.npy')
        self.a_mat_arr = load('A_Mat.npy')
        self.transform = transform

    def __getitem__(self, idx):
        # idx = 49652  # **************************************
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
        m = depth_half_0_1.shape[1]
        n = depth_half_0_1.shape[2]

        depth_half_0_1 = depth_half_0_1*10
        depth_half_0_1_numpy = np.array(depth_half_0_1)
        del depth_half_0_1
        # need to modify the dimension ordering to use in opencv
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 2)
        depth_half_0_1_numpy = np.swapaxes(depth_half_0_1_numpy, 0, 1)
        depth_half_0_1_3d = cv2.cvtColor(depth_half_0_1_numpy, cv2.COLOR_GRAY2RGB)  # transformed into 3 channels

        del depth_half_0_1_numpy

        beta_mat = self.beta_mat_arr[idx]
        beta_mat_mod = self.create_reorganize_dimension(beta_mat, m, n)

        # a_mat = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        a_mat = self.a_mat_arr[idx]
        a_mat_mod = self.create_reorganize_dimension(a_mat, m, n)

        tx1 = np.exp(-np.multiply(beta_mat_mod, depth_half_0_1_3d))

        # generate 3D unit matrix
        unit_mat = [1.0, 1.0, 1.0]
        unit_mat = self.create_reorganize_dimension(unit_mat, m, n)

        second_term = np.multiply(a_mat_mod, (np.subtract(unit_mat, tx1)))

        image_half_numpy = np.array(image_half)
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 2)  # making m * n *3
        image_half_numpy = np.swapaxes(image_half_numpy, 0, 1)  # making m * n *3
        haze_image = np.add((np.multiply(image_half_numpy, tx1)), second_term)

        # self.another_simple_image_save(haze_image, '/homes/t20monda/Water_Correction_Test_1/DenseDepth/haze_image_NN.png')
        # self.another_simple_image_save(image_half_numpy, '/homes/t20monda/Water_Correction_Test_1/DenseDepth/orig_imag_NN.png')
        # self.another_simple_image_save(depth_half_0_1_3d, '/homes/t20monda/Water_Correction_Test_1/DenseDepth/depth_imag_NN.png')

        complex_noisy_img = compute_complex_noise(image_half_numpy, depth_half_0_1_3d[:, :, 0]/10, beta_mat,
                                                  a_mat)
        complex_noisy_img = complex_noisy_img/255
        del a_mat
        del beta_mat

        image_half_numpy = self.only_reorganize_dimension(image_half_numpy)  # reorganize the dimension as m*n*3
        haze_image = self.only_reorganize_dimension(haze_image)  # reorganize the dimension as m*n*3
        depth_half_0_1_3d = self.only_reorganize_dimension(depth_half_0_1_3d)  # reorganize the dimension as m*n*3
        a_mat_mod = self.only_reorganize_dimension(a_mat_mod)  # reorganize the dimension as m*n*3
        beta_mat_mod = self.only_reorganize_dimension(beta_mat_mod)  # reorganize the dimension as m*n*3
        unit_mat = self.only_reorganize_dimension(unit_mat)  # reorganize the dimension as m*n*3
        complex_noisy_img = self.only_reorganize_dimension(complex_noisy_img)  # reorganize the dimension as m*n*3

        image_half_tensor = torch.from_numpy(image_half_numpy)  # convert into tensor
        haze_image_tensor = torch.from_numpy(haze_image)  # convert into tensor
        depth_half_0_1_3d = torch.from_numpy(depth_half_0_1_3d)  # convert into tensor
        a_mat_mod = torch.from_numpy(a_mat_mod)  # convert into tensor
        beta_mat_mod = torch.from_numpy(beta_mat_mod)  # convert into tensor
        unit_mat = torch.from_numpy(unit_mat)  # convert into tensor
        complex_image_tensor = torch.from_numpy(complex_noisy_img)  # convert into tensor

        # haze_image_cpu = haze_image_tensor.cpu()
        # save_image(haze_image_cpu, './haze_image_tensor.png')

        # image_half_cpu = image_half_tensor.cpu()
        # save_image(image_half_cpu, './image_half_tensor.png')

        # depth_half_cpu = depth_half_0_1_3d.cpu()
        # save_image(depth_half_cpu, './depth_half_tensor.png')

        del complex_noisy_img

        return {'image': image_full, 'image_half': image_half_tensor, 'depth': depth_half_10_1000,
                'depth_norm_simple': depth_half_0_1_3d, 'haze_image': haze_image_tensor, 'beta': beta_mat_mod,
                'a_val': a_mat_mod, 'unit_mat': unit_mat, 'complex_noise_img': complex_image_tensor}

    def __len__(self):
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


def ToTensorCustom(sample, is_test):
    image, depth = sample['image'], sample['depth']

    image_half = image.resize((320, 240))  # now it is PIL image, hence we can resize the image
    depth_half = depth.resize((320, 240))

    image_half_norm_0_1 = to_tensor_custom(image_half).float()
    image_norm_0_1 = to_tensor_custom(image).float()
    depth_half_0_1 = to_tensor_custom(depth_half).float()

    if is_test:
        depth_half_norm_10_1000 = to_tensor_custom(depth_half).float() / 1000
    else:
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
        ToTensor(is_test=is_test)
    ])


def getDefaultTrainTransform():
    return transforms.Compose([
        RandomHorizontalFlipCustom(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])


def getTrainingTestingData(batch_size):
    data, nyu2_train = loadZipToMem('/users/local/t20monda/nyu_data.zip')

    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_train, transform=getNoTransform())

    return DataLoader(transformed_training, batch_size, shuffle=True), DataLoader(transformed_testing, batch_size,
                                                                                  shuffle=False)

def getTestingDataOnly(batch_size):
    data, nyu2_test = loadZipToMemTest('/users/local/t20monda/nyu_data.zip')
    
    transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform(True))
    return DataLoader(transformed_testing, batch_size, shuffle=True, drop_last=True)

from matplotlib import pyplot as plt
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from io import BytesIO
from sklearn.utils import shuffle
import random
from sklearn.utils import shuffle
import random
import time


# here the objective of the code is to apply the technique on the bigger NYU dataset (50000 images)
# hence we have adapted the code accordingly

# Parameters
# pExp = 5.0/(1*5*5*5*5)

gamma = [0.03, 0.009, 0.043]
# alpha = [0.012, 0.012, 0.012]

beta = [0.020, 0.005, 0.010]  # beta de ricardo

# A_light = [00.0, 120.0, 30.0]
turbu_p = [0.15, 1.5]  # noise amount, est. dev.
turbu_c = [34.0, 200.0, 201.0]

u = 0.98  # weighting between scattering + attenuation image vs particle noise image

s = 5  # multiplier for particle noise image
depht_add = 80  # addition to values of depht map (a minimum distance for map)
depht_levels = 16
kern_size = 10

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


# Compute scatering diffusion for all pixels (Eq. 2, 3) using depht levels
def scatterPsdOp(rgb, depht, gamma, alpha, win_size, d_levels):
    dim = rgb.shape
    psd = np.zeros(dim)

    d_min = torch.min(depht)
    d_max = torch.max(depht)

    img_sum = torch.zeros(dim[0], dim[1], 3)

    for n in range(d_levels):
        img_l = torch.zeros(dim[0], dim[1])
        ker = torch.zeros(2 * win_size + 1, 2 * win_size + 1)

        di_min = ((d_max - d_min) / d_levels) * n + d_min
        di_max = ((d_max - d_min) / d_levels) * (n + 1) + d_min
        di_m = (di_min + di_max) / 2

        for c in range(3):

            # compute kernel
            for i in range(2 * win_size + 1):
                for j in range(2 * win_size + 1):
                    pos = [i - win_size, j - win_size]
                    v0 = -(np.power(pos[0], 2) + np.power(pos[1], 2))
                    v1 = np.exp(v0 / (2 * np.power(gamma[c] * di_m, 2)))
                    v2 = 1 / ((2 * np.pi) * np.power(gamma[c] * di_m, 1))
                    v3 = v1 * v2
                    ker[i, j] = v3

            ker = ker / torch.sum(ker)

            # extract pixels in depht range
            img_t = rgb[:, :, c]
            dht = (depht >= di_min).data.numpy().astype(np.float32)
            dht = np.multiply(dht, (depht < di_max).data.numpy().astype(np.float32))
            img_l = torch.from_numpy(np.multiply(dht, img_t.data.numpy()))

            # Convolve
            va = Variable(ker.view(1, 1, 2 * win_size + 1, 2 * win_size + 1))
            vb = Variable(img_l.view(1, 1, dim[0], dim[1]))
            img_c = F.conv2d(vb, va, padding=win_size)
            img_c = img_c[0, 0, :, :]

            img_sum[:, :, c] += img_c
            img_sum[img_sum > 255] = 255  # for saturated values

    return img_sum


# Compute degradation caused by colored particle turbidity
def turbidDeg(rgb, depht, turbu_p, turbu_c):
    dim = rgb.shape
    s_vs_p = 0.5
    out = np.zeros(rgb.shape)

    # Salt mode
    num_salt = np.ceil(turbu_p[0] * rgb.size * s_vs_p * 3)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in rgb.shape]

    for i in range(len(coords[0])):
        out[coords[0][i], coords[1][i], coords[2][i]] = 1.0

    out[:, :, 0] = out[:, :, 0] * turbu_c[0]
    out[:, :, 1] = out[:, :, 1] * turbu_c[1]
    out[:, :, 2] = out[:, :, 2] * turbu_c[2]

    # pass a gaussian filter (blurring) to noisy points
    # kernel = np.ones((5,5),np.float32)/25
    # out = cv2.filter2D(out,-1,kernel)
    out = cv2.GaussianBlur(out, (9, 9), turbu_p[1])

    return out


# Compute total arriving intensity for a pixel (Eq. 4)
def outTotal(rgb, depht, I_out, gamma, alpha, beta, A_light):
    dim = rgb.shape
    I_total = np.zeros(rgb.shape)
    # I_out = np.zeros(rgb.shape)

    for c in range(3):
        for i in range(0, dim[0], 1):
            for j in range(0, dim[1], 1):
                s = [i, j]
                v1 = np.exp(-alpha[c] * depht[s[0]][s[1]]) * rgb[s[0]][s[1]][c]
                v2 = np.exp(-beta[c] * depht[s[0]][s[1]])
                I_total[s[0]][s[1]][c] = (I_out[s[0]][s[1]][c] + v1) * v2 + (1 - v2) * A_light[c]

    return I_total


def processImg(imgD_Norm, imgRGB, alpha, A_light):

    # imgRGB_save = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)
    # fname_orig_imag = '/home/tmondal/Documents/Dataset/Dehazing/NYU_Depth_V2/test_1/test_orig_imag_large_dataset.png'
    # imgRGB_save = imgRGB_save.astype(np.uint8)
    # cv2.imwrite(fname_orig_imag, imgRGB_save)
    # del imgRGB_save

    # imgD_Norm_save = cv2.normalize(imgD_Norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # imgD_Norm_save = imgD_Norm_save.astype(np.uint8)
    # fname_orig_depth = '/home/tmondal/Documents/Dataset/Dehazing/NYU_Depth_V2/test_1/' \
    #                    'test_orig_depth_large_dataset.png'
    # cv2.imwrite(fname_orig_depth, imgD_Norm_save)
    # del imgD_Norm_save

    imgD_Norm += depht_add  # add a minimum to depth map

    imgD_Norm_T = torch.from_numpy(imgD_Norm)
    imgRGB_T = torch.from_numpy(imgRGB)

    # Compute scattering part
    I_out = scatterPsdOp(imgRGB_T, imgD_Norm_T, gamma, alpha, kern_size, depht_levels)
    I_out = I_out.data.numpy()

    # Compute total model (scattering & loss + attenuation + ambient)
    I_total = outTotal(imgRGB.astype(np.float32), imgD_Norm.astype(np.float32), I_out, gamma, alpha, beta, A_light)

    # particle turbidity effect
    dim = I_total.shape
    out = np.zeros(dim)
    out2 = turbidDeg(imgRGB.astype(np.float32), imgD_Norm.astype(np.float32), turbu_p, turbu_c)
    out = I_total * u + out2 * s * (1 - u)

    # Adjust and save result
    import math
    for c in range(3):
        for i in range(dim[0]):
            for j in range(dim[1]):
                if math.isnan(out[i][j][c]):
                    out[i][j][c] = 255
                else:
                    out[i][j][c] = out[i][j][c]

                if out[i][j][c] > 255:
                    out[i][j][c] = 255
                else:
                    out[i][j][c] = out[i][j][c]

    out = out.astype(np.uint8)

    # Save
    # out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    # fname = '/home/tmondal/Documents/Dataset/Dehazing/NYU_Depth_V2/test_1/test_generated_large_dataset_1.png'
    # out = out.astype(np.uint8)
    # cv2.imwrite(fname, out)

    return out


def getRndPar():
    r = np.random.randint(1, 20, size=1)
    rb = np.random.randint(1, 300, size=1)
    val = (1.0 / 2500000.0) * 16  # r[0]*r[0]*r[0]*r[0]
    val = val * (1.0 + (rb[0] / 1000.0))
    return val


def loadZipToMem(zip_file):
    # Load zip file into memory
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n')
                       if len(row) > 0))

    nyu2_train = shuffle(nyu2_train, random_state=0)

    # if True: nyu2_train = nyu2_train[:40]

    print('Loaded ({0}).'.format(len(nyu2_train)))
    return data, nyu2_train


def to_transform(pic):
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


def main():
    import glob
    pick_indx = 49652
    data, nyu2_train = loadZipToMem('/home/tmondal/Documents/Dataset/Dehazing/nyu_data.zip')
    data, nyu_dataset = data, nyu2_train

    sample = nyu_dataset[pick_indx]

    image = Image.open(BytesIO(data[sample[0]]))
    depth = Image.open(BytesIO(data[sample[1]]))
    image = to_transform(image)
    depth = to_transform(depth)

    image = (image - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255
    depth = (depth - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255

    image_numpy = np.array(image)
    image_numpy = np.swapaxes(image_numpy, 0, 2)
    image_numpy = np.swapaxes(image_numpy, 0, 1)

    depth_numpy = np.array(depth)
    depth_numpy = np.swapaxes(depth_numpy, 0, 2)
    depth_numpy = np.swapaxes(depth_numpy, 0, 1)

    depth_numpy = depth_numpy[:, :, 0]

    imgRGB = image_numpy

    imgD_Norm = depth_numpy
    imgD_Norm = imgD_Norm + 1
    imgD_Norm = (imgD_Norm / np.min([np.max(imgD_Norm), 255])) * 255  # values are > 0 & <= 255

    del image_numpy, depth_numpy

    alpha = [0.012, 0.012, 0.012]
    A_light = [00.0, 120.0, 30.0]

    # Process
    out = processImg(imgD_Norm, imgRGB, alpha, A_light)


if __name__ == '__main__':
    main()

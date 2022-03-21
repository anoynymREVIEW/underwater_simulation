from matplotlib import pyplot as plt
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import glob
import time
from ricardo_code_1 import processImg as ricardo_process_img
# Parameters
# pExp = 5.0/(1*5*5*5*5)

gamma = [0.03, 0.009, 0.043]
# alpha = [0.012, 0.012, 0.012]

beta = [0.020, 0.005, 0.010]

# A_light = [00.0, 120.0, 30.0]
turbu_p = [0.15, 1.5]  # noise amount, est. dev.
turbu_c = [34.0, 200.0, 201.0]

u = 0.98  # weighting between scattering + attenuation image vs particle noise image

s = 5  # multiplier for particle noise image
depht_add = 300  # addition to values of depht map (a minimum distance for map)
depht_levels = 16
kern_size = 10


# Compute scatering diffusion for all pixels (Eq. 2, 3) using depht levels
def scatter_ps_op(rgb, depht, gamma, win_size, d_levels):
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

            # va = va.type(torch.DoubleTensor)

            img_c = F.conv2d(vb, va, padding=win_size)
            img_c = img_c[0, 0, :, :]

            img_sum[:, :, c] += img_c
            img_sum[img_sum > 255] = 255  # for saturated values

    return img_sum


# Compute degradation caused by colored particle turbidity
def turbid_deg(rgb, depht, turbu_p, turbu_c):
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
def out_total(rgb, depht, I_out, alpha, beta, A_light):
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


def process_img(imgD_Norm, imgRGB, alpha, A_light):
    imgD_Norm += depht_add  # add a minimum to depth map

    imgD_Norm_T = torch.from_numpy(imgD_Norm)
    imgRGB_T = torch.from_numpy(imgRGB)

    # Compute scattering part
    I_out = scatter_ps_op(imgRGB_T, imgD_Norm_T, gamma, kern_size, depht_levels)
    I_out = I_out.data.numpy()

    # Compute total model (scattering & loss + attenuation + ambient)
    I_total = out_total(imgRGB.astype(np.float32), imgD_Norm.astype(np.float32), I_out, alpha, beta, A_light)

    # particle turbidity effect
    dim = I_total.shape
    out2 = turbid_deg(imgRGB.astype(np.float32), imgD_Norm.astype(np.float32), turbu_p, turbu_c)
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

    return out


def compute_complex_noise(input_image, input_depth, alpha_mat, A_light):

    input_image = (input_image - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255
    input_depth = (input_depth - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255

    # input_image = np.swapaxes(input_image, 0, 2)
    # input_image = np.swapaxes(input_image, 0, 1)

    imgD_Norm = input_depth
    imgD_Norm = imgD_Norm + 1
    imgD_Norm = (imgD_Norm / np.min([np.max(imgD_Norm), 255])) * 255  # values are > 0 & <= 255

    # alpha_mat = [0.012, 0.012, 0.012]
    # A_light = [00.0, 120.0, 30.0]

    A_light = np.array(A_light)*255
    A_light = A_light.tolist()

    # Process
    # out_noisy_img = process_img(imgD_Norm, input_image, alpha_mat, A_light)
    out_noisy_img = ricardo_process_img(imgD_Norm, input_image, alpha_mat, A_light)

    return out_noisy_img


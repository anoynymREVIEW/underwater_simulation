from matplotlib import pyplot as plt
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import glob
import time
from ricardo_code_1 import processImg as ricardo_process_img


def compute_complex_noise(input_image, input_depth, beta_mat, A_light):

    input_image = (input_image - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255
    input_depth = (input_depth - 0) * (255 - 0) / (1 - 0) + 0  # normalize the image within 0-255

    imgD_Norm = input_depth
    imgD_Norm = imgD_Norm + 1
    imgD_Norm = (imgD_Norm / np.min([np.max(imgD_Norm), 255])) * 255  # values are > 0 & <= 255

    A_light = np.array(A_light)*255
    A_light = A_light.tolist()

    # Process
    out_noisy_img = ricardo_process_img(imgD_Norm, input_image, beta_mat, A_light)

    return out_noisy_img


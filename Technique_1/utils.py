import matplotlib
import matplotlib.cm

import cv2
import numpy as np


def simple_save_images(nn_noisy_image, name):
    nn_noisy_image = nn_noisy_image.cpu()[1, :, :, :]
    nn_noisy_image_numpy = nn_noisy_image.detach().numpy()
    norm_noisy_generated = cv2.normalize(nn_noisy_image_numpy, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_32F)

    norm_noisy_generated = norm_noisy_generated.astype(np.uint8)
    norm_noisy_generated = np.swapaxes(norm_noisy_generated, 0, 2)
    norm_noisy_generated = np.swapaxes(norm_noisy_generated, 0, 1)
    cv2.imwrite(name, norm_noisy_generated)


def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))
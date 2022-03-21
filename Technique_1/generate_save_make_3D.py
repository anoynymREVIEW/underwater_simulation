import time
import argparse
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils  
from torchvision import transforms  
from tensorboardX import SummaryWriter

from data_2 import generate_and_save_atmosphere_light_beta, generate_and_save_haze_image_test, generate_and_save_atmosphere_light_beta_make_3D, generate_and_save_ricardo_image_make_3D

import joblib
from joblib import Parallel, delayed

import sys
import time



def main():
    # generate_and_save_atmosphere_light_beta_make_3D()
    
    # Load data
    my_index_list = [[0, 30], [30, 60], [60, 100], [100, 130], [130, 160], [160, 200], [200, 230],
                    [230, 260], [260, 300], [300, 330], [330, 360], [360, 400]]
    
    print("Python Version : ", sys.version)
    print("Joblib Version : ", joblib.__version__)

    number_of_cpu = joblib.cpu_count()
    print("The number of CPU is :", number_of_cpu)
    
    time_start = time.time()
    # generate_and_save_ricardo_image_make_3D(my_index_list[0][0], my_index_list[0][1])
    
    with Parallel(n_jobs=12) as parallel:
        (parallel([delayed(generate_and_save_ricardo_image_make_3D)(my_index_list[i][0], my_index_list[i][1])
                        for i in range(12)]))

    time_elapsed = (time.time() - time_start)
    print("Total computation time is :", time_elapsed)

if __name__ == '__main__':
    main()
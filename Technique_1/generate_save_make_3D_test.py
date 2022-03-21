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

from data_2 import generate_and_save_atmosphere_light_beta, generate_and_save_haze_image_test, generate_and_save_atmosphere_light_beta_make_3D, generate_and_save_ricardo_image_make_3D, generate_and_save_atmosphere_light_beta_make_3D_Test, generate_and_save_ricardo_image_make_3D_Test

import joblib
from joblib import Parallel, delayed

import sys
import time



def main():
    # generate_and_save_atmosphere_light_beta_make_3D_Test()
    # return
    
    
    # Load data
    my_index_list = [[0, 30], [30, 60], [60, 90], [90, 134]]
    
    print("Python Version : ", sys.version)
    print("Joblib Version : ", joblib.__version__)

    number_of_cpu = joblib.cpu_count()
    print("The number of CPU is :", number_of_cpu)
    
    time_start = time.time()

    # generate_and_save_ricardo_image_make_3D_Test(0, 134)
    # generate_and_save_ricardo_image_make_3D(my_index_list[0][0], my_index_list[0][1])
    
    with Parallel(n_jobs=4) as parallel:
        (parallel([delayed(generate_and_save_ricardo_image_make_3D_Test)(my_index_list[i][0], my_index_list[i][1])
                        for i in range(4)]))

    time_elapsed = (time.time() - time_start)
    print("Total computation time is :", time_elapsed)

if __name__ == '__main__':
    main()
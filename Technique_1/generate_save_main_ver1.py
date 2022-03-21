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

from data_2 import generate_and_save_atmosphere_light_beta, generate_and_save_haze_image

import joblib
from joblib import Parallel, delayed

import sys
import time


def main():
    
    # Load data
    my_index_list = [[47000, 48000], [48000, 49000], [49000, 50000], [50000, 50688]]
    
    print("Python Version : ", sys.version)
    print("Joblib Version : ", joblib.__version__)

    number_of_cpu = joblib.cpu_count()
    print("The number of CPU is :", number_of_cpu)
    
    time_start = time.clock()
    with Parallel(n_jobs=number_of_cpu) as parallel:
        (parallel([delayed(generate_and_save_haze_image)(my_index_list[i][0], my_index_list[i][1])
                        for i in range(4)]))

    time_elapsed = (time.clock() - time_start)
    print("Total computation time is :", time_elapsed)

if __name__ == '__main__':
    main()

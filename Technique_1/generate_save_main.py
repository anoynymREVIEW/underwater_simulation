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
    my_index_list = [[0, 2000], [2000, 4000], [4000, 6000], [6000, 8000], [8000, 10000], [10000, 12000], [12000, 14000],
                    [14000, 16000], [16000, 18000], [18000, 20000], [20000, 22000], [22000, 24000], [24000, 26000],
                    [26000, 28000], [28000, 30000], [30000, 32000], [32000, 34000], [34000, 36000], [36000, 38000],
                    [38000, 40000], [40000, 42000], [42000, 44000], [44000, 46000], [46000, 48000], [48000, 50688], [50000, 50688]]
    
    print("Python Version : ", sys.version)
    print("Joblib Version : ", joblib.__version__)

    number_of_cpu = joblib.cpu_count()
    print("The number of CPU is :", number_of_cpu)
    
    time_start = time.time()
    generate_and_save_haze_image(my_index_list[0][0], my_index_list[0][1])

    # with Parallel(n_jobs=20) as parallel:
    #     (parallel([delayed(generate_and_save_haze_image)(my_index_list[i][0], my_index_list[i][1])
    #                     for i in range(26)]))

    time_elapsed = (time.time() - time_start)
    print("Total computation time is :", time_elapsed)

if __name__ == '__main__':
    main()
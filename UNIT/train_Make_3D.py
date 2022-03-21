"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch

import torch.nn as nn
import torch.nn.utils as utils

from utils_custom import AverageMeter

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='/homes/t20monda/UNIT/configs/unit_edges2handbags_folder.yaml',
                    help='Path to the config file.')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--output_path', type=str, default='/homes/t20monda/UNIT/', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_use_cuda else "cpu")

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     trainer = nn.DataParallel(trainer.cuda())

# Create and save  data
# train_loader = getTrainingTestingData(batch_size=opts.batchSize)

train_loader = torch.load('/sanssauvegarde/homes/t20monda/DenseDepth_3/train_loader_make_3D.pkl')

N = len(train_loader)
need_train = round((N * 96) / 100)
cnt_batch = 0

train_display_images_a = torch.stack([train_loader.dataset[i]['image_half'] for i in range(display_size)]).cuda()
train_display_images_b = torch.stack([train_loader.dataset[i]['complex_noise_img'].type(torch.FloatTensor)  for i in range(display_size)]).cuda()

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
opts.resume = False
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
print("The check point directory is : ", checkpoint_directory)
print("The resume is : ", opts.resume)


while True:

    for i, sample_batched in enumerate(train_loader):

        trainer.update_learning_rate()
        # Prepare sample and target
        # image = torch.autograd.Variable(sample_batched['image'].to(device))  # full size
        input_A = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size ; image_half

        input_B = torch.autograd.Variable(
            sample_batched['complex_noise_img'].to(device))  # half size ; complex_image_tensor

        input_B = input_B.type(torch.cuda.FloatTensor) # converting into float tensor
        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(input_A, input_B, config)
            trainer.gen_update(input_A, input_B, config)
            torch.cuda.synchronize()

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')
            
        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save_make_3D(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

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

import torch.nn as nn
import torch.nn.functional as F

from data_3 import getTrainingTestingData
from utils_custom import AverageMeter

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

def main():

    # Create and save  data
    batch_size = 10
    
    # Load data
    test_loader = torch.load('/sanssauvegarde/homes/t20monda/UNIT/train_loader.pkl')
    # LogProgress(test_loader)
    ClacAccuracyOnly(test_loader)

def ClacAccuracyOnly(test_loader):
    
    N = len(test_loader)
    
    need_train = round((N*96)/100)
    cnt_batch = 0
    # valid_batch_cnt = 0

    a1_acc = 0.0
    cnt_1 = 0

    a2_acc = 0.0
    cnt_2 = 0

    a3_acc = 0.0
    cnt_3 = 0

    abs_rel_acc = 0.0
    cnt_4 = 0

    rmse_acc = 0.0
    cnt_5 = 0

    log_10_acc = 0.0
    cnt_6 = 0

    for i in range(need_train+1, N):
        # if cnt_batch > need_train : # to skip the last batch from training  

        saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/MUNIT_Resources/' + 'Batch_%d' % i + '_Complex_Imag_GT' + '.pt'
        saving_path_complex_imag_Pred =  '/sanssauvegarde/homes/t20monda/MUNIT_Resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred' + '.pt'

        complex_image_tensor = torch.load(saving_path_complex_imag_GT)
        pred_complex_image = torch.load(saving_path_complex_imag_Pred)

        abs_rel, rmse, log_10, a1, a2, a3 = add_results_1(complex_image_tensor, pred_complex_image, border_crop_size=16)

        if (torch.isfinite(a1)):
            a1_acc = a1_acc + a1.detach().to("cpu").numpy()
            cnt_1 = cnt_1 + 1
            
        if (torch.isfinite(a2)):
            a2_acc = a2_acc + a2.detach().to("cpu").numpy()
            cnt_2 = cnt_2 + 1

        if (torch.isfinite(a3)):
            a3_acc = a3_acc + a3.detach().to("cpu").numpy()
            cnt_3 = cnt_3 + 1

        if (torch.isfinite(abs_rel)):    
            abs_rel_acc = abs_rel_acc + abs_rel.detach().to("cpu").numpy()
            cnt_4 = cnt_4 + 1

        if (torch.isfinite(rmse)):    
            rmse_acc = rmse_acc + rmse.detach().to("cpu").numpy()
            cnt_5 = cnt_5 + 1

        if (torch.isfinite(log_10)):    
            log_10_acc = log_10_acc + log_10.detach().to("cpu").numpy()
            cnt_6 = cnt_6 + 1

    
    a1_acc = a1_acc / cnt_1 
    a2_acc = a2_acc / cnt_2  
    a3_acc = a3_acc / cnt_3 

    abs_rel_acc = abs_rel_acc / cnt_4
    rmse_acc = rmse_acc / cnt_5 
    log_10_acc = log_10_acc / cnt_6 

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(a1_acc, a2_acc, a3_acc, abs_rel_acc, rmse_acc, log_10_acc ))



def LogProgress(test_loader):

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/homes/t20monda/MUNIT/configs/edges2handbags_folder.yaml',
                        help='Path to the config file.')
    parser.add_argument('--seed', type=int, default=10, help="random seed")
    parser.add_argument('--checkpoint', type=str, default='/homes/t20monda/MUNIT/outputs/edges2handbags_folder/checkpoints/optimizer.pt', help="checkpoint of autoencoders")
    parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
    parser.add_argument('--output_folder', type=str, help="output image path")
    parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
    parser.add_argument('--output_path', type=str, default='/homes/t20monda/MUNIT/', help="outputs path")
    parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
    opts = parser.parse_args()

    cudnn.benchmark = True

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # Load experiment setting
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)

    config = get_config(opts.config)
    input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']
    display_size = config['display_size']

    if opts.trainer == 'MUNIT':
        trainer = MUNIT_Trainer(config)
    elif opts.trainer == 'UNIT':
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")
        
    N = len(test_loader)
    need_train = round((N * 96) / 100)
    cnt_batch = 0

    # train_display_images_a = torch.stack([test_loader.dataset[i]['image_half'] for i in range(display_size)]).cuda()
    # train_display_images_b = torch.stack([test_loader.dataset[i]['complex_noise_img'].type(torch.FloatTensor)  for i in range(display_size)]).cuda()

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

    # Start training
    print("The check point directory is : ", checkpoint_directory)
    trainer.resume_test_only(checkpoint_directory, hyperparameters=config)

    trainer.cuda()
    trainer.eval()

    if opts.trainer == 'MUNIT':
        cnt_batch = 0

        # Start testing
        # for i, (images, names) in enumerate(zip(data_loader, image_names)):
        for i, sample_batched in enumerate(test_loader):

            if cnt_batch > need_train:  # to skip the last batch from training
                
                # Prepare sample and target
                input_A = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size ; image_half

                input_B = torch.autograd.Variable(
                    sample_batched['complex_noise_img'].to(device))  # half size ; complex_image_tensor
                input_B = input_B.type(torch.cuda.FloatTensor) # converting into float tensor

                with Timer("Elapsed time in update: %f"):
                    # Main training code
                    trainer.dis_update_test(input_A, input_B, config)
                    trainer.gen_update_test(input_A, input_B, config)
                    torch.cuda.synchronize()

                with torch.no_grad():
                    # train_image_outputs_1 = trainer.sample(train_display_images_a, train_display_images_b)
                    train_image_outputs_2 = trainer.sample(input_A, input_B)
                # write_2images(train_image_outputs_2, display_size, image_directory, 'train_%08d' % (i + 1))
                                
                saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/MUNIT_Resources/' + 'Batch_%d' % i + '_Complex_Imag_GT' + '.pt'
                saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/MUNIT_Resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred' + '.pt'
                torch.save(input_B, saving_path_complex_imag_GT)
                torch.save(train_image_outputs_2[2], saving_path_complex_imag_Pred)
        
            cnt_batch = cnt_batch+1    

    else:
        pass


def compute_errors_nyu(pred, gt):
    # x = pred[crop]
    # y = gt[crop]
    y = gt
    x = pred
    thresh = torch.max((y / x), (x / y))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    abs_rel = torch.mean(torch.abs(y - x) / y)
    rmse = (y - x) ** 2
    rmse = torch.sqrt(rmse.mean())
    log_10 = (torch.abs(torch.log10(y) - torch.log10(x))).nanmean()
    return abs_rel, rmse, log_10, a1, a2, a3


def add_results_1(gt_image, pred_image, border_crop_size=16, use_224=False):
    
    predictions = []
    testSetDepths = []
    half_border_size = border_crop_size // 2

    gt_image_border_cut = gt_image[:, :, half_border_size:-half_border_size, half_border_size:-half_border_size] # cutting the border to remove the border problem/issue
    pred_image_border_cut = pred_image[:, :, half_border_size:-half_border_size, half_border_size:-half_border_size] # cutting the border to remove the border problem/issue
    
    del gt_image, pred_image

    replicate = nn.ReplicationPad2d(half_border_size)
    gt_image_border_cut = replicate(gt_image_border_cut)  # now extrapolate by using the inside content of the image 
    pred_image_border_cut = replicate(pred_image_border_cut)  # now extrapolate by using the inside content of the image

    gt_image_border_cut = F.interpolate(gt_image_border_cut, (480, 640), mode='bilinear', align_corners=True)
    pred_image_border_cut = F.interpolate(pred_image_border_cut, (480, 640), mode='bilinear', align_corners=True)
          
            
    # Compute errors per image in batch
    for j in range(len(gt_image_border_cut)):
        predictions.append(  pred_image_border_cut[j]   )
        testSetDepths.append(   gt_image_border_cut[j]   )

    predictions = torch.stack(predictions, axis=0)
    testSetDepths = torch.stack(testSetDepths, axis=0)

    del pred_image_border_cut, gt_image_border_cut
    abs_rel, rmse, log_10, a1, a2, a3  = compute_errors_nyu(predictions, testSetDepths)

    del predictions, testSetDepths

    return abs_rel, rmse, log_10, a1, a2, a3

if __name__ == '__main__':
    main()


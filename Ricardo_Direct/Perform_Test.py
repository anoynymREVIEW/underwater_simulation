import time
import argparse
import datetime
import os
import numpy as np
import torch
import math 
import sys

import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils  
from torchvision import transforms  
from tensorboardX import SummaryWriter

from model_3D import PTModel as Model3D
from loss import ssim
from data_3 import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize, simple_save_images


# This code is very same as "perform_test.py" but the only difference is here I am adding only the code to 
# compute accuracy metric

def main():

    # Create and save  data
    batch_size = 10
    
    # Load data
    test_loader = torch.load('/home/tamondal/Kepp_All/Ricardo_Direct/train_loader.pkl')
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

        saving_path_complex_imag_GT = '/home/tamondal/Kepp_All/Resources/Ricardo_Direct/' + 'Batch_%d' % i + '_Complex_Imag_GT' + '.pt'
        saving_path_complex_imag_Pred = '/home/tamondal/Kepp_All/Resources/Ricardo_Direct/' + 'Batch_%d' % i + '_Complex_Imag_Pred' + '.pt'

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
    writer = SummaryWriter('/home/tamondal/Kepp_All/Ricardo_Direct/runs/real_densedepth_running')
    model_name = "densenet_multi_task"
    epoch = 0
    checkpoint = torch.load('/home/tamondal/Kepp_All/Ricardo_Direct/checkpoint/' + model_name + '/Models' + '.ckpt')
    
    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    N = len(test_loader)

    # Create model
    model_3 = Model3D().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_3 = nn.DataParallel(model_3.cuda())

    model_3.load_state_dict(checkpoint['state_dict_3'])

    model_3.eval()

    # Loss
    l1_criterion = nn.L1Loss()
    losses = AverageMeter()
    # Here we are trying to calculate the loss for Test dataset 
    N_Test = len(test_loader)
    print('The number of images in test loader {}'.format(N_Test))
    print('We are testing for the training epoch {}'.format(epoch))
    print('-' * 10)
    
    need_train = round((N*96)/100)
    cnt_batch = 0
    valid_batch_cnt = 0

    for i, sample_batched in enumerate(test_loader):
        if cnt_batch > need_train : # to skip the last batch from training  

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].to(device))  # full size
            complex_image_tensor = torch.autograd.Variable(sample_batched['complex_noise_img'].to(device))  # half size

            # Predict
            output_ricardo_image_direct = model_3(image)
          
            # Compute the loss for complex haze image which is calculated directly from the original image
            loss_ricardo_image_direct = l1_criterion(output_ricardo_image_direct, complex_image_tensor)
            loss_ssim_ricardo_image_direct = torch.clamp((1 - ssim(output_ricardo_image_direct.float(), complex_image_tensor.float(),
                                                            val_range=1)) * 0.5, 0, 1)
            loss_ricardo_image_total = (1.0 * loss_ricardo_image_direct) + (0.1 * loss_ssim_ricardo_image_direct)
            # del complex_image_tensor, output_ricardo_image_direct

            saving_path_complex_imag_GT = '/home/tamondal/Kepp_All/Resources/Ricardo_Direct/' + 'Batch_%d' % i + '_Complex_Imag_GT' + '.pt'
            saving_path_complex_imag_Pred = '/home/tamondal/Kepp_All/Resources/Ricardo_Direct/' + 'Batch_%d' % i + '_Complex_Imag_Pred' + '.pt'
            torch.save(complex_image_tensor, saving_path_complex_imag_GT)
            torch.save(output_ricardo_image_direct, saving_path_complex_imag_Pred)

            # Update step
            total_batch_loss = loss_ricardo_image_total
            losses.update(total_batch_loss.data.item(), image.size(0))

            # Log progress
            if i % 5 == 0:
                # Log to tensorboard in each 5th batch after
                writer.add_scalar('Test/Loss', total_batch_loss.item(), i) # plotting the total 

            # Now print the data/images for the last batch only      
            if i == N_Test-1:

                writer.add_image('Train.1.Image_Half', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
                writer.add_image('Train.2.GT_Complex_Image', vutils.make_grid(complex_image_tensor.data, nrow=6, normalize=False),
                                epoch)

                # save the predicted images
                writer.add_image('Train.3.Pred_Complex_Image', (vutils.make_grid(output_ricardo_image_direct.data, nrow=6,
                                                                            normalize=False)), epoch)             


            valid_batch_cnt = valid_batch_cnt + 1

        cnt_batch = cnt_batch+1

    print('The average testing loss is: {:.4f} '.format(losses.avg))
        

def compute_complex_image(output_depth, output_black_box, beta_val, a_mat, unit_mat, image_half):

    output_depth_3d = torch.tile(output_depth, [1, 3, 1, 1])
    output_black_box_3d = output_black_box # torch.tile(output_black_box, [1, 3, 1, 1])

    tx1 = torch.exp(-torch.mul(beta_val, output_depth_3d))
    second_term = torch.mul(a_mat, (torch.subtract(unit_mat, tx1)))
    haze_image = torch.add((torch.mul(image_half, tx1)), second_term)

    pred_complex_image = output_black_box_3d + haze_image
    return pred_complex_image


def compute_haze_image(output_depth, beta_val, a_mat, unit_mat, image_half):
    
    output_depth_3d = torch.tile(output_depth, [1, 3, 1, 1])

    tx1 = torch.exp(-torch.mul(beta_val, output_depth_3d))
    second_term = torch.mul(a_mat, (torch.subtract(unit_mat, tx1)))
    haze_image = torch.add((torch.mul(image_half, tx1)), second_term)
    return haze_image


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

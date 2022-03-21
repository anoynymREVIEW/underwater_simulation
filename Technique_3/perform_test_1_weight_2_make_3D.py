import time
import argparse
import datetime
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils  
from torchvision import transforms  
from tensorboardX import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F

from model_weight_2_make_3D import PTModel as Model
from model_3D_weight_make_3D import PTModel as Model3D

from loss import ssim
from data_4_Make_3D_Test import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize, simple_save_images


# This code is very same as "perform_test.py" but the only difference is here I am adding only the code to 
# compute accuracy metric

def main():

    # Create and save  data
    batch_size = 10
    
    # Load data
    test_loader = getTrainingTestingData(batch_size=batch_size, transforms=None)
    # LogProgress(test_loader)
    ClacAccuracyOnly(test_loader)

def ClacAccuracyOnly(test_loader):
    
    N = len(test_loader)
    
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

    for i in range(0, N):
        # if cnt_batch > need_train : # to skip the last batch from training  

        saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/DenseDepth_Resources/DenseDepth_3/Weight_2/' + 'Batch_%d' % i + '_Complex_Imag_GT' + '.pt'
        saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/DenseDepth_Resources/DenseDepth_3/Weight_2/' + 'Batch_%d' % i + '_Complex_Imag_Pred' + '.pt'

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

        # valid_batch_cnt = valid_batch_cnt + 1

        # else :
            # break

    # cnt_batch = cnt_batch+1
    
    a1_acc = a1_acc / cnt_1 
    a2_acc = a2_acc / cnt_2  
    a3_acc = a3_acc / cnt_3 

    abs_rel_acc = abs_rel_acc / cnt_4
    rmse_acc = rmse_acc / cnt_5 
    log_10_acc = log_10_acc / cnt_6 

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(a1_acc, a2_acc, a3_acc, abs_rel_acc, rmse_acc, log_10_acc ))


def LogProgress(test_loader):
    writer = SummaryWriter('/homes/t20monda/DenseDepth_3/runs/Make_3D/real_DenseDepth_3_Make_3D_Weight_2_running')
    model_name = "densenet_multi_task"
    epoch = 0
    checkpoint = torch.load('/sanssauvegarde/homes/t20monda/DenseDepth_3/checkpoint/' + model_name + '/Models_Weight_2_Make_3D' + '.ckpt')
    
    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    N = len(test_loader)

    # Create model
    model_1 = Model().to(device)
    model_2 = Model3D().to(device)
    model_3 = Model3D().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_1 = nn.DataParallel(model_1.cuda())
        model_2 = nn.DataParallel(model_2.cuda())
        model_3 = nn.DataParallel(model_3.cuda())

    model_1.load_state_dict(checkpoint['state_dict_1'])
    model_2.load_state_dict(checkpoint['state_dict_2'])
    model_3.load_state_dict(checkpoint['state_dict_3'])

    model_1.eval()
    model_2.eval()
    model_3.eval()

    # Loss
    l1_criterion = nn.L1Loss()
    losses = AverageMeter()
    # Here we are trying to calculate the loss for Test dataset 
    N_Test = len(test_loader)
    print('The number of images in test loader {}'.format(N_Test))
    print('We are testing for the training epoch {}'.format(epoch))
    print('-' * 10)

    valid_batch_cnt = 0

    for i, sample_batched in enumerate(test_loader):

        # Prepare sample and target
        image_full = torch.autograd.Variable(sample_batched['image_full'].to(device))  # full size
        image_half = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size
        depth_half = torch.autograd.Variable(sample_batched['depth_half'].to(device))  # half size

        orig_haze_image = torch.autograd.Variable(sample_batched['haze_image'].to(device))  # half size

        beta_val_half = torch.autograd.Variable(sample_batched['beta'].to(device))  # half size
        a_mat_half = torch.autograd.Variable(sample_batched['a_val'].to(device))  # half size
        unit_mat_half = torch.autograd.Variable(sample_batched['unit_mat'].to(device))  # half size
        complex_image_tensor_half = torch.autograd.Variable(
            sample_batched['complex_noise_img'].to(device))  # half size

        # Predict
        output_depth, weight_feature_depth, weight_features_3_output = model_1(image_full)
        output_black_box, weight_feature_black_box = model_2(image_full)
        output_ricardo_image_direct, weight_feature_ricardo_direct = model_3(image_full)


        pred_complex_image = compute_complex_image(output_depth, output_black_box, beta_val_half, a_mat_half,
                                                       unit_mat_half, image_half)

        pred_haze_image = compute_haze_image(output_depth, beta_val_half, a_mat_half, unit_mat_half,
                                                    image_half)   
        
        weight_depth_mean = torch.mean(weight_feature_depth, dim=0)
        weight_black_box_mean = torch.mean(weight_feature_black_box, dim=0)
        weight_feature_ricardo_direct_mean = torch.mean(weight_feature_ricardo_direct, dim=0)   
        weight_features_3_output_mean = torch.mean(weight_features_3_output, dim=0)
  
        # Compute the loss for depth
        l_depth = l1_criterion(output_depth, depth_half)
        l_ssim = torch.clamp((1 - ssim(output_depth, depth_half, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
        loss_depth = ((1.0-weight_depth_mean[0]) * l_ssim) + (weight_depth_mean[0] * l_depth)

        # Compute the loss for haze image
        loss_haze_image = l1_criterion(pred_haze_image, orig_haze_image)
        loss_ssim_haze_image = torch.clamp((1 - ssim(pred_haze_image.float(), orig_haze_image.float(),
                                                        val_range=1)) * 0.5, 0, 1)
        loss_haze_total = ((1.0 - weight_black_box_mean[1]) * loss_ssim_haze_image) + (
                weight_black_box_mean[1] * loss_haze_image)

        # Compute the loss for complex haze image
        loss_complex_image = l1_criterion(pred_complex_image, complex_image_tensor_half)
        loss_ssim_complex_image = torch.clamp(
            (1 - ssim(pred_complex_image.float(), complex_image_tensor_half.float(),
                        val_range=1)) * 0.5, 0, 1)
        loss_complex_total = ((1.0 - weight_black_box_mean[0]) * loss_ssim_complex_image) + (
                weight_black_box_mean[0] * loss_complex_image)

        # Compute the loss for complex haze image which is calculated directly from the original image
        loss_ricardo_image_direct = l1_criterion(output_ricardo_image_direct, complex_image_tensor_half)
        loss_ssim_ricardo_image_direct = torch.clamp((1 - ssim(output_ricardo_image_direct.float(), complex_image_tensor_half.float(),
                                                        val_range=1)) * 0.5, 0, 1)
        loss_ricardo_image_total = ((1.0-weight_feature_ricardo_direct_mean[0]) * loss_ricardo_image_direct) + (weight_feature_ricardo_direct_mean[0] * loss_ssim_ricardo_image_direct)
        
        # Update step
        total_batch_loss = (weight_features_3_output_mean[0] * loss_complex_total) + (
                    weight_features_3_output_mean[1] * loss_depth) + (
                    weight_features_3_output_mean[2] * loss_haze_total) + ( (1 - ( weight_features_3_output_mean[0] + 
                                                                            weight_features_3_output_mean[1] + 
                                                                            weight_features_3_output_mean[2] ) ) * 
                                                                        loss_ricardo_image_total)

        saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/DenseDepth_Resources/DenseDepth_3/Weight_2/' + 'Batch_%d' % i + '_Complex_Imag_GT' + '.pt'
        saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/DenseDepth_Resources/DenseDepth_3/Weight_2/' + 'Batch_%d' % i + '_Complex_Imag_Pred' + '.pt'
        torch.save(complex_image_tensor_half, saving_path_complex_imag_GT)
        torch.save(pred_complex_image, saving_path_complex_imag_Pred)

        # Update step
        total_batch_loss = loss_complex_total + loss_depth + loss_haze_total + loss_ricardo_image_total
        losses.update(total_batch_loss.data.item(), image_full.size(0))

        # Log progress
        if i % 5 == 0:
            # Log to tensorboard in each 5th batch after
            writer.add_scalar('Test/Loss', total_batch_loss.item(), i) # plotting the total 

        # Now print the data/images for the last batch only      
        if i == N_Test-1:

            writer.add_image('Train.1.Image_Half', vutils.make_grid(image_half.data, nrow=6, normalize=True), epoch)
            writer.add_image('Train.2.Depth_Norm_True', (vutils.make_grid(depth_half.data, nrow=6, normalize=True)), epoch)
            writer.add_image('Train.2.Depth_Norm_False', (vutils.make_grid(depth_half.data, nrow=6, normalize=False)), epoch)
            writer.add_image('Train.3.GT_Haze_Image', vutils.make_grid(orig_haze_image.data, nrow=6, normalize=False),
                            epoch)
            writer.add_image('Train.4.GT_Complex_Image', vutils.make_grid(complex_image_tensor_half.data, nrow=6, normalize=False),
                            epoch)

            # save the predicted images
            writer.add_image('Train.5.Ours_Depth_Norm_True', (vutils.make_grid(output_depth.data, nrow=6,
                                                                            normalize=True)), epoch)
            writer.add_image('Train.5.Ours_Depth_Norm_False', (vutils.make_grid(output_depth.data, nrow=6,
                                                                            normalize=False)), epoch)                                                                
            writer.add_image('Train.6.Ours_Blackbox_Auto_Norm', (vutils.make_grid(output_black_box.data, nrow=6, normalize=True)), epoch)
            writer.add_image('Train.7.Pred_Complex_Image', (vutils.make_grid(pred_complex_image.data, nrow=6,
                                                                        normalize=False)), epoch)  
            writer.add_image('Train.8.Pred_Complex_Image_Direct', (vutils.make_grid(output_ricardo_image_direct.data, nrow=6,
                                                                        normalize=False)), epoch) 

        valid_batch_cnt = valid_batch_cnt + 1

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

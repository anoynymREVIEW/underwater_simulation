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

from model import PTModel as Model
from model_3D import PTModel as Model3D
from loss import ssim
from data_4 import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize, simple_save_images


# This code is very same as "train_2.py" but here I am using the saved "atmospheric_light.npy" and "beta.npy" files
# we are also using here the "index_haze_image.npz", "index_complex_haze_image.npz" and "index_complex_depth_half_3d.npz" files, during training 


def main():

    # Create and save  data
    batch_size = 10
    
    # Load data
    test_loader = torch.load('/sanssauvegarde/homes/t20monda/Dense_Depth_1/train_loader.pkl')
    LogProgress(test_loader)

def LogProgress(test_loader):
    writer = SummaryWriter('/homes/t20monda/DenseDepth_1/runs/real_densedepth_running')
    model_name = "densenet_multi_task"
    epoch = 0
    checkpoint = torch.load('/sanssauvegarde/homes/t20monda/Dense_Depth_1/checkpoint/' + model_name + '/Models' + '.ckpt')
    
    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    N = len(test_loader)

    # Create model
    model_1 = Model().to(device)
    model_2 = Model3D().to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_1 = nn.DataParallel(model_1.cuda())
        model_2 = nn.DataParallel(model_2.cuda())

    model_1.load_state_dict(checkpoint['state_dict_1'])
    model_2.load_state_dict(checkpoint['state_dict_2'])

    model_1.eval()
    model_2.eval()

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
            
    for i, sample_batched in enumerate(test_loader):
        if cnt_batch > need_train : # to skip the last batch from training  
            # sequential = test_loader
            # sample_batched = next(iter(sequential))
                
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].to(device))  # full size
            image_half = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size
            
            depth = torch.autograd.Variable(sample_batched['depth'].to(device))  # half size

            # depth_half = torch.autograd.Variable(sample_batched['depth_norm_simple'].to(device))  # half size
            orig_haze_image = torch.autograd.Variable(sample_batched['haze_image'].to(device))  # half size

            beta_val = torch.autograd.Variable(sample_batched['beta'].to(device))  # half size
            a_mat = torch.autograd.Variable(sample_batched['a_val'].to(device))  # half size
            unit_mat = torch.autograd.Variable(sample_batched['unit_mat'].to(device))  # half size
            complex_image_tensor = torch.autograd.Variable(sample_batched['complex_noise_img'].to(device))  # half size

            # Normalize depth
            depth_n = DepthNorm(depth)  # I think by this normalization, they will bring back the values within 0-1
            # so that they can calculate the loss w.r.t the estimated depth, which is within 0-1

            # Predict
            output_depth = model_1(image)
            output_black_box = model_2(image)

            output_depth_norm = DepthNorm(output_depth)  # performing strange normalization
            # output_black_box_norm = DepthNorm(output_black_box)  # performing strange normalization
            
            pred_haze_image = compute_haze_image(output_depth, beta_val, a_mat, unit_mat,
                                                        image_half)  
            
            pred_complex_image = compute_complex_image(output_depth, output_black_box, beta_val, a_mat, unit_mat, image_half)
            
            pred_manual_subtract = torch.abs(torch.subtract(pred_complex_image, pred_haze_image))
            
            # Compute the loss
            l_depth = l1_criterion(output_depth, depth_n)
            l_ssim = torch.clamp((1 - ssim(output_depth, depth, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
            loss_depth = (1.0 * l_ssim) + (0.1 * l_depth)

            loss_complex_image = l1_criterion(pred_complex_image, complex_image_tensor)
            loss_ssim_complex_image = torch.clamp((1 - ssim(pred_complex_image.float(), complex_image_tensor.float(),
                                                            val_range=1)) * 0.5, 0, 1)
            loss_complex_total = (1.0 * loss_ssim_complex_image) + (0.1 * loss_complex_image)

            # Update step
            total_batch_loss = loss_complex_total + loss_depth
            losses.update(total_batch_loss.data.item(), image.size(0))
            # Log progress
            if i % 5 == 0:
                # Log to tensorboard in each 5th batch after
                writer.add_scalar('Test/Loss', total_batch_loss.item(), i) # plotting the total 

            # Now print the data/images for the last batch only    
            if i == N_Test-1:

                writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
                writer.add_image('Train.2.Image_Half', vutils.make_grid(image_half.data, nrow=6, normalize=True), epoch)
                writer.add_image('Train.3.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)

                writer.add_image('Train.4.GT_Haze_Image', vutils.make_grid(orig_haze_image.data, nrow=6, normalize=False),
                                epoch)
                writer.add_image('Train.5.GT_Complex_Image', vutils.make_grid(complex_image_tensor.data, nrow=6, normalize=False),
                                epoch)

                # save the predicted images
                writer.add_image('Train.6.Ours_Depth_Norm', colorize(vutils.make_grid(output_depth_norm.data, nrow=6,
                                                                                normalize=False)), epoch)
                
                writer.add_image('Train.7.Ours_Haze_Image_No_Norm', (vutils.make_grid(pred_haze_image.data, nrow=6,
                                                                                normalize=False)), epoch)

                writer.add_image('Train.8.Ours_Blackbox_Auto_Norm', (vutils.make_grid(output_black_box.data, nrow=6, normalize=True)), epoch)
                
                writer.add_image('Train.9.Ours_Manual_Subtract_Auto_Norm', (vutils.make_grid(pred_manual_subtract.data, nrow=6, normalize=True)), epoch) 
                writer.add_image('Train.9.Ours_Manual_Subtract_No_Norm', (vutils.make_grid(pred_manual_subtract.data, nrow=6, normalize=False)), epoch) 

                writer.add_image('Train.9.Pred_Complex_Image', (vutils.make_grid(pred_complex_image.data, nrow=6,
                                                                            normalize=False)), epoch)             

            del image, image_half, orig_haze_image
            del depth
            del beta_val, a_mat, unit_mat, complex_image_tensor
        
        # else :
            # break
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
    log_10 = (torch.abs(torch.log10(y) - torch.log10(x))).mean()
    return abs_rel, rmse, log_10, a1, a2, a3

def _save_best_model(model_1, model_2, best_loss, epoch):
    # Save Model
    model_name = "densenet_multi_task"
    state = {
        'state_dict_1': model_1.state_dict(),
        'state_dict_2': model_2.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }

    if not os.path.isdir('/homes/t20monda/DenseDepth_1/checkpoint/' + model_name):
            os.makedirs('/homes/t20monda/DenseDepth_1/checkpoint/' + model_name)

    torch.save(state,
               '/homes/t20monda/DenseDepth_1/checkpoint/' +
               model_name + '/Models' + '.ckpt')  


if __name__ == '__main__':
    main()

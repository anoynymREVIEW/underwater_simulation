import argparse
import os

from PIL import Image
import torch
import torch.nn as nn

from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger

import time
import datetime

import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils
from torch.autograd import Variable

from models import *
from tensorboardX import SummaryWriter
from data_4_Make_3D_Test import getTrainingTestingData
from utils_custom import AverageMeter
from torchvision.utils import save_image

import numpy as np


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

        saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/Pix2Pix_Resources/' + 'Batch_%d' % i + '_Complex_Imag_GT_make3D' + '.pt'
        saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/Pix2Pix_Resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_make3D' + '.pt'
        saving_path_complex_imag_A_Pred = '/sanssauvegarde/homes/t20monda/Pix2Pix_Resources/' + 'Batch_%d' % i + '_Complex_Imag_A_Pred_make3D' + '.pt'

        complex_image_tensor = torch.load(saving_path_complex_imag_GT)
        pred_complex_image = torch.load(saving_path_complex_imag_Pred)
        pred_complex_A_image = torch.load(saving_path_complex_imag_A_Pred)

        # just save one set of images 
        # if i == N-5 :
        #     save_image(complex_image_tensor, 'GT_Complex_Image.png')
        #     save_image(pred_complex_image, 'Pred_Complex_Image.png')
        #     save_image(pred_complex_A_image, 'Pred_Complex_A_Image.png')

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
    writer = SummaryWriter('/homes/t20monda/Pix2Pix_GAN/runs/real_densedepth_running_test_make3D')
    model_name = "cycle_gan_task"
    epoch = 0
    
    # Initialize generator and discriminator
    generator = GeneratorUNet_Make3D()
    discriminator = Discriminator()
    
    # -------------------------- Losses --------------------------------------------------
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()
    
    
    checkpoint_G_A2B = torch.load(
            '/homes/t20monda/Pix2Pix_GAN/checkpoint/' + model_name + '/Models_Make3D' + '_%s' % 'netG_A2B' + '.ckpt')
    checkpoint_G_B2A = torch.load(
            '/homes/t20monda/Pix2Pix_GAN/checkpoint/' + model_name + '/Models_Make3D' + '_%s' % 'netD' + '.ckpt')

    
    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    N = len(test_loader)

    # Create model
    generator.to(device)
    discriminator.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        nn.DataParallel(generator.cuda())
        nn.DataParallel(discriminator.cuda())

    generator.load_state_dict(checkpoint_G_A2B['state_dict'])
    discriminator.load_state_dict(checkpoint_G_B2A['state_dict'])

    generator.eval()
    discriminator.eval()

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, 173 // 2 ** 4, 230 // 2 ** 4)
    # Tensor type
    Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100
    
    # Loss
    losses_generator = AverageMeter()
    losses_discriminator = AverageMeter()
    
    # Here we are trying to calculate the loss for Test dataset 
    N_Test = len(test_loader)
    print('The number of images in test loader {}'.format(N_Test))
    print('We are testing for the training epoch {}'.format(epoch))
    print('-' * 10)
    
    valid_batch_cnt = 0

    for i, sample_batched in enumerate(test_loader): 

        # Prepare sample and target
        # image = torch.autograd.Variable(sample_batched['image'].to(device))  # full size
        input_A = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size ; image_half

        input_B = torch.autograd.Variable(
            sample_batched['complex_noise_img'].to(device))  # half size ; complex_image_tensor

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((input_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((input_A.size(0), *patch))), requires_grad=False)

        # GAN loss
        fake_B = generator(input_A)
        fake_B = F.interpolate(fake_B, size=(173, 230), mode='bicubic', align_corners=False)

        pred_fake = discriminator(fake_B, input_A)
        loss_GAN = criterion_GAN(pred_fake, valid)

        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(fake_B, input_B)

        # Total Generator loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel
        
        # Real loss
        input_B = input_B.type(torch.cuda.FloatTensor) # converting into double tensor
        pred_real = discriminator(input_B, input_A)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_B.detach(), input_A)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total Discriminator loss
        loss_D = 0.5 * (loss_real + loss_fake)

        saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/Pix2Pix_Resources/' + 'Batch_%d' % i + '_Complex_Imag_GT_make3D' + '.pt'
        saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/Pix2Pix_Resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_make3D' + '.pt'
        saving_path_complex_imag_A_Pred = '/sanssauvegarde/homes/t20monda/Pix2Pix_Resources/' + 'Batch_%d' % i + '_Complex_Imag_A_Pred_make3D' + '.pt'
        torch.save(input_B, saving_path_complex_imag_GT)
        torch.save(fake_B, saving_path_complex_imag_Pred)
        torch.save(fake, saving_path_complex_imag_A_Pred)

        losses_generator.update(loss_G.data.item(), input_A.size(0))
        losses_discriminator.update(loss_D.data.item(), input_A.size(0))
        
        # Log progress
        if i % 5 == 0:
            # Log to tensorboard in each 5th batch after
            writer.add_scalar('Test/Loss Generator', loss_G.item(), i) # plotting the total 
            writer.add_scalar('Test/Loss Discriminator', loss_D.item(), i) # plotting the total 

        # Now print the data/images for the last batch only      
        if i == N_Test-1:
            # Log to tensorboard
            writer.add_image('real_A', vutils.make_grid(input_A.data, nrow=6, normalize=False),
                                epoch)
            writer.add_image('real_B', vutils.make_grid(input_B.data, nrow=6, normalize=False),
                                epoch)
            writer.add_image('fake_A', vutils.make_grid(fake.data, nrow=6, normalize=False),
                                epoch)
            writer.add_image('fake_B', vutils.make_grid(fake_B.data, nrow=6, normalize=False),
                                epoch)         

        valid_batch_cnt = valid_batch_cnt + 1

    
    print('The average testing loss of generator is: {:.4f} '.format(losses_generator.avg))
    print('The average testing loss of dsicriminator is: {:.4f} '.format(losses_discriminator.avg))
        

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

def add_results(gt_image, pred_image, border_crop_size=16):
    
    predictions = []
    testSetDepths = []
    gt_image_border_cut = gt_image[:, :, border_crop_size:-border_crop_size, border_crop_size:-border_crop_size]
    pred_image_border_cut = pred_image[:, :, border_crop_size:-border_crop_size, border_crop_size:-border_crop_size]

    del gt_image, pred_image

    # gt_image = gt_image.detach().cpu()
    # pred_image = pred_image.detach().cpu()

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

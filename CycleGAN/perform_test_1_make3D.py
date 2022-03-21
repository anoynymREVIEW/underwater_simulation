import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os

from PIL import Image
import torch
import torch.nn as nn
import torchvision.utils as vutils 

from tensorboardX import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger

from data_4_Make_3D_Test import getTrainingTestingData

from utils import weights_init_normal
from datasets import ImageDataset
from utils_custom import AverageMeter
import numpy as np
import warnings
warnings.filterwarnings("error")


# This code is very same as "perform_test.py" but the only difference is here I am adding only the code to 
# compute accuracy metric

def main():

    # Create and save  data
    batch_size = 2
    
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

        saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/CycleGAN_Resources/' + 'Batch_%d' % i + '_Complex_Imag_GT_make3D' + '.pt'
        saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/CycleGAN_Resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_make3D' + '.pt'
        # saving_path_complex_imag_Pred_A = '/home/tamondal/Kepp_All/Resources/CycleGAN/' + 'Batch_%d' % i + '_Complex_Imag_Pred_A' + '.pt'

        complex_image_tensor = torch.load(saving_path_complex_imag_GT)
        pred_complex_image = torch.load(saving_path_complex_imag_Pred)
        # torch.save(fake_A, saving_path_complex_imag_Pred_A)

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
    writer_1 = SummaryWriter('/homes/t20monda/CycleGAN/runs/real_cycleGAN_running_test_make3D')
    model_name = "cycle_gan_task"
    epoch = 0
    
    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    
    input_nc = 3
    output_nc = 3
    batchSize = 2

    # Networks
    netG_A2B = Generator(input_nc, output_nc)
    netG_B2A = Generator(output_nc, input_nc)
    netD_A = Discriminator(input_nc)  # discriminate the generated A
    netD_B = Discriminator(output_nc)  # discriminate the generated B
    
    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        nn.DataParallel(netG_A2B.cuda())
        nn.DataParallel(netG_B2A.cuda())
        nn.DataParallel(netD_A.cuda())
        nn.DataParallel(netD_B.cuda())

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # -------------------------- Losses --------------------------------------------------
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    checkpoint_G_A2B = torch.load('/homes/t20monda/CycleGAN/checkpoint/' + model_name + '/Models_Make_3D' + '_%s' % 'netG_A2B' + '.ckpt')
    checkpoint_G_B2A = torch.load('/homes/t20monda/CycleGAN/checkpoint/' + model_name + '/Models_Make_3D' + '_%s' % 'netG_B2A' + '.ckpt')
    checkpoint_D_A = torch.load('/homes/t20monda/CycleGAN/checkpoint/' + model_name + '/Models_Make_3D' + '_%s' % 'netD_A' + '.ckpt')
    checkpoint_D_B = torch.load('/homes/t20monda/CycleGAN/checkpoint/' + model_name + '/Models_Make_3D' + '_%s' % 'netD_B' + '.ckpt')

    netG_A2B.load_state_dict(checkpoint_G_A2B['state_dict'])
    netG_B2A.load_state_dict(checkpoint_G_B2A['state_dict'])
    netD_A.load_state_dict(checkpoint_D_A['state_dict'])
    netD_B.load_state_dict(checkpoint_D_B['state_dict'])

    N = len(test_loader)
    
    netG_A2B.train()
    netG_B2A.train()
    netD_A.train()
    netD_B.train()

    # Loss
    losses_G = AverageMeter()
    losses_G_Identity = AverageMeter()
    losses_G_GAN = AverageMeter()
    losses_G_Cycle = AverageMeter()
    losses_G_D = AverageMeter()
    
    # Here we are trying to calculate the loss for Test dataset 
    N_Test = len(test_loader)
    print('The number of images in test loader {}'.format(N_Test))
    print('We are testing for the training epoch {}'.format(epoch))
    print('-' * 10)
    

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if is_use_cuda else torch.Tensor

    target_real = Variable(Tensor(batchSize, 1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batchSize, 1).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    keep_all_batch_losses_G = []
    keep_all_batch_losses_D_A = []
    keep_all_batch_losses_D_B = []

    for i, sample_batched in enumerate(test_loader):

        # Prepare sample and target
        # image = torch.autograd.Variable(sample_batched['image'].to(device))  # full size
        input_A = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size ; image_half

        input_B = torch.autograd.Variable(
            sample_batched['complex_noise_img'].to(device))  # half size ; complex_image_tensor

        # Set model input
        real_A = input_A
        real_B = input_B

        real_A = real_A.to(device, dtype=torch.float)
        real_B = real_B.to(device, dtype=torch.float)
        # --------------------------------- Generators A2B and B2A -----------------------------------

        # Identity loss
        # G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)  # generate A from B
        same_B = F.interpolate(same_B, size=(173, 230), mode='bicubic', align_corners=False)
        loss_identity_B = criterion_identity(same_B, real_B) * 5.0
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)  # generate B from A
        same_A = F.interpolate(same_A, size=(173, 230), mode='bicubic', align_corners=False)
        loss_identity_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)  ###### One Image
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

        fake_A = netG_B2A(real_B)   ####### One Image
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)    #########  One image
        recovered_A = F.interpolate(recovered_A, size=(173, 230), mode='bicubic', align_corners=False)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = netG_A2B(fake_A) ######### One image
        recovered_B = F.interpolate(recovered_B, size=(173, 230), mode='bicubic', align_corners=False)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        # Total loss
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        losses_G.update(loss_G.data.item(), input_A.size(dim=0))
        losses_G_Identity.update((loss_identity_A + loss_identity_B).data.item(), input_A.size(dim=0))
        losses_G_GAN.update((loss_GAN_A2B + loss_GAN_B2A).data.item(), input_A.size(dim=0))
        losses_G_Cycle.update((loss_cycle_ABA + loss_cycle_BAB).data.item(), input_A.size(dim=0))

        keep_all_batch_losses_G.append(loss_G.item())

        # --------------------------------- Discriminator A --------------------------------------------------

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)  ######## one image
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        keep_all_batch_losses_D_A.append(loss_D_A.item())

        # ------------------------------------ Discriminator B ------------------------------------------------

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)  ######### one image
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        keep_all_batch_losses_D_B.append(loss_D_B.item())

        losses_G_D.update((loss_D_A + loss_D_B).data.item(), input_A.size(dim=0))      


        saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/CycleGAN_Resources/' + 'Batch_%d' % i + '_Complex_Imag_GT_make3D' + '.pt'
        saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/CycleGAN_Resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_make3D' + '.pt'
        saving_path_complex_imag_Pred_A = '/sanssauvegarde/homes/t20monda/CycleGAN_Resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_A_make3D' + '.pt'
        torch.save(real_B, saving_path_complex_imag_GT)
        torch.save(fake_B, saving_path_complex_imag_Pred)
        torch.save(fake_A, saving_path_complex_imag_Pred_A)
        
        # Log progress
        if i % 5 == 0:

            # Log to tensorboard in each 5th batch after
            writer_1.add_scalar('loss_G', loss_G, epoch)
            writer_1.add_scalar('loss_G_identity', (loss_identity_A + loss_identity_B), epoch)
            writer_1.add_scalar('loss_G_GAN', (loss_GAN_A2B + loss_GAN_B2A), epoch)
            writer_1.add_scalar('loss_G_cycle', (loss_cycle_ABA + loss_cycle_BAB), epoch)
            writer_1.add_scalar('loss_D', (loss_D_A + loss_D_B), epoch)

        # Now print the data/images for the last batch only      
        if i == N_Test-1:
            # Log to tensorboard
            writer_1.add_image('real_A', vutils.make_grid(real_A.data, nrow=6, normalize=False),
                        epoch)
            writer_1.add_image('real_B', vutils.make_grid(real_B.data, nrow=6, normalize=False),
                        epoch) 
            writer_1.add_image('fake_A', vutils.make_grid(fake_A.data, nrow=6, normalize=False),
                        epoch)
            writer_1.add_image('fake_B', vutils.make_grid(fake_B.data, nrow=6, normalize=False),
                        epoch) 
            # writer_1.add_image('recovered_A', vutils.make_grid(recovered_A.data, nrow=6, normalize=False),
            #             epoch)
            # writer_1.add_image('recovered_B', vutils.make_grid(recovered_B.data, nrow=6, normalize=False),
            #             epoch)         

    batch_mean_loss_G = (np.mean(keep_all_batch_losses_G))
    batch_mean_loss_D_A = (np.mean(keep_all_batch_losses_D_A))
    batch_mean_loss_D_B = (np.mean(keep_all_batch_losses_D_B))

    print('The average loss_G of all the batches, accumulated over this epoch is : {:.4f}'.format(batch_mean_loss_G))
    print('The average loss_G of all the batches, accumulated over this epoch is : {:.4f}'.format(batch_mean_loss_D_A))
    print('The average loss_G of all the batches, accumulated over this epoch is : {:.4f}'.format(batch_mean_loss_D_B))

        

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

import time
import argparse
import datetime
import numpy as np
import os
import math 
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils  
from torchvision import transforms  
from tensorboardX import SummaryWriter

from model_weight_make_3D import PTModel as Model
from model_3D_weight_make_3D import PTModel as Model3D

from loss import ssim
from data_4 import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize, simple_save_images


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.000001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=10, type=int, help='batch size')
    args = parser.parse_args()
    
    train_me_where = "from_middle" # "from_begining" 
    model_name = "densenet_multi_task"

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # Create model
    model_1 = Model().to(device)
    model_2 = Model3D().to(device)

    print('Model created.')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_1 = nn.DataParallel(model_1.cuda())
        model_2 = nn.DataParallel(model_2.cuda())

    print('model and cuda mixing done')

    # Training parameters
    optimizer_1 = torch.optim.Adam(model_1.parameters(), args.lr)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), args.lr)

    best_loss = 100.0
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)

    # Create and save  data
    # train_transform = Augmentation()
    # train_loader = getTrainingTestingData(batch_size=batch_size, transforms=train_transform)
    # torch.save(train_loader, '/sanssauvegarde/homes/t20monda/Dense_Depth_1/train_loader_make_3D.pkl')

    # Load data
    train_loader = torch.load('/sanssauvegarde/homes/t20monda/Dense_Depth_1/train_loader_make_3D.pkl')
  
    # Loss
    l1_criterion = nn.L1Loss()
    print("Total number of batches in train loader are :", len(train_loader))

    if train_me_where == "from_middle":
        checkpoint = torch.load('/sanssauvegarde/homes/t20monda/Dense_Depth_1/checkpoint/' + model_name + '/Models_Weight_1_Make_3D' + '.ckpt')

        model_1.load_state_dict(checkpoint['state_dict_1'])
        model_2.load_state_dict(checkpoint['state_dict_2'])

    # Start training...
    for epoch in range(args.epochs):
        
        losses = AverageMeter()
        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Switch to train model
        model_1.train()
        model_2.train()

        keep_all_batch_losses = []
        total_batch_loss = 0.0
        running_batch_loss = 0.0

        for i, sample_batched in enumerate(train_loader): 
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

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


            output_depth, weight_feature_depth = model_1(image_full)
            output_black_box, weight_feature_black_box = model_2(image_full)

            pred_complex_image = compute_complex_image(output_depth, output_black_box, beta_val_half, a_mat_half,
                                                       unit_mat_half, image_half)

            weight_depth_mean = torch.mean(weight_feature_depth, dim=0)
            weight_black_box_mean = torch.mean(weight_feature_black_box, dim=0)
        
            # Compute the loss
            l_depth = l1_criterion(output_depth, depth_half)
            l_ssim = torch.clamp((1 - ssim(output_depth, depth_half, val_range=1000.0 / 10.0)) * 0.5, 0, 1)
            loss_depth = ((1.0-weight_depth_mean[0]) * l_ssim) + (weight_depth_mean[0] * l_depth)

            loss_complex_image = l1_criterion(pred_complex_image, complex_image_tensor_half)
            loss_ssim_complex_image = torch.clamp((1 - ssim(pred_complex_image.float(), complex_image_tensor_half.float(),
                                                            val_range=1)) * 0.5, 0, 1)
            loss_complex_total = ((1.0-weight_black_box_mean[0]) * loss_ssim_complex_image) + (weight_black_box_mean[0] * loss_complex_image)

            # Update step
            total_batch_loss = loss_complex_total + loss_depth
            running_batch_loss += total_batch_loss.item() * image_full.size(0)  # dividing by number of images in the batch

            keep_all_batch_losses.append(total_batch_loss.item())

            losses.update(total_batch_loss.data.item(), image_full.size(0))
            total_batch_loss.backward()

            # Log progress
            niter = epoch*N +i

            optimizer_1.step()
            optimizer_2.step()

        if math.isnan(losses.avg):
            print('I need to check you')

        # Log progress; print after every epochs
        print('Epoch: [{:.4f}] \t The average loss until this epoch is: {:.4f} '.format(epoch, losses.avg))

        if losses.avg < best_loss:
            print("Here the training loss got reduced, hence printing")
            print('Current best epoch loss is {:.4f}'.format(losses.avg), 'previous best was {}'.format(best_loss))
            best_loss = losses.avg      
            _save_best_model(model_1, model_2, best_loss, epoch)


def compute_complex_image(output_depth, output_black_box, beta_val, a_mat, unit_mat, image_half):

    output_depth_3d = torch.tile(output_depth, [1, 3, 1, 1])
    output_black_box_3d = output_black_box # torch.tile(output_black_box, [1, 3, 1, 1])

    tx1 = torch.exp(-torch.mul(beta_val, output_depth_3d))
    second_term = torch.mul(a_mat, (torch.subtract(unit_mat, tx1)))
    haze_image = torch.add((torch.mul(image_half, tx1)), second_term)

    pred_complex_image = output_black_box_3d + haze_image
    return pred_complex_image

def _save_best_model(model_1, model_2, best_loss, epoch):
    # Save Model
    model_name = "densenet_multi_task"
    state = {
        'state_dict_1': model_1.state_dict(),
        'state_dict_2': model_2.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }

    if not os.path.isdir('/sanssauvegarde/homes/t20monda/Dense_Depth_1/checkpoint/' + model_name):
        os.makedirs('/sanssauvegarde/homes/t20monda/Dense_Depth_1/checkpoint/' + model_name)

    torch.save(state,
               '/sanssauvegarde/homes/t20monda/Dense_Depth_1/checkpoint/' +
               model_name + '/Models_Weight_1_Make_3D' + '.ckpt')  


if __name__ == '__main__':
    main()

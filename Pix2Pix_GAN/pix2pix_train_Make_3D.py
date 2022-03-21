#!/usr/bin/python3

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

import torchvision.utils as vutils
from torch.autograd import Variable

from models import *
from tensorboardX import SummaryWriter
from data_3 import getTrainingTestingData
from utils_custom import AverageMeter

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10, help='linearly decaying the learning rate to 0')

    # parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')

    opt = parser.parse_args()
    print(opt)

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    train_me_where = "from_begining" # "from_middle" 
    model_name = "cycle_gan_task"

    # -------------------------- Losses --------------------------------------------------
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Initialize generator and discriminator
    generator = GeneratorUNet_Make3D()
    discriminator = Discriminator()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

    # ------------------------------ Definition of variables -------------------------------------
    generator.to(device)
    discriminator.to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        nn.DataParallel(generator.cuda())
        nn.DataParallel(discriminator.cuda())

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    best_loss_G = 100.0
    best_loss_D = 100.0
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                          opt.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D,
                                                       lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,
                                                                          opt.decay_epoch).step)

    # Create and save  data
    train_loader = torch.load('/sanssauvegarde/homes/t20monda/DenseDepth_3/train_loader_make_3D.pkl')

    if train_me_where == "from_middle":
        checkpoint_G_A2B = torch.load(
            '/homes/t20monda/Pix2Pix_GAN/checkpoint/' + model_name + '/Models_Make3D' + '_%s' % 'netG_A2B' + '.ckpt')
        checkpoint_G_B2A = torch.load(
            '/homes/t20monda/Pix2Pix_GAN/checkpoint/' + model_name + '/Models_Make3D' + '_%s' % 'netD' + '.ckpt')

        generator.load_state_dict(checkpoint_G_A2B['state_dict'])
        discriminator.load_state_dict(checkpoint_G_B2A['state_dict'])

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, 173 // 2 ** 4, 230 // 2 ** 4)
    # Tensor type
    Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100
    prev_time = time.time()

    # ------------------------ Training ----------------------------------------------
    for epoch in range(opt.epoch, opt.n_epochs):

        losses_G = AverageMeter()
        losses_D = AverageMeter()

        N = len(train_loader)
        print('Epoch {}/{}'.format(epoch, opt.n_epochs - 1))
        print('-' * 10)

        # Switch to train model
        generator.train()
        discriminator.train()

        loss_G = 0.0
        loss_D = 0.0

        keep_all_batch_losses_G = []
        running_batch_losses_G = 0.0

        keep_all_batch_losses_D = []
        running_batch_losses_D = 0.0

        for i, sample_batched in enumerate(train_loader):

            # Prepare sample and target
            input_A = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size ; image_half

            input_B = torch.autograd.Variable(
                sample_batched['complex_noise_img'].to(device))  # half size ; complex_image_tensor

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((input_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((input_A.size(0), *patch))), requires_grad=False)

            #  Train Generators
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(input_A)
            fake_B = F.interpolate(fake_B, size=(173, 230), mode='bicubic', align_corners=False)

            pred_fake = discriminator(fake_B, input_A)
            loss_GAN = criterion_GAN(pred_fake, valid)

            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, input_B)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            running_batch_losses_G += loss_G.item() * input_A.size(0)
            losses_G.update(loss_G.data.item(), input_A.size(0))

            keep_all_batch_losses_G.append(loss_G.item())
            loss_G.backward()
            optimizer_G.step()

            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            input_B = input_B.type(torch.cuda.FloatTensor) # converting into double tensor
            pred_real = discriminator(input_B, input_A)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(fake_B.detach(), input_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            running_batch_losses_D += loss_D.item() * input_A.size(0)
            losses_D.update(loss_D.data.item(), input_A.size(0))

            keep_all_batch_losses_D.append(loss_D.item())

            loss_D.backward()
            optimizer_D.step()

            # Determine approximate time left
            batches_done = epoch * len(train_loader) + i
            batches_left = opt.n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()


        epoch_loss_G = running_batch_losses_G / N  # epoch error
        batch_mean_loss_G = (np.mean(keep_all_batch_losses_G))

        epoch_loss_D = running_batch_losses_D / N  # epoch error
        batch_mean_loss_D = (np.mean(keep_all_batch_losses_D))

        # Log progress; print after every epochs
        print('Epoch: [{:.4f}] \t The loss_G of this epoch is: {:.4f} '.format(epoch, epoch_loss_G))
        print(
            'The average loss_G of all the batches, accumulated over this epoch is : {:.4f}'.format(batch_mean_loss_G))

        print('Epoch: [{:.4f}] \t The loss_D_A of this epoch is: {:.4f} '.format(epoch, epoch_loss_D))
        print('The average loss_G of all the batches, accumulated over this epoch is : {:.4f}'.format(
            batch_mean_loss_D))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        if epoch_loss_G < best_loss_G:
            print("Here the training loss_G got reduced, hence printing")
            print('Current best epoch loss_G is {:.4f}'.format(epoch_loss_G),
                  'previous best was {}'.format(best_loss_G))
            best_loss_G = epoch_loss_G
            _save_best_model(generator, best_loss_G, epoch, 'netG_A2B')

        if epoch_loss_D < best_loss_D:
            print("Here the training loss_D_A got reduced, hence printing")
            print('Current best epoch loss_D_A is {:.4f}'.format(epoch_loss_D),
                  'previous best was {}'.format(best_loss_D))
            best_loss_D = epoch_loss_D
            _save_best_model(discriminator, best_loss_D, epoch, 'netD')


def _save_best_model(model_1, best_loss, epoch, str_to_add):
    # Save Model
    model_name = "cycle_gan_task"
    state = {
        'state_dict': model_1.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }

    if not os.path.isdir('/homes/t20monda/Pix2Pix_GAN/checkpoint/' + model_name):
        os.makedirs('/homes/t20monda/Pix2Pix_GAN/checkpoint/' + model_name)

    torch.save(state,
               '/homes/t20monda/Pix2Pix_GAN/checkpoint/' +
               model_name + '/Models_Make3D' + '_%s' % str_to_add + '.ckpt')


if __name__ == '__main__':
    main()

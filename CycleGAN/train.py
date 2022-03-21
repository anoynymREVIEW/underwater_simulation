#!/usr/bin/python3

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

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger

from data_3 import getTrainingTestingData
from utils import weights_init_normal
from datasets import ImageDataset
from utils_custom import AverageMeter
import numpy as np
import warnings
warnings.filterwarnings("error")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=3, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10, help='linearly decaying the learning rate to 0')

    # parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')

    opt = parser.parse_args()
    # print(opt)

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")
    
    train_me_where = "from_middle" # "from_begining"
    model_name = "cycle_gan_task"

    # ------------------------------ Definition of variables -------------------------------------
    # Networks
    netG_A2B = Generator(opt.input_nc, opt.output_nc)
    netG_B2A = Generator(opt.output_nc, opt.input_nc)
    netD_A = Discriminator(opt.input_nc)  # discriminate the generated A
    netD_B = Discriminator(opt.output_nc)  # discriminate the generated B

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

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    best_loss_G = 100.0
    best_loss_D_A = 100.0
    best_loss_D_B = 100.0

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                    lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                        lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                        lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if is_use_cuda else torch.Tensor

    target_real = Variable(Tensor(opt.batchSize, 1).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize, 1).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Create and save  data
    train_loader = getTrainingTestingData(batch_size=opt.batchSize)
    # torch.save(train_loader, '/sanssauvegarde/homes/t20monda/DenseDepth_2/train_loader.pkl')

    writer_1 = SummaryWriter('/home/tamondal/CycleGAN/runs/real_DenseDepth_2_running')

    if train_me_where == "from_middle":

        checkpoint_G_A2B = torch.load('/home/tamondal/CycleGAN/checkpoint/' + model_name + '/Models' + '_%s' % 'netG_A2B' + '.ckpt')
        checkpoint_G_B2A = torch.load('/home/tamondal/CycleGAN/checkpoint/' + model_name + '/Models' + '_%s' % 'netG_B2A' + '.ckpt')
        checkpoint_D_A = torch.load('/home/tamondal/CycleGAN/checkpoint/' + model_name + '/Models' + '_%s' % 'netD_A' + '.ckpt')
        checkpoint_D_B = torch.load('/home/tamondal/CycleGAN/checkpoint/' + model_name + '/Models' + '_%s' % 'netD_B' + '.ckpt')

        netG_A2B.load_state_dict(checkpoint_G_A2B['state_dict'])
        netG_B2A.load_state_dict(checkpoint_G_B2A['state_dict'])
        netD_A.load_state_dict(checkpoint_D_A['state_dict'])
        netD_B.load_state_dict(checkpoint_D_B['state_dict'])

    # ------------------------ Training ----------------------------------------------
    for epoch in range(opt.epoch, opt.n_epochs):

        losses_G = AverageMeter()
        losses_G_Identity = AverageMeter()
        losses_G_GAN = AverageMeter()
        losses_G_Cycle = AverageMeter()
        losses_G_D = AverageMeter()

        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, opt.n_epochs - 1))
        print('-' * 10)

        # Switch to train model
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()

        loss_G = 0.0
        loss_D_A = 0.0
        loss_D_B = 0.0

        keep_all_batch_losses_G = []
        running_batch_losses_G = 0.0

        keep_all_batch_losses_D_A = []
        running_batch_losses_D_A = 0.0

        keep_all_batch_losses_D_B = []
        running_batch_losses_D_B = 0.0

        need_train = round((N*96)/100)  # 96 %
        cnt_batch = 0

        for i, sample_batched in enumerate(train_loader):

            if cnt_batch < need_train : # to skip the last batch from training   
                # Prepare sample and target
                image = torch.autograd.Variable(sample_batched['image'].to(device))  # full size
                input_A = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size ; image_half

                input_B = torch.autograd.Variable(
                    sample_batched['complex_noise_img'].to(device))  # half size ; complex_image_tensor

                # Set model input
                real_A = input_A
                real_B = input_B

                del input_A, input_B

                real_A = real_A.to(device, dtype=torch.float)
                real_B = real_B.to(device, dtype=torch.float)
                # --------------------------------- Generators A2B and B2A -----------------------------------
                optimizer_G.zero_grad()

                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                same_B = netG_A2B(real_B)  # generate A from B
                loss_identity_B = criterion_identity(same_B, real_B) * 5.0
                # G_B2A(A) should equal A if real A is fed
                same_A = netG_B2A(real_A)  # generate B from A
                loss_identity_A = criterion_identity(same_A, real_A) * 5.0

                # GAN loss
                fake_B = netG_A2B(real_A)
                pred_fake = netD_B(fake_B)
                loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

                fake_A = netG_B2A(real_B)
                pred_fake = netD_A(fake_A)
                loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

                # Cycle loss
                recovered_A = netG_B2A(fake_B)
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

                recovered_B = netG_A2B(fake_A)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

                # Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                running_batch_losses_G += loss_G.item() * image.size(dim=0) 

                losses_G.update(loss_G.data.item(), image.size(dim=0))
                losses_G_Identity.update((loss_identity_A + loss_identity_B).data.item(), image.size(dim=0))
                losses_G_GAN.update((loss_GAN_A2B + loss_GAN_B2A).data.item(), image.size(dim=0))
                losses_G_Cycle.update((loss_cycle_ABA + loss_cycle_BAB).data.item(), image.size(dim=0))

                keep_all_batch_losses_G.append(loss_G.item())
                loss_G.backward()
                optimizer_G.step()

                # --------------------------------- Discriminator A --------------------------------------------------
                optimizer_D_A.zero_grad()

                # Real loss
                pred_real = netD_A(real_A)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5

                running_batch_losses_D_A += loss_D_A.item() * image.size(dim=0)
                keep_all_batch_losses_D_A.append(loss_D_A.item())
                loss_D_A.backward()

                optimizer_D_A.step()

                # ------------------------------------ Discriminator B ------------------------------------------------
                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real, target_real)

                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) * 0.5

                running_batch_losses_D_B += loss_D_B.item() * image.size(dim=0)
                keep_all_batch_losses_D_B.append(loss_D_B.item())
                loss_D_B.backward()

                optimizer_D_B.step()

                losses_G_D.update((loss_D_A + loss_D_B).data.item(), image.size(dim=0))      

            else :
                break
            cnt_batch = cnt_batch+1 

        epoch_loss_G = running_batch_losses_G / need_train  # epoch error
        batch_mean_loss_G = (np.mean(keep_all_batch_losses_G))

        epoch_loss_D_A = running_batch_losses_D_A / need_train  # epoch error
        batch_mean_loss_D_A = (np.mean(keep_all_batch_losses_D_A))

        epoch_loss_D_B = running_batch_losses_D_B / need_train  # epoch error
        batch_mean_loss_D_B = (np.mean(keep_all_batch_losses_D_B))

        # Log progress; print after every epochs
        print('Epoch: [{:.4f}] \t The loss_G of this epoch is: {:.4f} '.format(epoch, epoch_loss_G))
        print('The average loss_G of all the batches, accumulated over this epoch is : {:.4f}'.format(batch_mean_loss_G))

        print('Epoch: [{:.4f}] \t The loss_D_A of this epoch is: {:.4f} '.format(epoch, epoch_loss_D_A))
        print('The average loss_G of all the batches, accumulated over this epoch is : {:.4f}'.format(batch_mean_loss_D_A))

        print('Epoch: [{:.4f}] \t The loss_D_B of this epoch is: {:.4f} '.format(epoch, epoch_loss_D_B))
        print('The average loss_G of all the batches, accumulated over this epoch is : {:.4f}'.format(batch_mean_loss_D_B))

        # save the losses in each epoch
        writer_1.add_scalar('loss_G', loss_G, epoch)
        writer_1.add_scalar('loss_G_identity', (loss_identity_A + loss_identity_B), epoch)
        writer_1.add_scalar('loss_G_GAN', (loss_GAN_A2B + loss_GAN_B2A), epoch)
        writer_1.add_scalar('loss_G_cycle', (loss_cycle_ABA + loss_cycle_BAB), epoch)
        writer_1.add_scalar('loss_D', (loss_D_A + loss_D_B), epoch)

        if epoch == (opt.n_epochs-1): # if we are running the last epochs
            if cnt_batch == (need_train-1): # if we are running the last batch
                # Log to tensorboard
                writer_1.add_image('real_A', vutils.make_grid(real_A.data, nrow=6, normalize=False),
                            epoch)
                writer_1.add_image('real_B', vutils.make_grid(real_B.data, nrow=6, normalize=False),
                            epoch) 
                writer_1.add_image('fake_A', vutils.make_grid(fake_A.data, nrow=6, normalize=False),
                            epoch)
                writer_1.add_image('fake_B', vutils.make_grid(fake_B.data, nrow=6, normalize=False),
                            epoch)                            

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if epoch_loss_G < best_loss_G:
            print("Here the training loss_G got reduced, hence printing")
            print('Current best epoch loss_G is {:.4f}'.format(epoch_loss_G), 'previous best was {}'.format(best_loss_G))
            best_loss_G = epoch_loss_G      
            _save_best_model(netG_A2B, best_loss_G, epoch, 'netG_A2B')
            _save_best_model(netG_B2A, best_loss_G, epoch, 'netG_B2A')

        if epoch_loss_D_A < best_loss_D_A:
            print("Here the training loss_D_A got reduced, hence printing")
            print('Current best epoch loss_D_A is {:.4f}'.format(epoch_loss_D_A), 'previous best was {}'.format(best_loss_D_A))
            best_loss_D_A = epoch_loss_D_A      
            _save_best_model(netD_A, best_loss_D_A, epoch, 'netD_A')

        if epoch_loss_D_B < best_loss_D_B:
            print("Here the training loss_D_B got reduced, hence printing")
            print('Current best epoch loss_D_B is {:.4f}'.format(epoch_loss_D_B), 'previous best was {}'.format(best_loss_D_B))
            best_loss_D_B = epoch_loss_D_B      
            _save_best_model(netD_B, best_loss_D_B, epoch, 'netD_B')        


def _save_best_model(model_1, best_loss, epoch, str_to_add):
    # Save Model
    model_name = "cycle_gan_task"
    state = {
        'state_dict': model_1.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }

    if not os.path.isdir('/home/tamondal/CycleGAN/checkpoint/' + model_name):
        os.makedirs('/home/tamondal/CycleGAN/checkpoint/' + model_name)

    torch.save(state,
               '/home/tamondal/CycleGAN/checkpoint/' +
               model_name + '/Models' + '_%s' % str_to_add + '.ckpt')  

if __name__ == '__main__':
    main()
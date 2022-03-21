import time
import argparse
import datetime
import os
import numpy as np
import torch
import math 
import sys

import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils  
from torchvision import transforms  
from tensorboardX import SummaryWriter

from model_3D import PTModel as Model3D
from loss import ssim
from data_3 import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize, simple_save_images


# This code is very same as "train_2.py" but here I am using the saved "atmospheric_light.npy" and "beta.npy" files
# we are also using here the "index_haze_image.npz", "index_complex_haze_image.npz" and "index_complex_depth_half_3d.npz" files, during training 

# here I am just adding the haze image in loss function
# and also adding here the direct link to generate the complex ricardo image
def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size')
    args = parser.parse_args()

    train_me_where = "from_begining" #  "from_middle" 
    model_name = "densenet_multi_task"

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # Create model
    model_3 = Model3D().to(device)

    print('Model created.')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_3 = nn.DataParallel(model_3.cuda())

    print('model and cuda mixing done')

    # Training parameters
    optimizer_3 = torch.optim.Adam(model_3.parameters(), args.lr)
    
    best_loss = 100.0
    batch_size = args.bs
    prefix = 'densenet_' + str(batch_size)
   
    # Load data
    train_loader = torch.load('/sanssauvegarde/homes/t20monda/Dense_Depth_1/train_loader.pkl')
    # train_loader = getTrainingTestingData(batch_size=batch_size)
    # torch.save(train_loader, '/home/tamondal/Kepp_All/Ricardo_Direct/train_loader.pkl')
    
    # Loss
    l1_criterion = nn.L1Loss()

    print("Total number of batches in train loader are :", len(train_loader))

    if train_me_where == "from_middle":
        checkpoint = torch.load('/home/tamondal/Kepp_All/Ricardo_Direct/checkpoint/' + model_name + '/Models' + '.ckpt')
        model_3.load_state_dict(checkpoint['state_dict_3'])

    # Start training...
    for epoch in range(args.epochs):

        losses = AverageMeter()
        N = len(train_loader)

        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        # Switch to train mode
        model_3.train()

        keep_all_batch_losses = []
        total_batch_loss = 0.0
        running_batch_loss = 0.0
        
        need_train = round((N*96)/100)
        cnt_batch = 0

        for i, sample_batched in enumerate(train_loader):
            if cnt_batch < need_train : # to skip the last batch from training    
            # with torch.autograd.set_detect_anomaly(True):
                optimizer_3.zero_grad()

                # Prepare sample and target
                image = torch.autograd.Variable(sample_batched['image'].to(device))  # full size
                complex_image_tensor = torch.autograd.Variable(sample_batched['complex_noise_img'].to(device))  # half size

                output_ricardo_image_direct = model_3(image)

                # Compute the loss for complex haze image which is calculated directly from the original image
                loss_ricardo_image_direct = l1_criterion(output_ricardo_image_direct, complex_image_tensor)
                loss_ssim_ricardo_image_direct = torch.clamp((1 - ssim(output_ricardo_image_direct.float(), complex_image_tensor.float(),
                                                                val_range=1)) * 0.5, 0, 1)
                loss_ricardo_image_total = (1.0 * loss_ricardo_image_direct) + (0.1 * loss_ssim_ricardo_image_direct)
                del complex_image_tensor, output_ricardo_image_direct

                # Update step
                total_batch_loss = loss_ricardo_image_total
                running_batch_loss += total_batch_loss.item() * image.size(0)  # dividing by number of images in the batch

                keep_all_batch_losses.append(total_batch_loss.item())
                losses.update(total_batch_loss.data.item(), image.size(0))
                total_batch_loss.backward()

                optimizer_3.step()
                
                torch.cuda.empty_cache()  

            else :
                break
            cnt_batch = cnt_batch+1    

        if math.isnan(losses.avg):
            print('I need to check you')

        # Log progress; print after every epochs into the console
        print('Epoch: [{:.4f}] \t The loss of this epoch is: {:.4f} '.format(epoch, losses.avg))
        
        # Log to tensorboard
        # writer_1.add_scalar('Train/Each Epoch Loss', epoch_loss, niter)

        if losses.avg < best_loss:
            print("Here the training loss got reduced, hence printing")
            print('Current best epoch loss is {:.4f}'.format(losses.avg), 'previous best was {}'.format(best_loss))
            best_loss = losses.avg            
            _save_best_model(model_3, best_loss, epoch)
            

def _save_best_model(model_3, best_loss, epoch):
    # Save Model
    model_name = "densenet_multi_task"
    state = {
        'state_dict_3': model_3.state_dict(),
        'best_acc': best_loss,
        'cur_epoch': epoch
    }

    if not os.path.isdir('/home/tamondal/Kepp_All/Ricardo_Direct/checkpoint/' + model_name):
        os.makedirs('/home/tamondal/Kepp_All/Ricardo_Direct/checkpoint/' + model_name)

    torch.save(state,
               '/home/tamondal/Kepp_All/Ricardo_Direct/checkpoint/' +
               model_name + '/Models' + '.ckpt')  


if __name__ == '__main__':
    main()

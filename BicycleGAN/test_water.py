import torch
import torchvision

from dataloader import data_loader
import model
import util

import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils

import os.path
from os import path
from tensorboardX import SummaryWriter
import os
import numpy as np
import argparse


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
    
    iter_dloader = iter(test_loader)
    sample_batched = iter_dloader.next()
    image = torch.autograd.Variable(sample_batched['image_half']) 
    img_num = image.size(0) # just to get the batch size
    del test_loader, image

    for i in range(need_train+1, N):
        # if cnt_batch > need_train : # to skip the last batch from training  
        for j in range(0,img_num):
            saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/BicycleGAN-Simple/resources/' + 'Batch_%d' % i + 'Item_%d' % j + '_Complex_Imag_GT' + '.pt'
            
            saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/BicycleGAN-Simple/resources/' + '2_Batch_%d' % i + 'Item_%d' % j + '_Complex_Imag_Pred' + '.pt'
            if path.exists(saving_path_complex_imag_GT) and path.exists(saving_path_complex_imag_Pred):
                complex_image_tensor = torch.load(saving_path_complex_imag_GT)  
                pred_complex_image_1 = torch.load(saving_path_complex_imag_Pred)

                abs_rel, rmse, log_10, a1, a2, a3 = add_results_1(complex_image_tensor, pred_complex_image_1, border_crop_size=16)

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


'''
    < make_interpolation >
    Make linear interpolated latent code.
    
    * Parameters
    n : Input images number
    img_num : Generated images number per one input image
    z_dim : Dimension of latent code. Basically 8.
'''
def make_interpolation(n=200, img_num=9, z_dim=8):
    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # Make interpolated z
    step = 1 / (img_num-1)
    alpha = torch.from_numpy(np.arange(0, 1, step))
    interpolated_z = torch.FloatTensor(n, img_num, z_dim).type(dtype)

    for i in range(n):
        first_z = torch.randn(1, z_dim)
        last_z = torch.randn(1, z_dim)
        
        for j in range(img_num-1):
            interpolated_z[i, j] = (1 - alpha[j]) * first_z + alpha[j] * last_z
        interpolated_z[i, img_num-1] = last_z
    
    return interpolated_z

'''
    < make_z >
    Make latent code
    
    * Parameters
    n : Input images number
    img_num : Generated images number per one input image
    z_dim : Dimension of latent code. Basically 8.
    sample_type : random or interpolation
'''
def make_z(n, img_num, z_dim=8, sample_type='random'):
    if sample_type == 'random':
        z = util.var(torch.randn(n, img_num, 8))
    elif sample_type == 'interpolation':
        z = util.var(make_interpolation(n=n, img_num=img_num, z_dim=z_dim))
    
    return z


'''
    < make_img >
    Generate images.
    
    * Parameters
    dloader : Dataloader
    G : Generator
    z : Random latent code with size of (N, img_num, z_dim)
    img_size : Image size. Now only 128 is available.
    img_num : Generated images number per one input image.
'''
def make_img(dloader, G, z, img_size=128):

    writer = SummaryWriter('/homes/t20monda/BicycleGAN-Simple/runs')
  
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    N = len(dloader)
    need_train = round((N*96)/100)
    cnt_batch = 0
    for i, sample_batched in enumerate(dloader):
        if cnt_batch > need_train : # to skip the last batch from training  
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image_half'])  # full size
            complex_image_tensor = torch.autograd.Variable(sample_batched['complex_noise_img'])  # half size
            
            image = util.var(image.type(dtype))
            complex_image_tensor = util.var(complex_image_tensor.type(dtype))

            img_num = image.size(0) 

            # Generate img_num images per a domain A image
            for j in range(img_num):
                img_ = image[j].unsqueeze(dim=0)
                complex_get = complex_image_tensor[j].unsqueeze(dim=0)
                z_ = z[i, j, :].unsqueeze(dim=0)
                
                saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/BicycleGAN-Simple/resources/' + 'Batch_%d' % i + 'Item_%d' % j + '_Complex_Imag_GT' + '.pt'
                torch.save(complex_get, saving_path_complex_imag_GT)

                out_img = G(img_, z_)
                out_img_1 = out_img /2 + 0.5  ## change here
                out_img_2 = out_img   ## change here

                saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/BicycleGAN-Simple/resources/' + '1_Batch_%d' % i + 'Item_%d' % j + '_Complex_Imag_Pred' + '.pt'
                torch.save(out_img_1, saving_path_complex_imag_Pred)

                saving_path_complex_imag_Pred = '/sanssauvegarde/homes/t20monda/BicycleGAN-Simple/resources/' + '2_Batch_%d' % i + 'Item_%d' % j + '_Complex_Imag_Pred' + '.pt'
                torch.save(out_img_2, saving_path_complex_imag_Pred)

            if i == N-1:
                # Log to tensorboard
                writer.add_image('complex_GT', vutils.make_grid(complex_get.data, nrow=6, normalize=False))
                writer.add_image('complex_pred_1', vutils.make_grid(out_img_1.data, nrow=6, normalize=False)) 
                writer.add_image('complex_pred_2', vutils.make_grid(out_img_2.data, nrow=6, normalize=False))        


        cnt_batch = cnt_batch+1
        
def main(args):  
    
    dloader = torch.load('/sanssauvegarde/homes/t20monda/BicycleGAN-Simple/train_loader.pkl')
    dlen = len(dloader)
    
    # ClacAccuracyOnly(dloader)
    # return 


    if torch.cuda.is_available() is True:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    if args.epoch is not None:
        weight_name = '{epoch}-G.pkl'.format(epoch=args.epoch)
    else:
        weight_name = 'G.pkl'
        
    weight_path = os.path.join(args.weight_dir, weight_name)
    G = model.Generator(z_dim=8).type(dtype)
    G.load_state_dict(torch.load(weight_path))
    G.eval()
    
    if os.path.exists(args.result_dir) is False:
        os.makedirs(args.result_dir)
        
    # For example, img_name = random_55.png
    if args.epoch is None:
        args.epoch = 'latest'
    img_name = '{type}_{epoch}.png'.format(type=args.sample_type, epoch=args.epoch)
    img_path = os.path.join(args.result_dir, img_name)

    # Make latent code and images
    z = make_z(n=dlen, img_num=args.img_num, z_dim=8, sample_type=args.sample_type)
    
    make_img(dloader, G, z, img_size=128)   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_type', type=str, choices=['random', 'interpolation'], default='random',
                        help='Type of sampling : \'random\' or \'interpolation\'') 
    parser.add_argument('--root', type=str, default='data/edges2shoes', 
                        help='Data location')
    parser.add_argument('--result_dir', type=str, default='/homes/t20monda/BicycleGAN-Simple/result',
                        help='Ouput images location')
    parser.add_argument('--weight_dir', type=str, default='/sanssauvegarde/homes/t20monda/BicycleGAN-Simple/weight',
                        help='Trained weight location of generator. pkl file location')
    parser.add_argument('--img_num', type=int, default=5,
                        help='Generated images number per one input image')
    parser.add_argument('--epoch', type=int, default=46,
                        help='Epoch that you want to see the result. If it is None, the most recent epoch')

    args = parser.parse_args()
    main(args)
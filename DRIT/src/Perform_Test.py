import torch
from options import TestOptions
from dataset import dataset_unpair
from model import DRIT
import argparse

import torch.nn as nn
import torch.nn.functional as F

from saver import Saver
from data_3 import getTrainingTestingData
from utils import weights_init_normal

def main():
    parser = TestOptions()
    opts = parser.parse()
    
    # Load data
    test_loader = torch.load('/sanssauvegarde/homes/t20monda/DRIT/train_loader.pkl')
    # LogProgress(test_loader, opts)
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

        saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % i + '_Complex_Imag_GT' + '.pt'

        saving_path_complex_imag_Pred_1 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_1' + '.pt'
        saving_path_complex_imag_Pred_2 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_2' + '.pt'
        saving_path_complex_imag_Pred_3 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_3' + '.pt'
        saving_path_complex_imag_Pred_4 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_4' + '.pt'
        saving_path_complex_imag_Pred_5 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % i + '_Complex_Imag_Pred_5' + '.pt'
        
        complex_image_tensor = torch.load(saving_path_complex_imag_GT)

        pred_complex_image_1 = torch.load(saving_path_complex_imag_Pred_1)
        pred_complex_image_2 = torch.load(saving_path_complex_imag_Pred_2)
        pred_complex_image_3 = torch.load(saving_path_complex_imag_Pred_3)
        pred_complex_image_4 = torch.load(saving_path_complex_imag_Pred_4)
        pred_complex_image_5 = torch.load(saving_path_complex_imag_Pred_5)


        abs_rel, rmse, log_10, a1, a2, a3 = add_results_1(complex_image_tensor, pred_complex_image_5, border_crop_size=16)

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


def LogProgress(test_loader, opts):

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # data loader
    print('\n--- load dataset ---')

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    
    model.resume(opts.resume, train=False)
    model.eval()

    N = len(test_loader)
    need_train = round((N*96)/100)
    cnt_batch = 0

    for it, sample_batched in enumerate(test_loader):
        if cnt_batch > need_train : # to skip the last batch from training    

            with torch.autograd.set_detect_anomaly(True) :
                image_half = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size
                orig_haze_image = torch.autograd.Variable(sample_batched['complex_noise_img'].to(device))  # half size
                
                orig_haze_image = orig_haze_image.type(torch.cuda.FloatTensor) # converting into double tensor

                # input data
                image_half = image_half.cuda(opts.gpu).detach()
                orig_haze_image = orig_haze_image.cuda(opts.gpu).detach()

                keep_all_imgs = [image_half]
                keep_all_names = ['input_RGB']

                keep_all_imgs.append(orig_haze_image)
                keep_all_names.append('complex_img_GT')

                for idx2 in range(opts.num):
                    with torch.no_grad():
                        pred_complex_img = model.test_forward(image_half, a2b=opts.a2b)
                    keep_all_imgs.append(pred_complex_img)
                    keep_all_names.append('output_{}'.format(idx2))
                
                saving_path_complex_imag_GT = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % it + '_Complex_Imag_GT' + '.pt'

                saving_path_complex_imag_Pred_1 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % it + '_Complex_Imag_Pred_1' + '.pt'
                saving_path_complex_imag_Pred_2 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % it + '_Complex_Imag_Pred_2' + '.pt'
                saving_path_complex_imag_Pred_3 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % it + '_Complex_Imag_Pred_3' + '.pt'
                saving_path_complex_imag_Pred_4 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % it + '_Complex_Imag_Pred_4' + '.pt'
                saving_path_complex_imag_Pred_5 = '/sanssauvegarde/homes/t20monda/DRIT/resources/' + 'Batch_%d' % it + '_Complex_Imag_Pred_5' + '.pt'
                
                torch.save(orig_haze_image, saving_path_complex_imag_GT)

                torch.save(keep_all_imgs[2], saving_path_complex_imag_Pred_1)
                torch.save(keep_all_imgs[3], saving_path_complex_imag_Pred_2)
                torch.save(keep_all_imgs[4], saving_path_complex_imag_Pred_3)
                torch.save(keep_all_imgs[5], saving_path_complex_imag_Pred_4)
                torch.save(keep_all_imgs[6], saving_path_complex_imag_Pred_5)

        cnt_batch = cnt_batch+1                           

    return

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

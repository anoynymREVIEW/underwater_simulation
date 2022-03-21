import torch
from options import TrainOptions
from dataset import dataset_unpair
from model import DRIT
from saver import Saver
from data_3 import getTrainingTestingData
from utils import weights_init_normal


def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    is_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_use_cuda else "cpu")

    # data loader
    print('\n--- load dataset ---')

    # Create and save  data
    # train_loader = getTrainingTestingData(batch_size=opts.batch_size)
    # torch.save(train_loader, '/home/tamondal/DRIT/train_loader.pkl')
    train_loader = torch.load('/home/tamondal/DRIT/train_loader.pkl')

    # model
    print('\n--- load model ---')
    model = DRIT(opts)
    model.setgpu(opts.gpu)
    if opts.resume is None:
        model.initialize()
        ep0 = -1
        total_it = 0
    else:
        ep0, total_it = model.resume(opts.resume)
    model.set_scheduler(opts, last_ep=ep0)
    ep0 += 1
    print('start the training at epoch %d' % (ep0))

    # saver for display and output
    saver = Saver(opts)

    N = len(train_loader)
    need_train = round((N*96)/100)

    # train
    print('\n--- train ---')
    max_it = 500000
    for ep in range(ep0, opts.n_ep):
        
        cnt_batch = 0
        for it, sample_batched in enumerate(train_loader):
            if cnt_batch < need_train : # to skip the last batch from training    

                with torch.autograd.set_detect_anomaly(True) :
                    image_half = torch.autograd.Variable(sample_batched['image_half'].to(device))  # half size
                    orig_haze_image = torch.autograd.Variable(sample_batched['complex_noise_img'].to(device))  # half size


                    if image_half.size(0) != opts.batch_size or orig_haze_image.size(0) != opts.batch_size:
                        continue
                    
                    orig_haze_image = orig_haze_image.type(torch.cuda.FloatTensor) # converting into double tensor

                    # input data
                    image_half = image_half.cuda(opts.gpu).detach()
                    orig_haze_image = orig_haze_image.cuda(opts.gpu).detach()

                    # update model
                    if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
                        model.update_D_content(image_half, orig_haze_image)
                        continue
                    else:
                        model.update_D(image_half, orig_haze_image)
                        model.update_EG()

                    # save to display file
                    if not opts.no_display_img:
                        saver.write_display(total_it, model)

                    print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
                    print('disALoss: %08f disA2Loss %08f, disBLoss %08f, disB2Loss %08f, disContentLoss %08f' % (model.disA_loss, model.disA2_loss, model.disB_loss, model.disB2_loss, model.disContent_loss))
                    total_it += 1
                    # if total_it >= max_it:
                    #     saver.write_img(-1, model)
                    #     saver.write_model(-1, model)
                    #     break
            else :
                break
            cnt_batch = cnt_batch+1                           

        # decay learning rate
        if opts.n_ep_decay > -1:
            model.update_lr()

        # save result image after each epoch
        # saver.write_img(ep, model)
        saver.write_img(-1, model)

        # Save network weights after each epoch
        # saver.write_model(ep, total_it, model)
        saver.write_model(-1, total_it, model)

    return


if __name__ == '__main__':
    main()

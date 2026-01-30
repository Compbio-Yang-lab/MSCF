"""General-purpose training script for multi-task image-to-image translation.

This script works for various models (with option '--model': e.g., DeepLIIF) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.
"""
import time
from Model.options.base_options import TrainOptions
from Model.data import create_dataset
from Model.models import create_model, postprocess
from Model.options import read_model_params, Options, print_options
from Model.util.visualizer import Visualizer
from Model.util.util import prepare_training_config
from PIL import Image
import os
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random
import json
import torch
import torch.distributed as dist
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import click

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

def handle_visualization_logging_checkpointing(
    model, visualizer, epoch, epoch_iter, total_iters, dataset,
    display_freq, update_html_freq, print_freq, display_id,
    save_latest_freq, save_by_iter, debug, debug_data_size,n_epochs, n_epochs_decay#加这俩
):
    """
    Handles visualization, logging, checkpointing, and debug break.
    Returns True if debug mode should trigger an early break.
    """
    # Visualization
    if total_iters % display_freq == 0:
        save_result = total_iters % update_html_freq == 0
        model.compute_visuals()
        visualizer.display_current_results(
            {**model.get_current_visuals()}, epoch, save_result
        )

    # Logging
    if total_iters % print_freq == 0:
        losses = model.get_current_losses()
        visualizer.print_current_losses(epoch, epoch_iter, {**losses})
        if display_id > 0:
            visualizer.plot_current_losses(
                epoch, float(epoch_iter) / len(dataset), {**losses}
            )

    # Checkpointing
   # if total_iters % save_latest_freq == 0:
    #    print(f"Saving latest model (epoch {epoch}, total_iters {total_iters})")
     #   suffix = f"iter_{total_iters}" if save_by_iter else "latest"
      #  model.save_networks(suffix)

    total_epochs = n_epochs + n_epochs_decay
    if total_iters % save_latest_freq == 0 or (epoch >= total_epochs - 2 and epoch_iter == 0):
        print(f"Saving latest model (epoch {epoch}, total_iters {total_iters})")
        suffix = f"iter_{total_iters}" if save_by_iter else "latest"
        model.save_networks(suffix)

    # Debug early exit
    if debug and epoch_iter >= debug_data_size:
        print(f"[DEBUG] Early stop: epoch {epoch}, iter {epoch_iter} >= {debug_data_size}")
        return True

    return False

        
@click.command()
@click.option('--dataroot', required=True, type=str,
              help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
@click.option('--name', default='experiment_name',
              help='name of the experiment. It decides where to store samples and models')
@click.option('--gpu-ids', type=int, default=[0], multiple=True, help='gpu-ids 0 gpu-ids 1 or gpu-ids -1 for CPU')
@click.option('--checkpoints-dir', default='./checkpoints', help='models are saved here')
@click.option('--modalities-no', default=4, type=int, help='number of targets')
# model parameters
@click.option('--model', default='ParsiLIF', help='name of model class')
@click.option('--seghead', default=['unet_adv2'], help='name of seghead')
@click.option('--seg-weights', default='', type=str, help='weights used to aggregate modality images for the final segmentation image; numbers should add up to 1, and each number corresponds to the modality in order; example: 0.25,0.15,0.25,0.1,0.25')
@click.option('--loss-weights-g', default='', type=str, help='weights used to aggregate modality-wise losses for the final loss; numbers should add up to 1, and each number corresponds to the modality in order; example: 0.2,0.2,0.2,0.2,0.2')
@click.option('--loss-weights-d', default='', type=str, help='weights used to aggregate modality-wise losses for the final loss; numbers should add up to 1, and each number corresponds to the modality in order; example: 0.2,0.2,0.2,0.2,0.2')
@click.option('--input-nc', default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
@click.option('--output-nc', default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
@click.option('--ngf', default=32, help='# of gen filters in the last conv layer')
@click.option('--ndf', default=32, help='# of discrim filters in the first conv layer')
@click.option('--net-d', default='n_layers',
              help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 '
                   'PatchGAN. n_layers allows you to specify the layers in the discriminator')
@click.option('--net-g', default='resnet_9blocks,resnet_9blocks,resnet_9blocks,resnet_9blocks',
              help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_512 | unet_256 | unet_128 | unet_512_attention]; to specify different arch for generators, list arch for each generator separated by comma, e.g., --net-g=resnet_9blocks,resnet_9blocks,resnet_9blocks,unet_512_attention,unet_512_attention')
@click.option('--n-layers-d', default=4, help='only used if netD==n_layers')
@click.option('--norm', default='batch',
              help='instance normalization or batch normalization [instance | batch | none]')
@click.option('--init-type', default='normal',
              help='network initialization [normal | xavier | kaiming | orthogonal]')
@click.option('--init-gain', default=0.02, help='scaling factor for normal, xavier and orthogonal.')
@click.option('--no-dropout', is_flag=True, help='no dropout for the generator')
@click.option('--upsample', default='convtranspose', help='use upsampling instead of convtranspose [convtranspose | resize_conv | pixel_shuffle]')
@click.option('--label-smoothing', type=float,default=0.0, help='label smoothing factor to prevent the discriminator from being too confident')
# dataset parameters
@click.option('--direction', default='AtoB', help='AtoB or BtoA')
@click.option('--serial-batches', is_flag=True,
              help='if true, takes images in order to make batches, otherwise takes them randomly')
@click.option('--num-threads', default=4, help='# threads for loading data')
@click.option('--batch-size', default=1, help='input batch size')
@click.option('--load-size', default=512, help='scale images to this size')
@click.option('--crop-size', default=512, help='then crop to this size')
@click.option('--max-dataset-size', type=int,
              help='Maximum number of samples allowed per dataset. If the dataset directory contains more than '
                   'max_dataset_size, only a subset is loaded.')
@click.option('--preprocess', type=str,
              help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | '
                   'scale_width_and_crop | none]')
@click.option('--no-flip', is_flag=True,
              help='if specified, do not flip the images for data augmentation')
@click.option('--display-winsize', default=512, help='display window size for both visdom and HTML')
# additional parameters
@click.option('--epoch', default='latest',
              help='which epoch to load? set to latest to use latest cached model')
@click.option('--load-iter', default=0,
              help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; '
                   'otherwise, the code will load models by [epoch]')
@click.option('--verbose', is_flag=True, help='if specified, print more debugging information')
@click.option('--lambda-L1', default=100.0, help='weight for L1 loss')
@click.option('--is-train', is_flag=True, default=True)
@click.option('--continue-train', is_flag=True, help='continue training: load the latest model')
@click.option('--epoch-count', type=int, default=0,
              help='the starting  epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>')
@click.option('--phase', default='train', help='train, val, test, etc')
# training parameters
@click.option('--n-epochs', type=int, default=5,
              help='number of epochs with the initial learning rate')
@click.option('--n-epochs-decay', type=int, default=5,
              help='number of epochs to linearly decay learning rate to zero')
@click.option('--optimizer', type=str, default='adam',
              help='optimizer from torch.optim to use, applied to both generators and discriminators [adam | sgd | adamw | ...]; the current parameters however are set up for adam, so other optimziers may encounter issue')
@click.option('--beta1', default=0.5, help='momentum term of adam')
#@click.option('--lr', default=0.0002, help='initial learning rate for adam')
@click.option('--lr-g', default=0.0003, help='initial learning rate for generator adam optimizer')
@click.option('--lr-d', default=0.0003, help='initial learning rate for discriminator adam optimizer')
@click.option('--lr-policy', default='linear',
              help='learning rate policy. [linear | step | plateau | cosine]')
@click.option('--lr-decay-iters', type=int, default=50,
              help='multiply by a gamma every lr_decay_iters iterations')
@click.option('--seed', type=int, default=None, help='basic seed to be used for deterministic training, default to None (non-deterministic)')
# visdom and HTML visualization parameters
@click.option('--display-freq', default=400, help='frequency of showing training results on screen')
@click.option('--display-ncols', default=4,
              help='if positive, display all images in a single visdom web panel with certain number of images per row.')
@click.option('--display-id', default=1, help='window id of the web display')
@click.option('--display-server', default="http://localhost", help='visdom server of the web display')
@click.option('--display-env', default='main',
              help='visdom display environment name (default is "main")')
@click.option('--display-port', default=8097, help='visdom port of the web display')
@click.option('--update-html-freq', default=50, help='frequency of saving training results to html')
@click.option('--print-freq', default=50, help='frequency of showing training results on console')
@click.option('--no-html', is_flag=True,
              help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
# network saving and loading parameters
@click.option('--save-latest-freq', default=500, help='frequency of saving the latest results')
@click.option('--save-epoch-freq', default=2,
              help='frequency of saving checkpoints at the end of epochs')
@click.option('--save-by-iter', is_flag=True, help='whether saves model by iteration')
@click.option('--remote', type=bool, default=False, help='whether isolate visdom checkpoints or not; if False, you can run a separate visdom server anywhere that consumes the checkpoints')
@click.option('--remote-transfer-cmd', type=str, default=None, help='module and function to be used to transfer remote files to target storage location, for example mymodule.myfunction')
@click.option('--dataset-mode', type=str, default='aligned',
              help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
@click.option('--padding', type=str, default='zero',
              help='chooses the type of padding used by resnet generator. [reflect | zero]')
# DeepLIIFExt params
@click.option('--seg-gen', type=bool, default=True, help='True (Translation and Segmentation), False (Only Translation).')
@click.option('--net-ds', type=str, default='n_layers',
              help='specify discriminator architecture for segmentation task [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
@click.option('--net-gs', type=str, default='unet_512',
              help='specify generator architecture for segmentation task [resnet_9blocks | resnet_6blocks | unet_512 | unet_256 | unet_128 | unet_512_attention]; to specify different arch for generators, list arch for each generator separated by comma, e.g., --net-g=resnet_9blocks,resnet_9blocks,resnet_9blocks,unet_512_attention,unet_512_attention')
@click.option('--gan-mode', type=str, default='vanilla',
              help='the type of GAN objective for translation task. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
@click.option('--gan-mode-s', type=str, default='lsgan',
              help='the type of GAN objective for segmentation task. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
# DDP related arguments
@click.option('--local-rank', type=int, default=None, help='placeholder argument for torchrun, no need for manual setup')
# Others
@click.option('--with-val', is_flag=True,
              help='use validation set to evaluate model performance at the end of each epoch')
@click.option('--debug', default=False,
              help='debug mode, limits the number of data points per epoch to a small value')
@click.option('--debug-data-size', default=10, type=int, help='data size per epoch used in debug mode; due to batch size, the epoch will be passed once the completed no. data points is greater than this value (e.g., for batch size 3, debug data size 10, the effective size used in training will be 12)')


def train(dataroot, name, gpu_ids, checkpoints_dir, input_nc, output_nc, ngf, ndf, net_d, net_g,
          n_layers_d, norm, init_type, init_gain, no_dropout, upsample, label_smoothing, direction, serial_batches, num_threads,
          batch_size, load_size, crop_size, max_dataset_size, preprocess, no_flip, display_winsize, epoch, load_iter,
          verbose, lambda_l1, is_train, display_freq, display_ncols, display_id, display_server, display_env,
          display_port, update_html_freq, print_freq, no_html, save_latest_freq, save_epoch_freq, save_by_iter,
          continue_train, epoch_count, phase, lr_policy, n_epochs, n_epochs_decay, optimizer, beta1, lr_g, lr_d, lr_decay_iters,
          remote, remote_transfer_cmd, seed, dataset_mode, padding, model, seghead, seg_weights, loss_weights_g, loss_weights_d,
          modalities_no, seg_gen, net_ds, net_gs, gan_mode, gan_mode_s, local_rank, with_val, debug, debug_data_size):

    prep_config = prepare_training_config(
        dataroot=dataroot,
        gpu_ids=gpu_ids,
        model=model,
        net_g=net_g,
        net_gs=net_gs,
        dataset_mode=dataset_mode,
        modalities_no=modalities_no,
        seg_gen=seg_gen,
        seg_weights=seg_weights,
        loss_weights_g=loss_weights_g,
        loss_weights_d=loss_weights_d,
        seed=seed,
        padding=padding,
        local_rank=local_rank,
    )
    d_params = locals()
    d_params.update(prep_config)
    opt = Options(d_params=d_params)
    print_options(opt, save=True)
    # set dir for train and val
    dataset = create_dataset(opt)

    # get the number of images in the dataset.
    click.echo('The number of training images = %d' % len(dataset))
    
    if with_val:
        dataset_val = create_dataset(opt,phase='test')
        data_val = [batch for batch in dataset_val]
        click.echo('The number of validation images = %d' % len(dataset_val))
        
        if model in ['ParsiLIF']:
            metrics_val = json.load(open(os.path.join(dataset_val.dataset.dir_AB,'metrics.json')))

    # create a model given model and other options
    print(opt.netG)
    model = create_model(opt)
    model.setup(opt)

    visualizer = Visualizer(opt)
    total_iters = 0

    print('start training')
    for epoch in range(epoch_count, n_epochs + n_epochs_decay + 1):
        epoch_iter = 0
        visualizer.reset()

        for i, data in tqdm(enumerate(dataset), total=int(len(dataset) / batch_size + 1), desc="Training"):
            total_iters += batch_size
            epoch_iter += batch_size
            model.set_input(data)
            model.update()

            should_break = handle_visualization_logging_checkpointing(
                model=model,visualizer=visualizer,epoch=epoch,epoch_iter=epoch_iter,
                total_iters=total_iters,dataset=dataset,display_freq=display_freq,
                update_html_freq=update_html_freq,
                print_freq=print_freq, display_id=display_id,
                save_latest_freq=save_latest_freq,
                save_by_iter=save_by_iter, debug=debug,debug_data_size=debug_data_size,
                n_epochs = n_epochs, n_epochs_decay = n_epochs_decay  # 添加这两个参数
            )

            if should_break: break

        # cache our model every <save_epoch_freq> epochs

        #if epoch % save_epoch_freq == 0:
         #   print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
          #  model.save_networks('latest')
           # model.save_networks(epoch)
        total_epochs = n_epochs + n_epochs_decay
        if epoch % save_epoch_freq == 0 or epoch >= total_epochs - 2:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)


        # validation loss and metrics calculation
        if with_val:
            losses = model.get_current_losses() # get training losses to print
            
            model.eval()
            l_losses_val = []
            l_metrics_val = []
            
            # for each val image, calculate validation loss and cell count metrics
            for j, data_val_batch in enumerate(data_val):
                # batch size is effectively 1 for validation
                model.set_input(data_val_batch)
                model.calculate_losses() # this does not optimize parameters
                visuals = model.get_current_visuals()  # get image results
                
                # val losses
                losses_val_batch = model.get_current_losses()
                l_losses_val += [(k,v) for k,v in losses_val_batch.items()]
                
                # calculate cell count metrics
                if type(model).__name__ == 'ParsiLIFModel':
                    l_seg_names = ['fake_B_5']
                    assert l_seg_names[0] in visuals.keys(), f'Cannot find {l_seg_names[0]} in generated image names ({list(visuals.keys())})'
                    seg_mod_suffix = l_seg_names[0].split('_')[-1]
                    l_seg_names += [x for x in visuals.keys() if x.startswith('fake') and x.split('_')[-1].startswith(seg_mod_suffix) and x != l_seg_names[0]]
                    # print(f'Running postprocess for {len(l_seg_names)} generated images ({l_seg_names})')
        
                    img_name_current = data_val_batch['A_paths'][0].split('/')[-1][:-4] # remove .png
                    metrics_gt = metrics_val[img_name_current]
                    
                    for seg_name in l_seg_names:
                        images = {'Seg':ToPILImage()((visuals[seg_name][0].cpu()+1)/2),
                                  #'Marker':ToPILImage()((visuals['fake_B_4'][0].cpu()+1)/2)
                                  }
                        _, scoring = postprocess(ToPILImage()((data['A'][0]+1)/2), images, opt.scale_size, opt.model)
                        
                        for k,v in scoring.items():
                            if k.startswith('num') or k.startswith('percent'):
                                # to calculate the rmse, here we calculate (x_pred - x_true) ** 2
                                l_metrics_val.append((k+'_'+seg_name,(v - metrics_gt[k])**2))
                    
                if debug and epoch_iter >= debug_data_size:
                    print(f'debug mode, epoch {epoch} stopped at epoch iter {epoch_iter} (>= {debug_data_size})')
                    break
                    
            d_losses_val = {k+'_val':0 for k in losses_val_batch.keys()}
            for k,v in l_losses_val:
                d_losses_val[k+'_val'] += v
            for k in d_losses_val:
                d_losses_val[k] = d_losses_val[k] / len(data_val)
            
            d_metrics_val = {}
            for k,v in l_metrics_val:
                try:
                    d_metrics_val[k] += v
                except:
                    d_metrics_val[k] = v
            for k in d_metrics_val:
                # to calculate the rmse, this is the second part, where d_metrics_val[k] now represents sum((x_pred - x_true) ** 2)
                d_metrics_val[k] = np.sqrt(d_metrics_val[k] / len(data_val))
            
            model.train()
            visualizer.print_current_losses(epoch, epoch_iter, {**losses,**d_losses_val, **d_metrics_val})
            if display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / len(dataset), {**losses,**d_losses_val,**d_metrics_val})


        model.update_learning_rate()

if __name__ == '__main__':
    train()

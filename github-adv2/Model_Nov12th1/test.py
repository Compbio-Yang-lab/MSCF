"""General-purpose test script for image-to-image translation.

It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import time
from Model.options.base_options import TestOptions
from Model.options import read_model_params, Options, print_options
from Model.data import create_dataset
from Model.models import create_model
from Model.util.visualizer import save_images
from Model.util import html
import torch
import click
import subprocess
from Model.metrics.PostProcess_Metrics import postprocess
import shutil


@click.command()
@click.option('--dataroot', required=True, help='reads images from here; expected to have a subfolder')
@click.option('--name', default='.', help='name of the experiment, used as a subfolder under results_dir')
@click.option('--checkpoints_dir', default='checkpoints', help='models are saved here')
@click.option('--gpu_ids', default=[0], type=int, multiple=True)
@click.option('--num_test', default=10000, help='only run test for num_test images')
def test(dataroot, name, checkpoints_dir, gpu_ids, num_test):
    # organized the weights
    results_dir = f'{dataroot}_pred_{name}'
    organize_weights(checkpoints_dir, name)

    # retrieve options used in training setting, similar to cli.py test
    model_dir = os.path.join(checkpoints_dir, f'{name}/latest')

    opt = Options(path_file=os.path.join(model_dir,'train_opt.txt'), mode='test')
    
    # overwrite/supply unseen options using the values from the options provided in the command
    setattr(opt,'checkpoints_dir',model_dir)
    setattr(opt,'dataroot',dataroot)
    setattr(opt,'name','.')
    setattr(opt,'results_dir',results_dir)
    setattr(opt,'num_test',num_test)
        
    if not hasattr(opt,'seg_gen'): # old settings for DeepLIIF models
        opt.seg_gen = True
    
    number_of_gpus_all = torch.cuda.device_count()
    if number_of_gpus_all < len(gpu_ids) and -1 not in gpu_ids:
        number_of_gpus = 0
        gpu_ids = [-1]
        print(f'Specified to use GPU {opt.gpu_ids} for inference, but there are only {number_of_gpus_all} GPU devices. Switched to CPU inference.')

    if len(gpu_ids) > 0 and gpu_ids[0] == -1:
        gpu_ids = []
    elif len(gpu_ids) == 0:
        gpu_ids = list(range(number_of_gpus_all))

    opt.gpu_ids = gpu_ids # overwrite gpu_ids; for test command, default gpu_ids at first is [] which will be translated to a list of all gpus
    
    # hard-code some parameters for test.py
    opt.aspect_ratio = 1.0 # from previous default setting
    opt.display_winsize = 512 # from previous default setting
    opt.use_dp = True # whether to initialize model in DataParallel setting (all models to one gpu, then pytorch controls the usage of specified set of GPUs for inference)
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    print_options(opt)
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    torch.backends.cudnn.benchmark = False
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    model.eval()

    _start_time = time.time()

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()  # save the HTML

    postprocess(base_dir = f"{results_dir}/test_latest/images")

def organize_weights(checkpoints_dir, name):
    folder_a = os.path.join(f"{checkpoints_dir}/{name}", 'latest')
    #print(folder_a)
    os.makedirs(folder_a, exist_ok=True)


    for filename in os.listdir(f"{checkpoints_dir}/{name}"):
        file_path = os.path.join(f"{checkpoints_dir}/{name}", filename)

        if not os.path.isfile(file_path):
            continue  # 跳过子文件夹

        if filename.startswith('latest_net_'):
            shutil.move(file_path, os.path.join(folder_a, filename))

        if filename == 'train_opt.txt':
            shutil.copy2(file_path, os.path.join(folder_a, filename))

    #subprocess.run(['python', 'serialize.py', '--model_dir',
    #                f'{folder_a}'])

    print("organize weights: done")

if __name__ == '__main__':
    test()


    
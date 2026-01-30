import os
import json
import time
import random

import click
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage

from Model.data import create_dataset, transform
from Model.models import init_nets, infer_modalities, infer_results_for_wsi, create_model, postprocess
from Model.util import allowed_file, Visualizer, get_information, test_diff_original_serialized, \
    disable_batchnorm_tracking_stats
from Model.util.util import mkdirs
from Model.options import Options, print_options

import torch.distributed as dist

from packaging import version
import subprocess
import sys

import pickle

def ensure_exists(d):
    if not os.path.exists(d):
        os.makedirs(d)

@click.command()
@click.option('--model_dir', default='./model-server/ParsiLIF_Latest_Model', help='reads models from here')
@click.option('--output_dir', help='saves results here.')
# @click.option('--tile-size', type=int, default=None, help='tile size')
@click.option('--device', default='cpu', type=str,
              help='device to run serialization as well as load model for the similarity test, either cpu or gpu')
@click.option('--epoch', default='latest', type=str, help='epoch to load and serialize')
@click.option('--verbose', default=0, type=int, help='saves results here.')
def serialize(model_dir, output_dir, device, epoch, verbose):
    """Serialize ParsiLIF models using Torchscript
    """
    # if tile_size is None:
    #    tile_size = 512
    output_dir = output_dir or model_dir
    ensure_exists(output_dir)

    # copy train_opt.txt to the target location
    import shutil
    if model_dir != output_dir:
        shutil.copy(f'{model_dir}/train_opt.txt', f'{output_dir}/train_opt.txt')

    # load and update opt for serialization
    opt = Options(path_file=os.path.join(model_dir, 'train_opt.txt'), mode='test')
    opt.epoch = epoch
    if device == 'gpu':
        opt.gpu_ids = [0]  # use gpu 0, in case training was done on larger machines
    else:
        opt.gpu_ids = []  # use cpu

    print_options(opt)
    sample = transform(Image.new('RGB', (opt.scale_size, opt.scale_size)))
    sample = torch.cat([sample] * opt.input_no, 1)

    with click.progressbar(
            init_nets(model_dir, eager_mode=True, opt=opt, phase='test').items(),
            label='Tracing nets',
            item_show_func=lambda n: n[0] if n else n
    ) as bar:
        for name, net in bar:
            # the model should be in eval model so that there won't be randomness in tracking brought by dropout etc. layers
            # https://github.com/pytorch/pytorch/issues/23999#issuecomment-747832122
            net = net.eval()
            net = disable_batchnorm_tracking_stats(net)
            net = net.cpu()
            if name.startswith('GS'):
                traced_net = torch.jit.trace(net, torch.cat([sample, sample, sample], 1))
            else:
                #traced_net = torch.jit.trace(net, sample)
                traced_net = torch.jit.script(net)
            traced_net.save(f'{output_dir}/{name}.pt')

    # test: whether the original and the serialized model produces highly similar predictions
    print('testing similarity between prediction from original vs serialized models...')
    models_original = init_nets(model_dir, eager_mode=True, opt=opt, phase='test')
    models_serialized = init_nets(output_dir, eager_mode=False, opt=opt, phase='test')

    if device == 'gpu':
        sample = sample.cuda()
    else:
        sample = sample.cpu()
    for name in models_serialized.keys():
        print(name, ':')
        model_original = models_original[name].cuda().eval() if device == 'gpu' else models_original[name].cpu().eval()
        model_serialized = models_serialized[name].cuda().eval() if device == 'gpu' else models_serialized[
            name].cpu().eval()
        if name.startswith('GS'):
            test_diff_original_serialized(model_original, model_serialized, torch.cat([sample, sample, sample], 1),
                                          verbose)
        else:
            test_diff_original_serialized(model_original, model_serialized, sample, verbose)
        print('PASS')
class InferenceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.translate(x) # or remove `target` handling in the model


if __name__ == '__main__':
    serialize()
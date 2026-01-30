"""This module contains simple helper functions """
import os
from time import time
from functools import wraps

import torch
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim


def timeit(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        print(f'{f.__name__} {time() - ts}')

        return result

    return wrap


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    x, y, nc = image_numpy.shape
    
    if nc > 3:
        if nc % 3 == 0:
            nc_img = 3
            no_img = nc // nc_img
            
        elif nc % 2 == 0:
            nc_img = 2
            no_img = nc // nc_img
        else:
            nc_img = 1
            no_img = nc // nc_img
        print(f'image (numpy) has {nc}>3 channels, inferred to have {no_img} images each with {nc_img} channel(s)')
        l_image_numpy = np.dsplit(image_numpy,[nc_img*i for i in range(1,no_img)])
        image_numpy = np.concatenate(l_image_numpy, axis=1) # stack horizontally
        
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor_to_pil(t):
    return Image.fromarray(tensor2im(t))


def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=img2.max() - img2.min())


def check_multi_scale(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    max_ssim = (512, 0)
    for tile_size in range(100, 1000, 100):
        image_ssim = 0
        tile_no = 0
        for i in range(0, img2.shape[0], tile_size):
            for j in range(0, img2.shape[1], tile_size):
                if i + tile_size <= img2.shape[0] and j + tile_size <= img2.shape[1]:
                    tile = img2[i: i + tile_size, j: j + tile_size]
                    tile = cv2.resize(tile, (img1.shape[0], img1.shape[1]))
                    tile_ssim = calculate_ssim(img1, tile)
                    image_ssim += tile_ssim
                    tile_no += 1
        if tile_no > 0:
            image_ssim /= tile_no
            if max_ssim[1] < image_ssim:
                max_ssim = (tile_size, image_ssim)
    return max_ssim[0]


def set_seed(seed=0, rank=None):
    """
    seed: basic seed
    rank: rank of the current process, using which to mutate basic seed to have a unique seed per process

    output: a boolean flag indicating whether deterministic training is enabled (True) or not (False)
    """
    os.environ['ParsiLIF_SEED'] = str(seed)

    if seed is not None:
        if rank is not None:
            seed_final = seed + int(rank)
        else:
            seed_final = seed

        os.environ['PYTHONHASHSEED'] = str(seed_final)
        random.seed(seed_final)
        np.random.seed(seed_final)
        torch.manual_seed(seed_final)
        torch.cuda.manual_seed(seed_final)
        torch.cuda.manual_seed_all(seed_final)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        print(f'deterministic training, seed set to {seed_final}')
        return True
    else:
        print(f'not using deterministic training')
        return False

def prepare_training_config(
    dataroot, gpu_ids, model, net_g, net_gs, dataset_mode,
    modalities_no, seg_gen, seg_weights, loss_weights_g, loss_weights_d,
    seed, padding, local_rank=None
):
    """
    Preprocess input arguments and infer configuration settings needed for training.
    Returns a dictionary of parameters to be passed into the main training loop.
    """
    import os
    from PIL import Image

    seg_no = 1 if model == 'ParsiLIF' else 0
    if model != 'ParsiLIF':
        seg_gen = False

    # Local rank and device setup
    local_rank = int(os.getenv('LOCAL_RANK') or -1)
    rank = int(os.getenv('RANK') or -1)
    if gpu_ids and gpu_ids[0] == -1:
        gpu_ids = []

    if gpu_ids:
        torch.cuda.set_device(gpu_ids[local_rank] if local_rank >= 0 else gpu_ids[0])
        gpu_ids = [gpu_ids[local_rank]] if local_rank >= 0 else [gpu_ids[0]]

    if local_rank >= 0:
        dist.init_process_group(backend="nccl", rank=rank, world_size=int(os.getenv('WORLD_SIZE')))
        flag_deterministic = set_seed(seed, local_rank)
    elif rank >= 0:
        flag_deterministic = set_seed(seed, rank)
    else:
        flag_deterministic = set_seed(seed)

    if flag_deterministic:
        padding = 'zero'
        print("Deterministic padding enforced.")

    # Dataset inference
    def get_img_shape(folder):
        img = Image.open(os.path.join(folder, sorted(os.listdir(folder))[0]))
        return img.size

    if dataset_mode == 'unaligned':
        input_no = 1
        pool_size = 50
        scale_size = get_img_shape(os.path.join(dataroot, 'trainA'))[1]
    else:
        img_size = get_img_shape(os.path.join(dataroot, 'train'))
        width, height = img_size
        num_img = width // height
        input_no = num_img - modalities_no - seg_no
        pool_size = 0
        scale_size = height

    # Process network architecture specs
    net_g_list = net_g.split(',')
    if len(net_g_list) == 1:
        net_g_list *= modalities_no

    net_gs_list = net_gs.split(',')
    if model == 'ParsiLIF' and len(net_gs_list) == 1:
        net_gs_list *= (modalities_no + seg_no)
    elif len(net_gs_list) == 1:
        net_gs_list *= seg_no

    # Normalize weights
    def normalize_or_default(w_str, default_len, default_val):
        if not w_str:
            return [default_val] * default_len
        vals = [float(x) for x in w_str.split(',')]
        assert abs(sum(vals) - 1.0) < 1e-5, "weights must sum to 1"
        return vals

    expected_len = modalities_no
    seg_weights = normalize_or_default(seg_weights, expected_len, 1 / expected_len)
    loss_weights_g = normalize_or_default(loss_weights_g, expected_len, 1 / expected_len)
    loss_weights_d = normalize_or_default(loss_weights_d, expected_len, 1 / expected_len)

    return {
        'seg_no': seg_no,
        'seg_gen': seg_gen,
        'gpu_ids': gpu_ids,
        'local_rank': local_rank,
        'padding': padding,
        'input_no': input_no,
        'scale_size': scale_size,
        'pool_size': pool_size,
        'net_g': net_g_list,
        'net_gs': net_gs_list,
        'seg_weights': seg_weights,
        'loss_G_weights': loss_weights_g,
        'loss_D_weights': loss_weights_d,
        'lambda_identity': 0,
    }
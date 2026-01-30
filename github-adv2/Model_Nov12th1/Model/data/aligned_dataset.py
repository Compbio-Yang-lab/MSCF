
from .base_dataset import BaseDataset, get_params, get_transform
import os
from PIL import Image
import numpy as np
import albumentations as A
from torchvision.transforms.functional import to_tensor
import torch
from torchvision import transforms
import torch.utils.data as data
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=None):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images[:max_dataset_size] if max_dataset_size else images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)




class AlignedDataset(BaseDataset):
    def __init__(self, opt, phase='train'):
        BaseDataset.__init__(self, opt.dataroot)
        self.dir_AB = os.path.join(opt.dataroot, phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))

        assert opt.load_size >= opt.crop_size
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.no_flip = opt.no_flip
        self.modalities_no = opt.modalities_no  # number of B images (excl. Seg)
        self.load_size = opt.load_size
        self.crop_size = opt.crop_size
        self.preprocess=opt.preprocess
        # augmentation knobs
        self.augment = True if phase=='train' else False
        self.aug_prob = getattr(opt, 'aug_prob', 0.8)

        self._build_albu()

    def _build_albu(self):
        """Build two-stage pipeline:
           (1) geometry: applied to A, all B, and Seg (as mask)
           (2) color/noise: applied ONLY to A_gen
        """
        if not self.augment:
            self.albu_geom = None
            self.albu_color_A = None
            return

        self.albu_geom = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                    border_mode=4, p=0.5  # constant fill
                ),
                # 如需更强形变可开启（训练不稳可先关）
                # A.ElasticTransform(alpha=40, sigma=6, alpha_affine=8, border_mode=0, p=0.2),
            ],
            is_check_shapes=False
        )

        # 仅作用于 A_gen（颜色/噪声）
        self.albu_color_A = A.Compose(
            [
                A.RandomBrightnessContrast(0.2, 0.2, p=0.35),
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.25),
            ],
            is_check_shapes=False
        )

    @staticmethod
    def _pil_to_np(img_pil):
        arr = np.array(img_pil)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, -1)
        return arr

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size

        # A + modalities_no (B) + 1 (Seg)
        num_img = self.modalities_no + 2
        w2 = w // num_img

        # split slices
        slices_pil = [AB.crop((w2 * i, 0, w2 * (i + 1), h)) for i in range(num_img)]
        A_img = slices_pil[0]
        B_imgs = slices_pil[1:-1]
        Seg_img = slices_pil[-1]

        # numpy
        A_np = self._pil_to_np(A_img)
        B_nps = [self._pil_to_np(b) for b in B_imgs]
        Seg_np = self._pil_to_np(Seg_img)

        # ===== 几何一致：对 A、所有 B、Seg（mask）=====
        if self.albu_geom is not None and np.random.rand() < self.aug_prob:
            payload = {'image': A_np, 'mask': Seg_np}
            extra_targets = {'mask': 'mask'}
            for i in range(len(B_nps)):
                payload[f'im{i}'] = B_nps[i]
                extra_targets[f'im{i}'] = 'image'
            aug = A.Compose(self.albu_geom.transforms,
                            additional_targets=extra_targets,
                            is_check_shapes=False)
            out = aug(**payload)
            A_np = out['image']
            Seg_np = out['mask']
            B_nps = [out[f'im{i}'] for i in range(len(B_nps))]

        # ===== 颜色/噪声：只给 A_gen，加在几何之后 =====
        A_gen_np = A_np.copy()
        if self.albu_color_A is not None and np.random.rand() < self.aug_prob:
            A_gen_np = self.albu_color_A(image=A_gen_np)['image']

        # 回到 PIL
        A_pil_geom_only = Image.fromarray(A_np)
        A_pil_gen = Image.fromarray(A_gen_np)
        B_pils = [Image.fromarray(b) for b in B_nps]
        Seg_pil = Image.fromarray(Seg_np)

        # 同一组 transform_params 确保几何对齐一致
        transform_params = get_params(
            self.preprocess, self.load_size, self.crop_size, A_pil_geom_only.size
        )
        A_transform = get_transform(
            self.preprocess, self.load_size, self.crop_size, self.no_flip, transform_params,
            grayscale=(self.input_nc == 1)
        )

        B_transform = get_transform(
            self.preprocess, self.load_size, self.crop_size, self.no_flip, transform_params,
            grayscale=(self.output_nc == 1)
        )

        A_seg = A_transform(A_pil_geom_only)
        A_gen = A_transform(A_pil_gen)

        B_tensors = [B_transform(b) for b in B_pils]
        Seg_tensor = B_transform(Seg_pil)#to_tensor(Seg_pil)
        B_tensors = B_tensors + [Seg_tensor]

        return {
            'A': A_gen,            # for gan branch
            'A_seg': A_seg,
            'B': B_tensors,
            'A_paths': AB_path,
            'B_paths': AB_path
        }

    def __len__(self):
        return len(self.AB_paths)



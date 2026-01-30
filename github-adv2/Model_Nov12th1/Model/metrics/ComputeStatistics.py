from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import cv2
import numpy as np
import csv
from numba import cuda

from Segmentation_Metrics import Scompute_segmentation_metrics
from pathlib import Path

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage import img_as_float, io, measure
from skimage.color import rgb2gray
import collections
from swd import compute_swd


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--gt_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--model_name', type=str, required=False, default='ParsiLIF')
parser.add_argument('--mode', type=str, default='Segmentation',
                    help='Mode of the statistics computation including Segmentation, ImageSynthesis, All')
parser.add_argument('--raw_segmentation', action='store_true')
parser.add_argument('--device', type=str, default='cuda', help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--batch_size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of processes to use for data loading')
parser.add_argument('--image_types', type=str, default='Hema,DAPI,Lap2,Marker')


class Statistics:
    def __init__(self, args):
        self.gt_path = args.gt_path
        self.model_path = args.model_path
        self.output_path = args.output_path
        self.model_name = args.model_name
        self.mode = args.mode
        self.raw_segmentation = args.raw_segmentation
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = args.device
        self.image_types = args.image_types.replace(' ', '').split(',')

        # Image Similarity Metrics
        self.mse_avg = collections.defaultdict(float)
        self.mse_std = collections.defaultdict(float)

        self.ssim_avg = collections.defaultdict(float)
        self.ssim_std = collections.defaultdict(float)

        self.swd_value = collections.defaultdict(float)

        self.all_info = {}
        self.all_info['Model'] = self.model_name

        # Segmentation Metrics
        self.segmentation_metrics = collections.defaultdict(float)
        self.segmentation_info = None

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)





    def compute_mse_ssim_scores(self):
        def _load_rgb_float(path):
            img = io.imread(path)
            img = img_as_float(img)  # -> float64 in [0,1]
            # 统一成 3 通道
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.ndim == 3 and img.shape[-1] == 4:
                img = img[..., :3]
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
            if not np.isfinite(img).all():
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
            return img
        def _pair_real_fake_in_same_dir(data_dir, image_types=None):
            """
            在 data_dir 递归查找，按 `_real_` <-> `_fake_` 规则配对。
            返回 [(gt_fp, pred_fp, fname), ...]
            - image_types: ['B_1','B_2','B_3','B_4'] 等；若为 None/空，则不过滤
            """
            exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
            data_dir = Path(data_dir).resolve()

            # 归一化类型关键字，便于不区分大小写匹配
            types_lower = None
            if image_types:
                types_lower = [t.lower() for t in image_types]

            pairs = []
            # 仅以包含 `_real_` 的文件作为起点
            for p in data_dir.rglob('*'):
                if not p.is_file():
                    continue
                if p.suffix.lower() not in exts:
                    continue
                name_lower = p.name.lower()
                if '_real_' not in name_lower:
                    continue

                # 类型过滤（例如只要包含 B_1 / B_2 / B_3 / B_4）
                if types_lower is not None and not any(t in name_lower for t in types_lower):
                    continue

                # 构造预测文件名：_real_ -> _fake_
                pred_name = p.name.replace('_real_', '_fake_')
                pred_path = p.with_name(pred_name)

                if not pred_path.exists():
                    stem = pred_path.stem  # 不含扩展名的部分
                    cand = [pred_path.with_suffix(e) for e in exts]
                    found = None
                    for c in cand:
                        # 保持同名匹配：同 stem + 合法后缀
                        if c.with_suffix('').name == stem and c.exists():
                            found = c
                            break
                    if found is not None:
                        pred_path = found

                if pred_path.exists():
                    pairs.append((str(p), str(pred_path), p.name))

            return pairs

        data_dir = Path(self.model_path).resolve().parent  # 同一目录下含有 *_real_* 与 *_fake_*
        print(f"Search & pair in: {data_dir}")

        all_pairs = _pair_real_fake_in_same_dir(data_dir, image_types=['B_1','B_2','B_3','B_4'])
        print(f"Total paired images: {len(all_pairs)}")

        type_to_pairs = {t: [] for t in (['B_1','B_2','B_3','B_4'] or [])}
        if not ['B_1','B_2','B_3','B_4']:
            type_to_pairs = {'ALL': all_pairs}
        else:
            for gt_fp, pred_fp, fname in all_pairs:
                name_lower = fname.lower()
                for t in ['B_1','B_2','B_3','B_4']:
                    if t.lower() in name_lower:
                        type_to_pairs[t].append((gt_fp, pred_fp, fname))
                        break

        for img_type, pairs in type_to_pairs.items():
            print(f"[{img_type}] pairs: {len(pairs)}")

            mse_arr, ssim_arr = [], []

            for gt_fp, pred_fp, _ in pairs:
                orig_img = _load_rgb_float(gt_fp)
                mask_img = _load_rgb_float(pred_fp)

                # 尺寸对齐（若需要你也可以改成 resize）
                if orig_img.shape != mask_img.shape:
                    h = min(orig_img.shape[0], mask_img.shape[0])
                    w = min(orig_img.shape[1], mask_img.shape[1])
                    orig_img = orig_img[:h, :w, :3]
                    mask_img = mask_img[:h, :w, :3]

                # MSE
                mse_val = mean_squared_error(orig_img, mask_img)

                # SSIM（三通道）
                try:
                    ssim_val = ssim(
                        orig_img, mask_img,
                        multichannel=True,
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                        data_range=1.0,channel_axis=-1
                    )


                except TypeError:
                    ssim_val = ssim(
                        orig_img, mask_img,
                        multichannel=True,  # skimage < 0.19
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                        data_range=1.0
                    )

                mse_arr.append(float(mse_val))
                ssim_arr.append(float(ssim_val))

            # 统计
            avg_mse = float(np.mean(mse_arr)) if mse_arr else np.nan
            std_mse = float(np.std(mse_arr)) if mse_arr else np.nan
            avg_ssim = float(np.mean(ssim_arr)) if ssim_arr else np.nan
            std_ssim = float(np.std(ssim_arr)) if ssim_arr else np.nan

            # 写回
            self.mse_avg[img_type] = avg_mse
            self.mse_std[img_type] = std_mse
            self.ssim_avg[img_type] = avg_ssim
            self.ssim_std[img_type] = std_ssim

        print("SSIM average per type:", self.ssim_avg)
        print("MSE average per type:", self.mse_avg)

    def compute_swd(self):
        for img_type in self.image_types:
            orig_images = []
            mask_images = []
            images = os.listdir(self.model_path);print(images)
            for img_name in images:
                if img_type in img_name:
                    orig_img = cv2.cvtColor(cv2.imread(os.path.join(self.gt_path, img_name)), cv2.COLOR_BGR2RGB)
                    mask_img = cv2.cvtColor(cv2.imread(os.path.join(self.model_path, img_name)), cv2.COLOR_BGR2RGB)
                    orig_images.append(orig_img)
                    mask_images.append(mask_img)

            self.swd_value[img_type] = compute_swd(np.array(orig_images), np.array(mask_images), self.device)

    def compute_image_similarity_metrics(self):
        self.compute_mse_ssim_scores()
        print('SSIM Computed')

        #self.compute_swd()
        #print('swd Computed')

        for key in self.mse_avg:
            self.all_info[key + '_' + 'MSE_avg'] = self.mse_avg[key]
            self.all_info[key + '_' + 'MSE_std'] = self.mse_std[key]
            self.all_info[key + '_' + 'ssim_avg'] = self.ssim_avg[key]
            self.all_info[key + '_' + 'ssim_std'] = self.ssim_std[key]
            self.all_info[key + '_' + 'swd_value'] = self.swd_value[key]


    def compute_IHC_scoring(self):
        images = os.listdir(self.gt_path)
        IHC_info = []
        metric_diff_ihc_score = 0
        for img in images:
            gt_image = cv2.cvtColor(cv2.imread(os.path.join(self.gt_path, img)), cv2.COLOR_BGR2RGB)
            if 'ParsiLIF' in self.model_name:
                mask_image = cv2.cvtColor(cv2.imread(os.path.join(self.model_path, img.replace('.png', '_SegRefined.png'))), cv2.COLOR_BGR2RGB)
            else:
                mask_image = cv2.cvtColor(cv2.imread(os.path.join(self.model_path, img)), cv2.COLOR_BGR2RGB)
            gt_image[gt_image < 10] = 0
            label_image_red_gt = measure.label(gt_image[:, :, 0], background=0)
            label_image_blue_gt = measure.label(gt_image[:, :, 2], background=0)
            number_of_positive_cells_gt = (len(np.unique(label_image_red_gt)) - 1)
            number_of_negative_cells_gt = (len(np.unique(label_image_blue_gt)) - 1)
            number_of_all_cells_gt = number_of_positive_cells_gt + number_of_negative_cells_gt
            gt_IHC_score = number_of_positive_cells_gt / number_of_all_cells_gt if number_of_all_cells_gt > 0 else 0

            mask_image[mask_image < 10] = 0
            label_image_red_mask = measure.label(mask_image[:, :, 0], background=0)
            label_image_blue_mask = measure.label(mask_image[:, :, 2], background=0)
            number_of_positive_cells_mask = (len(np.unique(label_image_red_mask)) - 1)
            number_of_negative_cells_mask = (len(np.unique(label_image_blue_mask)) - 1)
            number_of_all_cells_mask = number_of_positive_cells_mask + number_of_negative_cells_mask
            mask_IHC_score = number_of_positive_cells_mask / number_of_all_cells_mask if number_of_all_cells_mask > 0 else 0
            diff = abs(gt_IHC_score * 100 - mask_IHC_score * 100)
            IHC_info.append({'Model': self.model_name, 'Sample': img, 'Diff_IHC_Score': diff})
            metric_diff_ihc_score += diff
        self.write_list_to_csv(IHC_info, IHC_info[0].keys(),
                               filename='IHC_Scoring_info_' + self.mode + '_' + self.model_name + '.csv')
        metric_diff_ihc_score /= len(images)
        print('Diff_IHC_Score:', metric_diff_ihc_score)
        print('-------------------------------------------------------')

    def compute_segmentation_metrics(self):

        thresh = 100
        boundary_thresh = 100
        noise_size = 50
        print(thresh, noise_size)
        self.segmentation_info, self.segmentation_metrics = Scompute_segmentation_metrics(self.gt_path, self.model_path, self.model_name, image_size=512, thresh=thresh, boundary_thresh=boundary_thresh, small_object_size=noise_size, raw_segmentation=self.raw_segmentation)
        self.write_list_to_csv(self.segmentation_info, self.segmentation_info[0].keys(),
                               filename='segmentation_info_' + self.mode + '_' + self.model_name + '_' + str(thresh) + '_' + str(noise_size) + '.csv')
        for key in self.segmentation_metrics:
            self.all_info[key] = self.segmentation_metrics[key]
            print(key, self.all_info[key])
        print('-------------------------------------------------------')






    def create_all_info(self):
        self.write_dict_to_csv(self.all_info, list(self.all_info.keys()), filename='metrics_' + self.mode + '_' + self.model_name + '.csv')

    def compute_statistics(self):
        self.compute_image_similarity_metrics()
        self.compute_segmentation_metrics()
        self.create_all_info()

    def write_dict_to_csv(self, info_dict, csv_columns, filename='info.csv'):
        print('Writing in csv')
        info_csv = open(os.path.join(self.output_path, filename), 'w')
        writer = csv.DictWriter(info_csv, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerow(info_dict)

    def write_list_to_csv(self, info_dict, csv_columns, filename='info.csv'):
        print('Writing in csv')
        info_csv = open(os.path.join(self.output_path, filename), 'w')
        writer = csv.DictWriter(info_csv, fieldnames=csv_columns)
        writer.writeheader()
        for data in info_dict:
            writer.writerow(data)


if __name__ == '__main__':
    args = parser.parse_args()
    stat = Statistics(args)
    print(stat.mode)
    if stat.mode == 'All':
        stat.compute_statistics()
        stat.compute_IHC_scoring()
    elif stat.mode == 'Segmentation':
        stat.compute_segmentation_metrics()
        stat.compute_IHC_scoring()
    elif stat.mode == 'ImageSynthesis':
        stat.compute_image_similarity_metrics()

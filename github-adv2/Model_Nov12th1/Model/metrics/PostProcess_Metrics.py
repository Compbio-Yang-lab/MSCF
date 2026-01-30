import os.path
import sys
import json

import cv2
import numpy as np
import scipy.ndimage as ndi

from Model.util.postprocessing import compute_final_results
import subprocess
from pathlib import Path
from skimage.io import imsave
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import threshold_otsu
def post_process_segmentation_mask0(input_dir, seg_thresh=150, size_thresh='default'):
    images = os.listdir(input_dir)
    image_extensions = ['.png', '.jpg', '.tif', '.tiff']
    if not os.path.exists(input_dir.replace('images','json')): os.makedirs(input_dir.replace('images','json'), exist_ok=True)
    for img in images:
        seg_file = None

        if '_fake_B_5.png' in img:
            orig_file = os.path.join(input_dir, img.replace('_fake_B_5', '_real_A'))
            seg_file = os.path.join(input_dir, img)
            overlaid_file = os.path.join(input_dir, img.replace('_fake_B_5', '_SegOverlaid'))
            refined_file = os.path.join(input_dir, img.replace('_fake_B_5', '_SegRefined'))
            score_file = os.path.join(input_dir.replace('images','json'), img.replace('_fake_B_5.png', '.json'))

        elif '_Seg.png' in img:
            orig_img_ext = None
            for img_ext in image_extensions:
                if os.path.exists(os.path.join(input_dir, img.replace('_Seg.png', img_ext))):
                    orig_img_ext = img_ext
                    break
            orig_file = os.path.join(input_dir, img.replace('_Seg.png', orig_img_ext)) if orig_img_ext is not None else None
            seg_file = os.path.join(input_dir, img)
            overlaid_file = os.path.join(input_dir, img.replace('_Seg', '_SegOverlaid'))
            refined_file = os.path.join(input_dir, img.replace('_Seg', '_SegRefined'))
            score_file = os.path.join(input_dir, img.replace('_Seg.png', '.json'))

        if seg_file is not None:
            if orig_file is not None:
                orig_image = cv2.cvtColor(cv2.imread(orig_file), cv2.COLOR_BGR2RGB)
            else:
                orig_image = cv2.cvtColor(cv2.imread(seg_file), cv2.COLOR_BGR2RGB)
            seg_image = cv2.cvtColor(cv2.imread(seg_file), cv2.COLOR_BGR2RGB)
            overlaid, refined, scoring = compute_final_results(orig_image, seg_image, None, '40x', size_thresh, seg_thresh=seg_thresh)
            if orig_file is not None:
                cv2.imwrite(overlaid_file, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
            cv2.imwrite(refined_file, cv2.cvtColor(refined, cv2.COLOR_RGB2BGR))
            if scoring is not None:
                with open(score_file, 'w') as f:
                    json.dump(scoring, f, indent=2)


def post_process_segmentation_mask0(input_dir, seg_thresh=150, size_thresh='default'):
    images = os.listdir(input_dir)
    image_extensions = ['.png', '.jpg', '.tif', '.tiff']
    json_dir = input_dir.replace('images', 'json')
    if not os.path.exists(json_dir):
        os.makedirs(json_dir, exist_ok=True)

    for img in images:
        seg_file = None

        if '_fake_B_5.png' in img:
            orig_file = os.path.join(input_dir, img.replace('_fake_B_5', '_real_A'))
            seg_file = os.path.join(input_dir, img)
            overlaid_file = os.path.join(input_dir, img.replace('_fake_B_5', '_SegOverlaid'))
            refined_file  = os.path.join(input_dir, img.replace('_fake_B_5', '_SegRefined'))
            score_file    = os.path.join(json_dir, img.replace('_fake_B_5.png', '.json'))

        elif '_Seg.png' in img:
            orig_img_ext = None
            for img_ext in image_extensions:
                path_try = os.path.join(input_dir, img.replace('_Seg.png', img_ext))
                if os.path.exists(path_try):
                    orig_img_ext = img_ext
                    break
            orig_file = os.path.join(input_dir, img.replace('_Seg.png', orig_img_ext)) if orig_img_ext else None
            seg_file  = os.path.join(input_dir, img)
            overlaid_file = os.path.join(input_dir, img.replace('_Seg', '_SegOverlaid'))
            refined_file  = os.path.join(input_dir, img.replace('_Seg', '_SegRefined'))
            score_file    = os.path.join(input_dir, img.replace('_Seg.png', '.json'))

        if seg_file is None:
            continue

        # Read original for overlay (RGB for visualization is fine)
        if orig_file is not None and os.path.exists(orig_file):
            orig_image = cv2.cvtColor(cv2.imread(orig_file), cv2.COLOR_BGR2RGB)
        else:
            # fallback – won’t be used if compute_final_results only needs seg
            orig_image = cv2.cvtColor(cv2.imread(seg_file), cv2.COLOR_BGR2RGB)

        # *** CRITICAL: read seg as grayscale ***
        seg_gray = cv2.imread(seg_file, cv2.IMREAD_GRAYSCALE)

        # If seg was saved as probability (0..255 but soft), binarize it here.
        # seg_thresh defaults to 150; adjust if your probs map is dim.
        _, seg_bin = cv2.threshold(seg_gray, seg_thresh, 255, cv2.THRESH_BINARY)

        # If compute_final_results expects 3-channel, build a mask RGB where
        # the mask is in a single channel; otherwise pass seg_bin directly.
        seg_for_compute = seg_bin  # (H,W) uint8

        overlaid, refined, scoring = compute_final_results(
            orig_image, seg_for_compute, None, '40x', size_thresh, seg_thresh=seg_thresh
        )

        if orig_file is not None:
            cv2.imwrite(overlaid_file, cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
        cv2.imwrite(refined_file, cv2.cvtColor(refined, cv2.COLOR_RGB2BGR))

        if scoring is not None:
            with open(score_file, 'w') as f:
                json.dump(scoring, f, indent=2)




import os, cv2, json
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu


def _to01_gray(x):
    """
    将 seg 或 marker 转为 [0,1] 灰度:
    - 支持 (H,W), (H,W,1), (H,W,3)
    - 支持 float(0..1)、uint8(0..255)、uint16(0..65535)
    - RGB 统一转灰度
    """
    if x is None:
        return None
    x = np.asarray(x)
    if x.ndim == 3:
        if x.shape[2] == 3:
            # 注意：若原图是 BGR，需要先转成 RGB 再灰度；这里假设已经是 RGB
            x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        elif x.shape[2] == 1:
            x = x[..., 0]

    if np.issubdtype(x.dtype, np.floating):
        # 若是 float，判断是否已在 0..1
        vmax = float(np.nanmax(x)) if np.isfinite(x).any() else 1.0
        if vmax > 1.0001:
            # 可能是 0..255 的 float
            x = x / vmax
        # 否则认为已在 0..1
        x = np.clip(x, 0.0, 1.0).astype(np.float32)
    elif x.dtype == np.uint8:
        x = (x.astype(np.float32) / 255.0)
    elif x.dtype == np.uint16:
        x = (x.astype(np.float32) / 65535.0)
    else:
        # 其他整型
        vmax = float(x.max()) if x.size else 1.0
        x = (x.astype(np.float32) / max(vmax, 1.0))
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def _resolve_prob_thresh_01(prob_thresh, pm01):
    """
    将 prob_thresh 解析为 0..1：
    - 数值 >1 视为 0..255 标度，自动 /255
    - 'auto' 用 Otsu；若 Otsu 很低，用 95 分位兜底
    - 'pXX' 用百分位（例如 'p98'）
    """
    if isinstance(prob_thresh, str):
        ps = prob_thresh.strip().lower()
        if ps == 'auto':
            # Otsu on probs; 若失败或过低则用 95 分位
            try:
                t = float(threshold_otsu(pm01))
            except Exception:
                t = float(np.percentile(pm01, 95)) if pm01.size else 0.5
            if t < 0.02:
                t = float(np.percentile(pm01, 95))
            return t
        if ps.startswith('p'):
            q = float(ps[1:])
            return float(np.percentile(pm01, q))
        raise ValueError(f"Unknown prob_thresh string: {prob_thresh}")
    # 数值
    thr = float(prob_thresh)
    if thr > 1.0:
        thr = thr / 255.0
    return np.clip(thr, 0.0, 1.0)

def postprocess2(
    seg_prob,          # (H,W) 概率或 uint8，语义分割概率图
    marker_img,        # (H,W[,3]) 0–1 或 0–255，mpIF marker 或 IHC 灰度/RGB
    prob_thresh=0.5,   # 建议 0.5；若想用 DeepLIIF 的 150，传 150/255.
    size_thresh='auto',
    size_thresh_upper='none',
    marker_thresh='auto',   # 'auto' 或 [0,1] 浮点
    use_quantile=False,     # True 用 90分位替代中位数
):
    # 1) 统一标度并二值化
    pm01 = _to01_gray(seg_prob)
    thr01 = _resolve_prob_thresh_01(prob_thresh, pm01)
    bin_seg = (pm01 >= thr01)

    # 小孔填充（半径太小帮助有限，可适当调大）
    bin_seg = remove_small_holes(bin_seg, area_threshold=16)

    # 2) 初次连通域 & 面积门限(下限)
    inst = label(bin_seg, connectivity=2)
    props = regionprops(inst)
    areas = [p.area for p in props]

    if size_thresh == 'auto':
        min_area = int(np.percentile(areas, 10)) if len(areas) else 0
    else:
        min_area = int(size_thresh) if size_thresh is not None else 0

    if isinstance(size_thresh_upper, str):
        if size_thresh_upper.lower() == 'none':
            max_area = None
        elif size_thresh_upper.lower() == 'auto':
            # 简单取 99% 分位作上限，可按数据分布微调
            max_area = int(np.percentile(areas, 99)) if len(areas) else None
        else:
            raise ValueError(f"Unknown size_thresh_upper: {size_thresh_upper}")
    else:
        max_area = int(size_thresh_upper) if size_thresh_upper is not None else None

    if min_area > 0:
        inst = remove_small_objects(inst, min_size=min_area)
    if max_area is not None:
        keep = np.zeros_like(inst, dtype=np.int32)
        nid = 1
        for p in regionprops(inst):
            if p.area <= max_area:
                keep[inst == p.label] = nid
                nid += 1
        inst = keep

    # 再规范一次 label，保证 id 连续
    inst = label(inst > 0, connectivity=2)

    h, w = inst.shape
    pos_mask = np.zeros((h, w), np.uint8)
    neg_mask = np.zeros((h, w), np.uint8)

    props = regionprops(inst)
    if len(props) == 0:
        return pos_mask, neg_mask, inst, {"cell_count": 0}

    # 3) 没有 marker 的情况：全部记为阴性（或按需要改）
    if marker_img is None:
        neg_mask[inst > 0] = 1
        return pos_mask, neg_mask, inst, {"cell_count": len(props)}

    # 4) 计算每细胞强度（median/quantile），做“实例级 Otsu”
    mk01 = _to01_gray(marker_img)
    per_cell_vals = []
    for p in props:
        m = (inst == p.label)
        if not np.any(m):
            per_cell_vals.append(0.0)
            continue
        vals = mk01[m]
        if use_quantile:
            per_cell_vals.append(float(np.percentile(vals, 90)))  # 更偏“亮点”
        else:
            per_cell_vals.append(float(np.median(vals)))          # 更抗噪

    per_cell_vals = np.asarray(per_cell_vals, dtype=np.float32)

    # 确定阈值
    if marker_thresh == 'auto':
        try:
            t_cell = float(threshold_otsu(per_cell_vals))
        except Exception:
            t_cell = float(np.percentile(per_cell_vals, 75))
    else:
        t_cell = float(marker_thresh)

    # 5) 判阳/阴（逐细胞）
    pos_ids = set(np.nonzero(per_cell_vals >= t_cell)[0] + 1)  # label 从1起
    pos_mask = np.isin(inst, list(pos_ids)).astype(np.uint8)
    neg_mask = ((inst > 0) & (~np.isin(inst, list(pos_ids)))).astype(np.uint8)

    stats = {
        "cell_count": int(len(props)),
        "pos_count":  int((per_cell_vals >= t_cell).sum()),
        "neg_count":  int((per_cell_vals <  t_cell).sum()),
        "marker_thr": float(t_cell),
        #"marker_vals": per_cell_vals,  # 如需可删
    }
    return pos_mask, neg_mask, inst, stats


def save_instance_results(
        seg_prob01: np.ndarray,  # (H,W) 0–1 概率
        marker_rgb01: np.ndarray,  # (H,W,3) 0–1 RGB，可为 None
        pos_mask: np.ndarray,  # (H,W) {0,1}
        neg_mask: np.ndarray,  # (H,W) {0,1}
        inst: np.ndarray,  # (H,W) int 实例ID（1..N，0为背景）
        stats: dict,  # 统计信息（pos/neg数量、阈值等）
        out_dir: Path,  # 根输出目录（Path 或 str）
        idx_global: int,  # 当前样本序号
        img_ori: np.ndarray,
) -> dict:
    out_dir = Path(out_dir)
    #stem = f"{idx_global:05d}"
    save_dir = out_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_save_path = Path(str(save_dir).replace('postprocessed', 'pred'))
    pred_save_path.mkdir(parents=True, exist_ok=True)

    # 1) 伪彩实例图
    inst_rgb = label2rgb(inst,bg_label=0, bg_color=(0, 0, 0))  # float [0,1]
    imsave(save_dir / f"{idx_global}_inst_rgb.png", (inst_rgb*255).astype(np.uint8))


    # 5) overlay（红色边界叠加在 marker）
    if marker_rgb01 is not None:
        marker_rgb01 = np.clip(marker_rgb01, 0, 1)
        overlay = (marker_rgb01).astype(np.uint8).copy()
        edges = find_boundaries(inst, mode="inner")
        overlay[edges] = [255, 0, 0]
        cv2.imwrite(str(save_dir / f"{idx_global}_overlay_pred.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


    img_ori = np.clip(img_ori, 0, 256)
    base = (img_ori).astype(np.uint8).copy()
    overlay = base.copy()
    overlay[pos_mask.astype(bool)] = [255, 0, 0]  # 红
    overlay[neg_mask.astype(bool)] = [0, 0, 255]  # 蓝
    edges = find_boundaries(inst, mode="inner")
    overlay[edges] = [0, 255, 0]
    blended = cv2.addWeighted(base, 0.6, overlay, 0.4, 0)
    cv2.imwrite(str(save_dir / f"{idx_global}_overlay_ori.png"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    img_save = np.clip(img_ori, 0, 255).astype(np.uint8)
    cv2.imwrite(str(save_dir / f"{idx_global}_img_ori.png"), cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))

    # 7) 阳/阴可视化组合图 (红=阳, 蓝=阴, 绿=边界)
    comp = np.zeros_like(overlay, dtype=np.uint8)
    comp[pos_mask.astype(bool)] = [255, 0, 0]  # 红色阳性
    comp[neg_mask.astype(bool)] = [0, 0, 255]  # 蓝色阴性
    comp[edges] = [0, 255, 0]  # 绿色边界
    cv2.imwrite(str(save_dir / f"{idx_global}_composite.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(pred_save_path/ f"{idx_global}_SegRefined.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

    # 6) 统计信息
    with open(save_dir / f"{idx_global}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    paths = {
        "seg_prob": save_dir / f"{idx_global}_seg_prob.png",
        "inst_label": save_dir / f"{idx_global}_inst_label.tif",
        "inst_rgb": save_dir / f"{idx_global}_inst_rgb.png",
        "pos_mask": save_dir / f"{idx_global}_pos_mask.png",
        "neg_mask": save_dir / f"{idx_global}_neg_mask.png",
        "overlay": (save_dir / f"{idx_global}_overlay.png") if marker_rgb01 is not None else None,
        "marker_rgb": (save_dir / f"{idx_global}_marker_rgb.png") if marker_rgb01 is not None else None,
        "stats": save_dir / f"{idx_global}_stats.json",
    }
    return {"dir": save_dir, "paths": paths}


# ======== 主函数：遍历并调用 postprocess2 ========
def post_process_segmentation_mask(result_dir, prob_thresh=0.5):
    files = [f for f in os.listdir(result_dir) if f.endswith('_fake_B_5.png')]
    save_dir = os.path.join(result_dir, 'postprocessed')
    os.makedirs(save_dir, exist_ok=True)

    for f in files:
        seg_path = os.path.join(result_dir, f)
        base = f.replace('_fake_B_5.png', '')
        marker_path = os.path.join(result_dir, base + '_fake_B_4.png')  # Ki67 marker
        if not os.path.exists(marker_path):
            print(f"⚠️ No marker found for {f}, skipping Ki67 analysis.")
            marker_img = None
        else:
            marker_img = cv2.cvtColor(cv2.imread(marker_path), cv2.COLOR_BGR2RGB)

        seg_prob = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        img_ori = cv2.imread(os.path.join(result_dir, base + '_real_A.png'))
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        pos_mask, neg_mask, inst, stats = postprocess2(
            seg_prob,
            marker_img,
            prob_thresh=prob_thresh,
            size_thresh='auto',
            marker_thresh='auto'
        )

        ret = save_instance_results(
            seg_prob01=seg_prob,
            marker_rgb01=marker_img,
            pos_mask=pos_mask,
            neg_mask=neg_mask,
            inst=inst,
            stats=stats,
            out_dir=save_dir,
            idx_global=base,
            img_ori=img_ori,
        )

        # 保存结果
        cv2.imwrite(os.path.join(save_dir, base + '_pos.png'),  pos_mask * 255)
        cv2.imwrite(os.path.join(save_dir, base + '_neg.png'),  neg_mask * 255)
        cv2.imwrite(os.path.join(save_dir, base + '_inst.png'), (inst > 0).astype(np.uint8) * 255)
        with open(os.path.join(save_dir, base + '_stats.json'), 'w') as fjson:
            json.dump(stats, fjson, indent=2)

        print(f"[✓] {base}: {stats['cell_count']} cells, {stats['pos_count']} positive")





import os
import shutil

def organize_images(folder_path):
    folder_a = os.path.join(folder_path, 'pred')
    folder_b = os.path.join(folder_path, 'gt')
    os.makedirs(folder_a, exist_ok=True)
    os.makedirs(folder_b, exist_ok=True)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue  # 跳过子文件夹

        # 分类到 a/
        if filename.endswith('_SegRefined.png'):
            shutil.move(file_path, os.path.join(folder_a, filename))

        # 分类并重命名到 b/
        elif filename.endswith('_overlay.png'):
            new_name = filename.replace('_overlay', '')  # 删除 "_real_B_"
            new_path = os.path.join(folder_b, new_name)
            shutil.move(file_path, new_path)

    print("✅ 文件分类完成。")


def postprocess(base_dir = 'testdata'):

    post_process_segmentation_mask(base_dir)
    organize_images(base_dir)
    subprocess.run(['python', 'Model/metrics/ComputeStatistics.py', '--gt_path',
                    f'{base_dir}/gt', '--model_path', f'{base_dir}/pred',
                    '--output_path', f'{base_dir}/metrics', '--mode', 'All'])

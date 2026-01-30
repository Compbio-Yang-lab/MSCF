import collections

import numpy as np
import cv2
import os
from numba import jit
from skimage import measure
import time
from SegmentationMask import positive_negative_masks
import torch
import collections


@jit(nopython=True)
def compute_metrics_gpu(mask_img, gt_img, image_size):
    eps = 1e-7

    p = (mask_img > 0).ravel()
    g = (gt_img   > 0).ravel()

    tp = np.sum(p & g)
    fp = np.sum(p & (~g))
    fn = np.sum((~p) & g)
    tn = np.sum((~p) & (~g))

    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)
    tn_f = float(tn)

    iou  = tp_f / (tp_f + fp_f + fn_f + eps)
    dice = (2.0 * tp_f) / (2.0 * tp_f + fp_f + fn_f + eps)
    acc  = (tp_f + tn_f) / (tp_f + tn_f + fp_f + fn_f + eps)
    precision = tp_f / (tp_f + fp_f + eps)
    recall    = tp_f / (tp_f + fn_f + eps)
    f1        = (2.0 * precision * recall) / (precision + recall + eps)

    return iou, precision, recall, f1, dice, acc




def compute_metrics(mask_img, gt_img):
    smooth = 0.0001
    intesection_TP = np.logical_and(gt_img, mask_img)
    intesection_FN = np.logical_and(gt_img, 1 - mask_img)
    intesection_FP = np.logical_and(1 - gt_img, mask_img)
    intesection_TN = np.logical_and(1 - gt_img, 1 - mask_img)
    union = np.logical_or(gt_img, mask_img)

    iou_score = (np.sum(intesection_TP) + smooth) / (np.sum(union) + smooth)
    precision_score = (np.sum(intesection_TP) + smooth) / (np.sum(intesection_TP) + np.sum(intesection_FP) + smooth)
    recall_score = (np.sum(intesection_TP) + smooth) / (np.sum(intesection_TP) + np.sum(intesection_FN) + smooth)
    f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    dice_score = (2 * np.sum(intesection_TP) + smooth) / (2 * np.sum(intesection_TP) + np.sum(intesection_FN) + np.sum(intesection_FP) + smooth)
    pix_acc_score = (np.sum(intesection_TP) + np.sum(intesection_TN) + smooth) / (np.sum(intesection_TP) + np.sum(intesection_TN) + np.sum(intesection_FN) + np.sum(intesection_FP) + smooth)
    return iou_score, precision_score, recall_score, f1_score, dice_score, pix_acc_score


def compute_jaccard_index(set_1, set_2):
    n = len(set_1.intersection(set_2))
    return n / float(len(set_1) + len(set_2) - n)


def compute_aji(gt_image, mask_image):
    label_image_gt = measure.label(gt_image, background=0)
    label_image_mask = measure.label(mask_image, background=0)
    gt_labels = np.unique(label_image_gt)
    mask_labels = np.unique(label_image_mask)
    mask_components = []
    mask_marked = []
    for mask_label in mask_labels:
        if mask_label == 0:
            continue
        comp = np.zeros((gt_image.shape[0], gt_image.shape[1]), dtype=np.uint8)
        comp[label_image_mask == mask_label] = 1
        mask_components.append(comp)
        mask_marked.append(False)

    total_intersection = 0
    total_union = 0
    total_U = 0
    for gt_label in gt_labels:
        if gt_label == 0:
            continue
        comp = np.zeros((gt_image.shape[0], gt_image.shape[1]), dtype=np.uint8)
        comp[label_image_gt == gt_label] = 1
        intersection = [0, 0, 0]    # index, intersection, union
        for i in range(len(mask_components)):
            if not mask_marked[i]:
                comp_intersection = np.sum(np.logical_and(comp, mask_components[i]))
                if comp_intersection > intersection[1]:
                    union = np.sum(np.logical_or(comp, mask_components[i]))
                    intersection = [i, comp_intersection, union]
        if intersection[1] > 0:
            mask_marked[intersection[0]] = True
            total_intersection += intersection[1]
            total_union += intersection[2]
    for i in range(len(mask_marked)):
        if not mask_marked[i]:
            total_U += np.sum(mask_components[i])
    aji = total_intersection / (total_union + total_U) if (total_union + total_U) > 0 else 0
    return aji

def Scompute_segmentation_metrics(
    gt_dir, model_dir, model_name,
    image_size=512, thresh=100, boundary_thresh=100,
    small_object_size=20, raw_segmentation=True
):
    info_dict = []
    metrics = collections.defaultdict(float)
    images = os.listdir(model_dir)
    counter = 0

    postfix = '_SegRefined.png'

    for mask_name in images:
        if postfix not in mask_name:
            continue
        counter += 1

        mask_image = cv2.cvtColor(
            cv2.imread(os.path.join(model_dir, mask_name)),
            cv2.COLOR_BGR2RGB
        )
        mask_image = cv2.resize(mask_image, (image_size, image_size))

        if not raw_segmentation:
            positive_mask = mask_image[:, :, 0]
            negative_mask = mask_image[:, :, 2]
        else:
            positive_mask, negative_mask = positive_negative_masks(
                mask_image, thresh, boundary_thresh, small_object_size
            )

        positive_mask = (positive_mask > 0).astype(np.uint8)
        negative_mask = (negative_mask > 0).astype(np.uint8)

        gt_img = cv2.cvtColor(
            cv2.imread(os.path.join(gt_dir, mask_name.replace(postfix, '.png'))),
            cv2.COLOR_BGR2RGB
        )
        gt_img = cv2.resize(gt_img, (image_size, image_size))

        positive_gt = (gt_img[:, :, 0] > 0).astype(np.uint8)
        negative_gt = (gt_img[:, :, 2] > 0).astype(np.uint8)

        AJI_positive = compute_aji(positive_gt, positive_mask)   # 原函数
        AJI_negative = compute_aji(negative_gt, negative_mask)   # 原函数
        AJI_mean = (AJI_positive + AJI_negative) / 2.0

        pred_mask = ((positive_mask > 0) | (negative_mask > 0)).astype(np.uint8)
        gt_mask   = ((positive_gt   > 0) | (negative_gt   > 0)).astype(np.uint8)

        IOU, precision, recall, f1, Dice, pixAcc = compute_metrics_gpu(
            pred_mask, gt_mask, gt_img.shape
        )

        info_dict.append({
            'Model': model_name,
            'image_name': mask_name,
            'cell_type': 'All',
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'Dice': Dice * 100,
            'IOU': IOU * 100,
            'PixAcc': pixAcc * 100,
            'AJI': AJI_mean * 100,
            'AJI_positive': AJI_positive * 100,
            'AJI_negative': AJI_negative * 100,
        })

        # 累计到整体均值
        metrics['precision'] += precision * 100
        metrics['recall']    += recall * 100
        metrics['f1']        += f1 * 100
        metrics['Dice']      += Dice * 100
        metrics['IOU']       += IOU * 100
        metrics['PixAcc']    += pixAcc * 100

        metrics['AJI']           += AJI_mean * 100
        metrics['AJI_positive']  += AJI_positive * 100
        metrics['AJI_negative']  += AJI_negative * 100

    # 求平均；防止空目录
    if counter > 0:
        for k in metrics:
            metrics[k] /= counter
    else:
        metrics = {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'Dice': 0.0, 'IOU': 0.0, 'PixAcc': 0.0,
            'AJI': 0.0, 'AJI_positive': 0.0, 'AJI_negative': 0.0
        }

    return info_dict, metrics

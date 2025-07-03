# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
import matplotlib.pyplot as plt

# Example bounding boxes: each box is defined by (x_center, y_center, width, height)
bounding_boxes = [
    (50, 50, 10, 20),
    (150, 100, 30, 40),
    (200, 200, 20, 30),
    (300, 250, 50, 60)
]

# Create a blank image with scatter plot
fig, ax = plt.subplots()
for box in bounding_boxes:
    x_center, y_center, width, height = box
    # Calculate the size of the dot based on the area of the bounding box
    dot_size = width * height
    ax.scatter(x_center, y_center, s=dot_size)

# Set plot limits and aspect
ax.set_xlim(0, 400)
ax.set_ylim(0, 300)
ax.set_aspect('equal')

# Remove axis for cleaner presentation
ax.axis('off')

plt.show()
from .two_stage import TwoStageDetector
import torch
import copy
from torch import Tensor
from mmdet.structures import SampleList
from .grid_generator import make_grid_generator
import torch.nn.functional as F
from .saliency_loss import saliency_loss
import pandas as pd
from collections import defaultdict
from .aspect_ratio_loss import aspect_ratio_loss
from .foreground_align_loss import offset_flip_loss
from .foreground_align_uniform_loss import foreground_align_uniform_loss
import cv2
import numpy as np
from torchvision import transforms
# from docx import Document
import time

from mmengine.dist import is_main_process
from mmengine.structures import InstanceData

from copy import deepcopy
def time_synchronized(t1=None, m=None):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.time()
    if m is not None and t1 is not None:
        print('timing {} is {}'.format(m, t2 - t1))
    return t2

def unwarp_bboxes(bboxes, img_shape, grid, ori_shape):
    """Unwarps a tensor of bboxes of shape (n, 4) or (n, 5) according to the grid \
    of shape (h, w, 2) used to warp the corresponding image and the \
    output_shape (H, W, ...)."""
    bboxes = bboxes.clone()
    # image map of unwarped (x,y) coordinates
    img = grid.permute(2, 0, 1).unsqueeze(0)

    warped_height, warped_width = img_shape[0], img_shape[1]
    xgrid = 2 * (bboxes[:, 0:4:2] / warped_width) - 1
    ygrid = 2 * (bboxes[:, 1:4:2] / warped_height) - 1
    # xgrid = 2 * (bboxes[:, 0:4:2] / (warped_width - 1)) - 1
    # ygrid = 2 * (bboxes[:, 1:4:2] / (warped_height - 1)) - 1
    grid = torch.stack((xgrid, ygrid), dim=2).unsqueeze(0)

    # warped_bboxes has shape (2, num_bboxes, 2)
    warped_bboxes = F.grid_sample(
        img, grid, align_corners=False, padding_mode='border').squeeze(0)
    # bboxes[:, 0:4:2] = torch.round((warped_bboxes[0] + 1) / 2 * (ori_shape[1] - 1))
    # bboxes[:, 1:4:2] = torch.round((warped_bboxes[1] + 1) / 2 * (ori_shape[0] - 1))
    bboxes[:, 0:4:2] = torch.round((warped_bboxes[0] + 1) / 2 * ori_shape[1])
    bboxes[:, 1:4:2] = torch.round((warped_bboxes[1] + 1) / 2 * ori_shape[0])
    # bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * ori_shape[1]
    # bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * ori_shape[0]
    # bboxes[:, 0:4:2] = (warped_bboxes[0] + 1) / 2 * (ori_shape[1] - 1)
    # bboxes[:, 1:4:2] = (warped_bboxes[1] + 1) / 2 * (ori_shape[0] - 1)

    return bboxes

def warp_bboxes(batch_gt_bboxes, interpolators, ori_shape, filter_invalid=False):
    """Warps a tensor of gt bboxes to the resample image"""
    device = batch_gt_bboxes[0].device

    transformed_gt_bboxes = []
    keeps = []
    for i in range(len(batch_gt_bboxes)):
        gt_xyxy = copy.deepcopy(batch_gt_bboxes[i])
        # gt_xyxy[:, 0] /= ori_shape[1] - 1
        # gt_xyxy[:, 2] /= ori_shape[1] - 1
        # gt_xyxy[:, 1] /= ori_shape[0] - 1
        # gt_xyxy[:, 3] /= ori_shape[0] - 1
        # gt_xyxy = torch.clamp(gt_xyxy,min=0.,max=1.)
        gt_xyxy[:, 0] /= ori_shape[1]
        gt_xyxy[:, 2] /= ori_shape[1]
        gt_xyxy[:, 1] /= ori_shape[0]
        gt_xyxy[:, 3] /= ori_shape[0]


        # box transform
        y1new = torch.cat((torch.tensor([i], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                           torch.ones(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                           gt_xyxy[:, :2]), dim=-1)
        x1new = torch.cat((torch.tensor([i], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                           torch.zeros(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                           gt_xyxy[:, :2]), dim=-1)
        y2new = torch.cat((torch.tensor([i], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                           torch.ones(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                           gt_xyxy[:, 2:]), dim=-1)
        x2new = torch.cat((torch.tensor([i], dtype=torch.int).unsqueeze(0).expand(gt_xyxy.shape[0], 1).to(device),
                           torch.zeros(gt_xyxy.shape[0]).unsqueeze(-1).to(device),
                           gt_xyxy[:, 2:]), dim=-1)

        gt_sampled_xyxy = torch.stack([interpolators(x1new).to(device),
                                       interpolators(y1new).to(device),
                                       interpolators(x2new).to(device),
                                       interpolators(y2new).to(device)], dim=-1)

        # x_equal_mask = gt_sampled_xyxy[:, 0] == gt_sampled_xyxy[:, 2]
        # y_equal_mask = gt_sampled_xyxy[:, 1] == gt_sampled_xyxy[:, 3]

        # gt_sampled_xyxy[x_equal_mask, 2] += 1
        # gt_sampled_xyxy[y_equal_mask, 3] += 1

        if filter_invalid:
            x_larger_mask = gt_sampled_xyxy[:, 0] >= gt_sampled_xyxy[:, 2]
            y_larger_mask = gt_sampled_xyxy[:, 1] >= gt_sampled_xyxy[:, 3]

            keep = ~(x_larger_mask | y_larger_mask)
            gt_sampled_xyxy = gt_sampled_xyxy[keep]

            keeps.append(keep)

            if x_larger_mask.any() or y_larger_mask.any():
                print('except y x larger or equal mask, {} box removed'.format((~keep).sum()))

        # x_larger_mask = gt_sampled_xyxy[:, 0] > gt_sampled_xyxy[:, 2]
        # y_larger_mask = gt_sampled_xyxy[:, 1] > gt_sampled_xyxy[:, 3]
        #
        # if x_larger_mask.any() or y_larger_mask.any():
        #     print('except y x larger mask')


        transformed_gt_bboxes.append(gt_sampled_xyxy)
    if filter_invalid:
        return transformed_gt_bboxes, keeps
    else:
        return transformed_gt_bboxes

def compute_zoom_in_multiple(batch_gt_bboxes, interpolators, ori_shapes, test_cfg, min_ratio):
    warp_batch_gt_bboxes = warp_bboxes(batch_gt_bboxes, interpolators, ori_shapes)

    h = batch_gt_bboxes[0][:, 2] - batch_gt_bboxes[0][:, 0]
    w = batch_gt_bboxes[0][:, 3] - batch_gt_bboxes[0][:, 1]
    area_before = w * h

    small_indice = area_before < 1024
    medium_indice = (area_before > 1024) & (area_before < 9216)
    large_indice = area_before > 9216

    area_before_small = area_before[small_indice]
    area_before_medium = area_before[medium_indice]
    area_before_large = area_before[large_indice]

    # h1 = warp_batch_gt_bboxes[0][:, 2]/ min_ratio - warp_batch_gt_bboxes[0][:, 0]/ min_ratio
    # w1 = warp_batch_gt_bboxes[0][:, 3]/ min_ratio - warp_batch_gt_bboxes[0][:, 1]/ min_ratio
    h1 = warp_batch_gt_bboxes[0][:, 2] - warp_batch_gt_bboxes[0][:, 0]
    w1 = warp_batch_gt_bboxes[0][:, 3] - warp_batch_gt_bboxes[0][:, 1]
    area_after = w1 * h1

    area_after_small = area_after[small_indice]
    area_after_medium = area_after[medium_indice]
    area_after_large = area_after[large_indice]
    max_mutliple = float(torch.max(area_after / area_before).cpu().item())

    test_cfg['multiple_sum']['sum'] += float(torch.sum(area_after / area_before).cpu().item())
    test_cfg['multiple_sum']['small_sum'] += float(torch.sum(area_after_small / area_before_small).cpu().item())
    test_cfg['multiple_sum']['medium_sum'] += float(torch.sum(area_after_medium / area_before_medium).cpu().item())
    test_cfg['multiple_sum']['large_sum'] += float(torch.sum(area_after_large / area_before_large).cpu().item())
    if max_mutliple > test_cfg['multiple_sum']['max_multiple']:
        test_cfg['multiple_sum']['max_multiple'] = max_mutliple

    test_cfg['box_num_sum']['image_sum'] += 1
    test_cfg['box_num_sum']['sum'] += batch_gt_bboxes[0].shape[0]
    test_cfg['box_num_sum']['small_sum'] += area_before_small.shape[0]
    test_cfg['box_num_sum']['medium_sum'] += area_before_medium.shape[0]
    test_cfg['box_num_sum']['large_sum'] += area_before_large.shape[0]
    test_cfg['box_num_sum']['multiple_more_than_2'] += torch.sum((area_after / area_before) > 2).item()

    # if test_cfg['box_num_sum']['image_sum'] >= test_cfg['valdata_total'] and is_main_process():
    #     print("global multiple is:", test_cfg['multiple_sum']['sum'] / (test_cfg['box_num_sum']['sum']+1e-5))
    #     print("small multiple is:", test_cfg['multiple_sum']['small_sum'] / (test_cfg['box_num_sum']['small_sum']+1e-5))
    #     print("medium multiple is:", test_cfg['multiple_sum']['medium_sum'] / (test_cfg['box_num_sum']['medium_sum']+1e-5))
    #     print("large multiple is:", test_cfg['multiple_sum']['large_sum'] / (test_cfg['box_num_sum']['large_sum']+1e-5))
    #     print("max multiple is:", test_cfg['multiple_sum']['max_multiple'])
    #     print("more than 2 multiple sum:", test_cfg['box_num_sum']['multiple_more_than_2'])

def calculate_iou(box1, box2):
    x1_inter = torch.max(box1[:, 0], box2[:, 0])
    y1_inter = torch.max(box1[:, 1], box2[:, 1])
    x2_inter = torch.min(box1[:, 2], box2[:, 2])
    y2_inter = torch.min(box1[:, 3], box2[:, 3])

    intersection_area = torch.clamp(x2_inter - x1_inter + 1, min=0) * torch.clamp(y2_inter - y1_inter + 1, min=0)

    area_box1 = (box1[:, 2] - box1[:, 0] + 1) * (box1[:, 3] - box1[:, 1] + 1)
    area_box2 = (box2[:, 2] - box2[:, 0] + 1) * (box2[:, 3] - box2[:, 1] + 1)

    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area
    return torch.sum(iou), box1.shape[0]

def calculate_iou_for_2_box(box1, box2):
    """
    计算两个框之间的交并比（IoU）。

    参数：
    box1：第一个框，形状为 (x1, y1, x2, y2)
    box2：第二个框，形状为 (x1, y1, x2, y2)

    返回值：
    iou：交并比
    """
    # 获取相交部分的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算相交部分的面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算并集部分的面积
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection

    # 计算交并比
    iou = intersection / union if union > 0 else 0

    return iou

def calculate_ap(pred_boxes, gt_boxes, cls_id, doc, threshold=0.5):
    # pred_boxes: 预测框的 tensor，形状为 (n, 4)
    # gt_boxes: 真实框的 tensor，形状为 (m, 4)
    # threshold: 用于确定预测框与真实框匹配的 IoU 阈值

    if gt_boxes.ndim == 1:
        gt_boxes = gt_boxes.unsqueeze(0)

    if pred_boxes.ndim == 1:
        pred_boxes = pred_boxes.unsqueeze(0)

    n = pred_boxes.shape[0]
    m = gt_boxes.shape[0]

    # 初始化 True Positive (TP) 和 False Positive (FP) 的计数器
    tp = torch.zeros(n)
    fp = torch.ones(n)
    gt_is_used = torch.ones(m)

    # 根据 IoU 进行预测框与真实框的匹配
    for i in range(n):
        for j in range(m):
            iou = calculate_iou_for_2_box(pred_boxes[i], gt_boxes[j])
            if iou >= threshold:
                if gt_is_used[j]:
                    tp[i] = 1
                    fp[i] = 0
                    gt_is_used[j] = 0

    # 计算 Precision 和 Recall
    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)
    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / m

    # 计算 Precision-Recall 曲线下的面积
    # 使用插值计算
    ap = torch.tensor([0.]).to('cuda')
    for i in range(n):
        if i == 0:
            ap += precision[i] * recall[i]
        else:
            ap += precision[i] * (recall[i] - recall[i - 1])

    doc.add_paragraph(f'cls_id:{cls_id},precision:{precision.tolist()}, recall:{recall.tolist()}, ap:{ap}')

    return ap, doc

def compute_IOU(transformed_gt_bboxes, unwarp_gt_bboxes, test_cfg):
    h = transformed_gt_bboxes[:, 2] - transformed_gt_bboxes[:, 0]
    w = transformed_gt_bboxes[:, 3] - transformed_gt_bboxes[:, 1]
    area_before = w * h

    small_indice = area_before < 1024
    medium_indice = (area_before > 1024) & (area_before < 9216)
    large_indice = area_before > 9216

    iou_small_sum, small_sum = calculate_iou(transformed_gt_bboxes[small_indice], unwarp_gt_bboxes[small_indice])
    iou_medium_sum, medium_sum = calculate_iou(transformed_gt_bboxes[medium_indice], unwarp_gt_bboxes[medium_indice])
    iou_large_sum, large_sum = calculate_iou(transformed_gt_bboxes[large_indice], unwarp_gt_bboxes[large_indice])
    iou_sum, _ = calculate_iou(transformed_gt_bboxes, unwarp_gt_bboxes)

    test_cfg['iou_sum']['small_sum'] += iou_small_sum.cpu().item()
    test_cfg['iou_sum']['medium_sum'] += iou_medium_sum.cpu().item()
    test_cfg['iou_sum']['large_sum'] += iou_large_sum.cpu().item()
    test_cfg['iou_sum']['sum'] += iou_sum.cpu().item()

    test_cfg['iou_box_num_sum']['image_sum']+=1
    test_cfg['iou_box_num_sum']['small_sum'] += small_sum
    test_cfg['iou_box_num_sum']['medium_sum'] += medium_sum
    test_cfg['iou_box_num_sum']['large_sum'] += large_sum
    test_cfg['iou_box_num_sum']['sum'] += transformed_gt_bboxes.shape[0]

    # if test_cfg['iou_box_num_sum']['image_sum'] >= test_cfg['valdata_total'] and is_main_process():
    #     print('sum', test_cfg['iou_box_num_sum']['sum'])
    #     print("mean IOU is:", test_cfg['iou_sum']['sum'] / (test_cfg['iou_box_num_sum']['sum'] + 1e-5))
    #     print("small mean IOU is:", test_cfg['iou_sum']['small_sum'] / (test_cfg['iou_box_num_sum']['small_sum']+1e-5))
    #     print("medium mean IOU is:", test_cfg['iou_sum']['medium_sum'] / (test_cfg['iou_box_num_sum']['medium_sum']+1e-5))
    #     print("large mean IOU is:", test_cfg['iou_sum']['large_sum'] / (test_cfg['iou_box_num_sum']['large_sum']+1e-5))

@MODELS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)


@MODELS.register_module()
class RezoomedFasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.grid_generator = make_grid_generator(sep_fwhm=self.train_cfg['sep_fwhm']['fwhm'],
                                                  nonsep_fwhm=self.train_cfg['nonsep_fwhm']['fwhm'])

    def loss(self, batch_inputs_orig: Tensor,
             batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        batch_input_shape = batch_data_samples[0].batch_input_shape

        detach_switch = self.train_cfg['detach_detector_loss']['switch']

        batch_inputs = F.interpolate(batch_inputs_orig, size=batch_input_shape, mode='bilinear', align_corners=True)
        grid, uniform_grid, interpolators, saliency_list = self.grid_generator(batch_inputs, batch_input_shape)

        batch_sampled_inputs = F.grid_sample(batch_inputs_orig, grid, padding_mode='reflection')
        # batch_gt_bboxes = [copy.deepcopy(data_sample.gt_instances.bboxes) for data_sample in batch_data_samples]
        batch_gt_bboxes_orig = [copy.deepcopy(data_sample.gt_instances_orig_maybelarger_due2batch.bboxes) for data_sample in batch_data_samples]
        # batch_gt_bboxes_orig = [copy.deepcopy(data_sample.ori_gt_bboxes.tensor) for data_sample in batch_data_samples]


        # if len(batch_gt_bboxes[0])>0:
        #     batch_transformed_gt_bboxes = warp_bboxes(batch_gt_bboxes, interpolators, batch_input_shape)
        #     unwarp_transformed_bboxes = unwarp_bboxes(batch_transformed_gt_bboxes[0], batch_input_shape, grid[0], batch_input_shape)
        #     # unwarp_transformed_bboxes = batch_transformed_gt_bboxes[0]
        #     # compute_IOU(batch_gt_bboxes[0], unwarp_transformed_bboxes, self.test_cfg)
        #     # if (unwarp_transformed_bboxes - batch_transformed_gt_bboxes[0]).sum() != 0:
        #     #     print('xx')
        #     compute_IOU(batch_gt_bboxes[0], unwarp_transformed_bboxes, self.test_cfg)
        # else:
        #     self.test_cfg['iou_box_num_sum']['image_sum'] += 1

        def generate_clusters_for_bounding_boxes(bounding_boxes, img, bandwidth=0.1):
            import os
            import numpy as np
            import cv2
            import matplotlib.pyplot as plt

            data_pts = np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in bounding_boxes],
                                dtype=np.float32)

            if len(data_pts) == 0:
                print("No bounding boxes found.")
                return []

            K = min(len(data_pts), 10)  # Choose K as the minimum of the number of bounding boxes and 10

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001)
            compactness, labels, centers = cv2.kmeans(data_pts, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            unique_labels = np.unique(labels)
            clusters = []
            cluster_boxes = []  # Store the cluster boxes
            for label in unique_labels:
                cluster_pts = data_pts[labels.flatten() == label]
                centroid = np.mean(cluster_pts, axis=0)
                clusters.append((centroid[0], centroid[1]))

                boxes_cluster = bounding_boxes[labels.flatten() == label]
                min_x = boxes_cluster[:, 0].min()
                min_y = boxes_cluster[:, 1].min()
                max_x = boxes_cluster[:, 2].max()
                max_y = boxes_cluster[:, 3].max()
                # Calculate cluster box
                cluster_box = torch.stack([min_x, min_y, max_x, max_y])
                cluster_boxes.append(cluster_box)

            plt.figure(figsize=(10, 8))
            img = img.cpu().permute(1,2,0)
            plt.imshow(img)
            for bbox in bounding_boxes:
                plt.gca().add_patch(
                    plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor='g',
                                  linewidth=2))
            for cluster_box in cluster_boxes:
                plt.gca().add_patch(
                    plt.Rectangle((cluster_box[0], cluster_box[1]), cluster_box[2] - cluster_box[0], cluster_box[3] - cluster_box[1], fill=False,
                                  edgecolor='r', linewidth=2))
            plt.axis('off')
            plt.show()

            return clusters

        # area = (batch_gt_bboxes_orig[0][:, 2:]- batch_gt_bboxes_orig[0][:, :2]).prod(1)
        # if area.min() < 300:
        # if batch_data_samples[0].ori_shape != (800, 800):
        #     show_gt = True
        # else:
        #     show_gt = False
        show_gt = False
        if show_gt:
            import os
            import numpy as np
            import cv2
            import matplotlib.pyplot as plt

            mean = torch.tensor([0., 0., 0.])
            std = torch.tensor([255., 255., 255.])
            MEAN = [-mean / std for mean, std in zip(mean, std)]
            STD = [1 / std for std in std]
            denormalizer = transforms.Normalize(mean=MEAN, std=STD)

            # # ######################### show orig batch gt ###########################
            batch_inputs = batch_data_samples[0].batch_inputs
            img_visualize_ori = (denormalizer(batch_inputs[0])).byte().contiguous()
            image = img_visualize_ori.permute(1, 2, 0).cpu().numpy()
            output_path = '/root/autodl-tmp/img_ori.jpg'
            cv2.imwrite(output_path, image)
            # # ######################### 可视化 ###########################
            image = cv2.imread('/root/autodl-tmp/img_ori.jpg')
            # if len(batch_gt_bboxes[0])>2:
            #     clusters = generate_clusters_for_bounding_boxes(batch_gt_bboxes[0].cpu(), batch_inputs_orig[0])
            # cluster_bounding_boxes(batch_gt_bboxes[0].cpu(), batch_inputs_orig[0])
            # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
            import matplotlib.pyplot as plt
            plt.imshow(image)
            plt.show()
            batch_gt_bboxes = [copy.deepcopy(data_sample.gt_instances.bboxes) for data_sample in batch_data_samples]
            box_list = batch_gt_bboxes[0].cpu().numpy()
            # 遍历每个检测框并在图像上绘制
            for box in box_list:
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
            # 保存绘制好的图像到指定文件夹
            # output_path = '/root/autodl-tmp/img_ori_with_gt.jpg'
            # cv2.imwrite(output_path, image)

            plt.imshow(image)
            plt.show()


            # ######################## show new batch scaled gt ###########################
            # img_visualize_ori = (denormalizer(batch_inputs_orig[0])).byte().contiguous()
            # image = img_visualize_ori.permute(1, 2, 0).cpu().numpy()
            # output_path = '/root/autodl-tmp/img_ori.jpg'
            # cv2.imwrite(output_path, image)
            # # # ######################### 可视化 ###########################
            # image = cv2.imread('/root/autodl-tmp/img_ori.jpg')
            # # if len(batch_gt_bboxes[0])>2:
            # #     clusters = generate_clusters_for_bounding_boxes(batch_gt_bboxes[0].cpu(), batch_inputs_orig[0])
            # # cluster_bounding_boxes(batch_gt_bboxes[0].cpu(), batch_inputs_orig[0])
            # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # plt.show()
            #
            # box_list = batch_gt_bboxes_orig[0].cpu().numpy()
            # # 遍历每个检测框并在图像上绘制
            # for box in box_list:
            #     x1, y1, x2, y2 = box
            #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
            # # 保存绘制好的图像到指定文件夹
            # # output_path = '/root/autodl-tmp/img_ori_with_gt.jpg'
            # # cv2.imwrite(output_path, image)
            #
            # plt.imshow(image)
            # plt.show()
            # print('xx')

        losses = dict()
        orig_sample_combine_train = False
        if orig_sample_combine_train:
            x = self.extract_feat(torch.cat([batch_sampled_inputs, batch_inputs]))
            temp_batch_data_samples = deepcopy(batch_data_samples)
        else:
            x = self.extract_feat(batch_sampled_inputs)

        # warp bboxes to the resample image
        ori_shape = (batch_inputs_orig.shape[2:])
        batch_transformed_gt_bboxes = warp_bboxes(batch_gt_bboxes_orig, interpolators, ori_shape)
        for i in range(len(batch_data_samples)):
            batch_data_samples[i].gt_instances.bboxes = batch_transformed_gt_bboxes[i]

        if orig_sample_combine_train:
            batch_data_samples = batch_data_samples + temp_batch_data_samples

        # # #################### 可视化 ###########################
        ## show warp box
        if show_gt:
            mean = torch.tensor([0., 0., 0.])
            std = torch.tensor([255., 255., 255.])
            MEAN = [-mean / std for mean, std in zip(mean, std)]
            STD = [1 / std for std in std]
            denormalizer = transforms.Normalize(mean=MEAN, std=STD)
            # ######################## 可视化 ###########################
            img_visualize_ori = (denormalizer(batch_sampled_inputs[0])).byte().contiguous()
            image = img_visualize_ori.permute(1, 2, 0).cpu().numpy()
            output_path = '/root/autodl-tmp/img_ori.jpg'
            cv2.imwrite(output_path, image)
            # # ######################### 可视化 ###########################
            image = cv2.imread('/root/autodl-tmp/img_ori.jpg')
            import matplotlib.pyplot as plt
            plt.imshow(image)
            plt.show()

            box_list = batch_transformed_gt_bboxes[0].cpu().numpy()
            # 遍历每个检测框并在图像上绘制
            for box in box_list:
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
            # 保存绘制好的图像到指定文件夹
            # output_path = '/root/autodl-tmp/img_ori_with_gt.jpg'
            # cv2.imwrite(output_path, image)

            plt.imshow(image)
            plt.show()


            ## show unwarp box
            # batch_inputs = batch_data_samples[0].batch_inputs
            batch_inputs = batch_inputs_orig
            img_visualize_ori = (denormalizer(batch_inputs[0])).byte().contiguous()
            image = img_visualize_ori.permute(1, 2, 0).cpu().numpy()
            output_path = '/root/autodl-tmp/img_ori.jpg'
            cv2.imwrite(output_path, image)
            # # ######################### 可视化 ###########################
            image = cv2.imread('/root/autodl-tmp/img_ori.jpg')
            # if len(batch_gt_bboxes[0])>2:
            #     clusters = generate_clusters_for_bounding_boxes(batch_gt_bboxes[0].cpu(), batch_inputs_orig[0])
            # cluster_bounding_boxes(batch_gt_bboxes[0].cpu(), batch_inputs_orig[0])
            # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
            import matplotlib.pyplot as plt
            plt.imshow(image)
            plt.show()
            unwarp_transformed_bboxes = unwarp_bboxes(batch_transformed_gt_bboxes[0], batch_input_shape, grid[0],
                                                      tuple(batch_inputs_orig.shape[2:]))

            box_list = unwarp_transformed_bboxes.detach().cpu().numpy()
            # 遍历每个检测框并在图像上绘制
            for box in box_list:
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
            # 保存绘制好的图像到指定文件夹
            # output_path = '/root/autodl-tmp/img_ori_with_gt.jpg'
            # cv2.imwrite(output_path, image)

            plt.imshow(image)
            plt.show()



        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)

        mag_weight = self.train_cfg['saliency_loss_weight']['weight']
        small_th = self.train_cfg['saliency_loss_weight']['small_threshold']
        medium_th = self.train_cfg['saliency_loss_weight']['medium_threshold']
        large_th = self.train_cfg['saliency_loss_weight']['large_threshold']
        saliency_losses = saliency_loss(saliency_list, grid, uniform_grid, batch_gt_bboxes_orig,
                                        ori_shape, mag_weight, small_th, medium_th, large_th)
        losses.update(saliency_losses)

        # fa_weight = self.train_cfg['fa_loss_weight']['weight']
        # fa_losses = offset_flip_loss(saliency_list, batch_transformed_gt_bboxes, batch_gt_bboxes_orig, ori_shape, fa_weight, mag_weight)
        # losses.update(fa_losses)
        return losses

    def predict(self,
                batch_inputs_orig: Tensor,
                batch_data_samples: SampleList,
                min_ratio=None,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        # import torch.nn as nn
        # nn.init.constant_(self.grid_generator.saliency_network.conv_last.weight, 0.)
        # nn.init.constant_(self.grid_generator.saliency_network.conv_last.bias, 0.)
        # batch_inputs : orig res inputs
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_input_shape = batch_data_samples[0].batch_input_shape
        batch_input_orig_shape = tuple(batch_inputs_orig.shape[2:])

        detach_switch = self.train_cfg['detach_detector_loss']['switch']

        batch_gt_bboxes = [data_sample.gt_instances.bboxes for data_sample in batch_data_samples]
        batch_gt_bboxes_orig = [copy.deepcopy(data_sample.gt_instances_orig_maybelarger_due2batch.bboxes) for data_sample in batch_data_samples]

        batch_inputs = F.interpolate(batch_inputs_orig, size=batch_input_shape, mode='bilinear', align_corners=True)
        # batch_inputs = batch_data_samples[0].batch_inputs

        grid, _, interpolators, saliency_map = self.grid_generator(batch_inputs, batch_input_shape)

        if len(batch_gt_bboxes[0])>0:
            batch_transformed_gt_bboxes = warp_bboxes(batch_gt_bboxes, interpolators, batch_input_shape)
            unwarp_transformed_bboxes = unwarp_bboxes(batch_transformed_gt_bboxes[0], batch_input_shape, grid[0], batch_input_shape)
            # unwarp_transformed_bboxes = batch_transformed_gt_bboxes[0]
            # compute_IOU(batch_gt_bboxes[0], unwarp_transformed_bboxes, self.test_cfg)
            # if (unwarp_transformed_bboxes - batch_transformed_gt_bboxes[0]).sum() != 0:
            #     print('xx')
            compute_IOU(batch_gt_bboxes[0], unwarp_transformed_bboxes, self.test_cfg)
        else:
            self.test_cfg['iou_box_num_sum']['image_sum'] += 1

        # start_time = time_synchronized()
        batch_sampled_inputs = F.grid_sample(batch_inputs_orig, grid, padding_mode='reflection')
        # end_time = time_synchronized(start_time, 'grid_sample time')

        # compute zoom-in multiple
        if len(batch_gt_bboxes[0])>0:
            compute_zoom_in_multiple(batch_gt_bboxes, interpolators, batch_input_shape, self.test_cfg, min_ratio)

        else:
            self.test_cfg['box_num_sum']['image_sum'] += 1

        # mean = torch.tensor([0., 0., 0.])
        # std = torch.tensor([255., 255., 255.])
        # MEAN = [-mean / std for mean, std in zip(mean, std)]
        # STD = [1 / std for std in std]
        # denormalizer = transforms.Normalize(mean=MEAN, std=STD)
        # ######################## 可视化 ###########################
        # img_visualize_ori = (denormalizer(batch_inputs[0])).byte().contiguous()
        # image = img_visualize_ori.permute(1, 2, 0).cpu().numpy()
        # # image = cv2.resize(image, tuple([batch_inputs.shape[3], batch_inputs.shape[2]]))
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled_ori.jpg'
        # cv2.imwrite(output_path, image)
        # # # # # # ######################### 可视化 ###########################
        # # # image = cv2.imread('/root/autodl-tmp/best_05_25/img_resampled_ori_{}.jpg'.format(img_id))
        # # # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # # # box_list = batch_gt_bboxes[0].cpu().numpy()
        # # # # 遍历每个检测框并在图像上绘制
        # # # for box in box_list:
        # # #     x1, y1, x2, y2 = box
        # # #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # # # 保存绘制好的图像到指定文件夹
        # # # output_path = '/root/autodl-tmp/best_05_25/img_ori_with_gt_{}.jpg'.format(img_id)
        # # # cv2.imwrite(output_path, image)
        # # # # # # ######################### 可视化 ###########################
        # # # # image = cv2.imread('/root/autodl-tmp/img_resampled_ori.jpg')
        # # # # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # # # # box_list = unwarp_transformed_bboxes.cpu().numpy()
        # # # # # 遍历每个检测框并在图像上绘制
        # # # # for box in box_list:
        # # # #     x1, y1, x2, y2 = box
        # # # #     cv2.rectangle(image, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # # # # 保存绘制好的图像到指定文件夹
        # # # # output_path = '/root/autodl-tmp/img_ori_with_unwarp_gt.jpg'
        # # # # cv2.imwrite(output_path, image)
        # ####################### 可视化 ###########################
        # img_visualize_resampled = (denormalizer(batch_sampled_inputs[0])).byte().contiguous()
        # image_2 = img_visualize_resampled.permute(1, 2, 0).cpu().numpy()
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled.jpg'
        # cv2.imwrite(output_path, image_2)
        # # ######################### 可视化 ###########################
        # image = cv2.imread('/root/autodl-tmp/img_resampled.jpg')
        # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # box_list = batch_transformed_gt_bboxes[0].cpu().numpy()
        # # 遍历每个检测框并在图像上绘制
        # for box in box_list:
        #     x1, y1, x2, y2 = box
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled_with_gt.jpg'
        # cv2.imwrite(output_path, image)
        # ##################### 绘制采样点 ##########################
        # img = cv2.imread('/root/autodl-tmp/img_resampled_ori.jpg')
        # for i in range(0, grid.shape[1], 4):
        #     for j in range(0, grid.shape[2], 4):
        #         x = (grid[0, i, j, 0] + 1) / 2 * batch_inputs.shape[3]
        #         y = (grid[0, i, j, 1] + 1) / 2 * batch_inputs.shape[2]
        #         cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)  # 绘制红色的点，半径为1
        # # 保存处理后的图像
        # # img = cv2.resize(img, tuple([batch_inputs.shape[3], batch_inputs.shape[2]]))
        # cv2.imwrite("/root/autodl-tmp/sample_location.jpg", img)

        ## orig+sample inference
        orig_sample_combine_inference = False
        if orig_sample_combine_inference:
            x = self.extract_feat(torch.cat([batch_sampled_inputs, batch_inputs]))
            batch_data_samples.append(batch_data_samples[0])
        else:
            x = self.extract_feat(batch_sampled_inputs)


        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=False)

        import torchvision.ops as ops
        def combine_results(boxes1, labels1, scores1, boxes2, labels2, scores2, threshold=0.8):
            # 将两个模型的结果合并
            combined_boxes = torch.cat((boxes1, boxes2), dim=0)
            combined_labels = torch.cat((labels1, labels2), dim=0)
            combined_scores = torch.cat((scores1, scores2), dim=0)

            # 使用NMS去除重叠的边界框
            keep = ops.nms(combined_boxes, combined_scores, threshold)

            # 保留NMS后的结果
            combined_boxes = combined_boxes[keep]
            combined_labels = combined_labels[keep]
            combined_scores = combined_scores[keep]

            return combined_boxes, combined_labels, combined_scores

        if orig_sample_combine_inference:
            combined_result = InstanceData()
            results_list[0]['bboxes'] = unwarp_bboxes(results_list[0]['bboxes'], batch_input_shape, grid[0], tuple(batch_inputs_orig.shape[2:]))
            results_list[1]['bboxes'] /= min_ratio
            combined_result.bboxes, combined_result.labels, combined_result.scores = combine_results(results_list[0].bboxes, results_list[0].labels, results_list[0].scores,
                                results_list[1].bboxes, results_list[1].labels, results_list[1].scores)
            del results_list
            results_list = []
            results_list.append(combined_result)
            batch_data_samples.pop(1)

        show = self.test_cfg['show_pred']
        if show:
        # if show and batch_input_orig_shape!=(800,800):
            mean = torch.tensor([0., 0., 0.])
            std = torch.tensor([255., 255., 255.])
            MEAN = [-mean / std for mean, std in zip(mean, std)]
            STD = [1 / std for std in std]
            denormalizer = transforms.Normalize(mean=MEAN, std=STD)

            image = denormalizer(batch_sampled_inputs[0]).byte().contiguous()
            image = image.permute(1, 2, 0).cpu().numpy()
            output_path = '/root/autodl-tmp/img_ori.jpg'
            cv2.imwrite(output_path, image)
            show_gt = False
            if show_gt:
                import matplotlib.pyplot as plt
                image = cv2.imread('/root/autodl-tmp/img_ori.jpg')
                # image = cv2.UMat(image.permute(1, 2, 0).cpu().numpy())
                box_list = batch_gt_bboxes[0].cpu().numpy()

                # # 遍历每个检测框并在图像上绘制
                for box in box_list:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                                  2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度

                plt.imshow(image)
                plt.show()

            image = cv2.imread('/root/autodl-tmp/img_ori.jpg')
            import matplotlib.pyplot as plt
            plt.imshow(image)
            plt.show()
            # image = cv2.UMat(image.permute(1, 2, 0).cpu().numpy())
            box_list = results_list[0]['bboxes'].cpu().numpy()
            # 遍历每个检测框并在图像上绘制
            for box in box_list:
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                              2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
            # 保存绘制好的图像到指定文件夹
            # output_path = '/root/autodl-tmp/img_resampled_predict.jpg'
            # cv2.imwrite(output_path, image)
            plt.imshow(image)
            plt.show()
            ######################## 可视化 ###########################
            img_visualize_ori = (denormalizer(batch_inputs_orig[0])).byte().contiguous()
            image_2 = img_visualize_ori.permute(1, 2, 0).cpu().numpy()

            plt.imshow(image_2)
            plt.show()

        if not orig_sample_combine_inference:
            img_shape = batch_data_samples[0].pad_shape
            # start_time_2 = time_synchronized()
            # results_list[0]['bboxes'] = unwarp_bboxes(results_list[0]['bboxes'], img_shape,
            #                                           grid[0], tuple(batch_inputs_orig.shape[2:]))
            results_list[0]['bboxes'] = unwarp_bboxes(results_list[0]['bboxes'], img_shape,
                                                      grid[0], batch_data_samples[0].batch_orig_shape_for_inference)

        # end_time = time_synchronized(start_time_2, 'unwarp time')

        # # ######################## 可视化 ###########################
        if show:
        # if show and batch_input_orig_shape != (800, 800):
            image = denormalizer(batch_data_samples[0].ori_inputs).byte().contiguous()
            image = image.permute(1, 2, 0).cpu().numpy()
            output_path = '/root/autodl-tmp/img_ori.jpg'
            cv2.imwrite(output_path, image)
            image = cv2.imread('/root/autodl-tmp/img_ori.jpg')
            # image = cv2.UMat(image.permute(1, 2, 0).cpu().numpy())
            box_list = results_list[0]['bboxes'].cpu().numpy()

            # # 遍历每个检测框并在图像上绘制
            for box in box_list:
                x1, y1, x2, y2 = box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                              2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度

            plt.imshow(image)
            plt.show()
        # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_resampled_ori_predict.jpg'
        # cv2.imwrite(output_path, image)
        # ###################################################

        # colors = [
        #     (0, 0, 255),  # 红色 pedestrian
        #     (0, 255, 0),  # 绿色people
        #     (255, 0, 0),  # 蓝色bicycle
        #     (0, 255, 255),  # 黄色car
        #     (128, 0, 128),  # 紫色van
        #     (255, 255, 0),  # 青色truck
        #     (0, 165, 255),  # 橙色tricycle
        #     (203, 192, 255),  # 粉色awning-tricycle
        #     (128, 128, 128),  # 灰色bus
        #     (19, 69, 139)  # 棕色motor
        # ]
        # ######################### 可视化 ###########################
        # image = cv2.imread('/root/autodl-tmp/img_thr_1_1_1/img_resampled_ori_{}.jpg'.format(img_id))
        # # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
        # box_list = results_list[0]['bboxes']
        # for i in range(box_list.shape[0]):
        #     score = results_list[0]['scores'][i]
        # # 遍历每个检测框并在图像上绘制
        #     x1, y1, x2, y2 = box_list[i]
        #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[results_list[0]['labels'][i]],
        #                   2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        #     cv2.putText(image, f'{score.item():.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[results_list[0]['labels'][i]], 2)
        # # 保存绘制好的图像到指定文件夹
        # # gt_list = batch_gt_bboxes[0].cpu().numpy()
        # # # 遍历每个检测框并在图像上绘制
        # # for i in range(gt_list.shape[0]):
        # #     x1, y1, x2, y2 = gt_list[i]
        # #     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[batch_data_samples[0].gt_instances.labels[i]], 2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
        # # # 保存绘制好的图像到指定文件夹
        # output_path = '/root/autodl-tmp/img_thr_1_1_1/img_resampled_ori_predict_{}.jpg'.format(img_id)
        # cv2.imwrite(output_path, image)
        ####################################################

        # # 创建一个新的Word文档
        # doc = Document()
        #
        # # 添加标题
        # doc.add_heading('TP and FP for each class', level=1)
        #
        # pred_bboxes = results_list[0].bboxes
        # gt_bboxes = batch_gt_bboxes[0].tensor
        # pred_labels = results_list[0].labels
        # gt_labels = batch_data_samples[0].gt_instances.labels
        # gt_unique_label = gt_labels.unique()
        # ap_sum = 0

        # equal_cls_7_id = (gt_labels == 7).nonzero()
        # seven_gt_bboxes = gt_bboxes[equal_cls_7_id]
        # for i in range(seven_gt_bboxes.shape[0]):
        #     x1, y1, x2, y2 = seven_gt_bboxes[i]
        #     if (x2- x1)*(y2-y1) > 4096:
        #         print()

        # for i in range(gt_unique_label.shape[0]):
        #     label = gt_unique_label[i]
        #     gt_equal_indices = torch.nonzero(torch.eq(gt_labels, label)).squeeze()
        #     pred_equal_indices = torch.nonzero(torch.eq(pred_labels, label)).squeeze()
        #     if pred_equal_indices.size == 0:
        #         ap_sum += 0
        #     else:
        #         ap, doc = calculate_ap(pred_bboxes[pred_equal_indices], gt_bboxes[gt_equal_indices], gt_unique_label[i], doc)
        #         ap_sum += ap
        #
        # # map = ap_sum / gt_unique_label.shape[0]
        # doc.save('/root/autodl-tmp/analysis/{}.docx'.format(batch_data_samples[0].metainfo['img_id']))
        # self.test_cfg['ap_excel']['img_id'].append(batch_data_samples[0].metainfo['img_id'])
        # self.test_cfg['ap_excel']['ap'].append(map.item())
        # self.test_cfg['box_num_sum']['sum'] += 1
        #
        # if self.test_cfg['box_num_sum']['sum'] == 548:
        #     # 创建一个 DataFrame 对象
        #     df = pd.DataFrame(self.test_cfg['ap_excel'])
        #     # 指定 Excel 文件的名称和路径
        #     excel_file = "/root/autodl-tmp/deformable_ap.xlsx"
        #     # 将 DataFrame 写入 Excel 文件
        #     df.to_excel(excel_file, index=False)

        # if avg_map > self.test_cfg['max_ap']['ap']:
        #     self.test_cfg['max_ap']['ap'] = avg_map
        #     self.test_cfg['max_ap']['img_id'] = batch_data_samples[0].metainfo['img_id']
        #     print("max_ap_img is", self.test_cfg['max_ap']['img_id'])

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples



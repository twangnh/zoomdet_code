# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector

from typing import List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
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

from .faster_rcnn import unwarp_bboxes, warp_bboxes, compute_zoom_in_multiple, calculate_iou, calculate_iou_for_2_box, calculate_ap, compute_IOU
from copy import deepcopy

@MODELS.register_module()
class RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)


@MODELS.register_module()
class RetinaNetZoom(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.grid_generator = make_grid_generator(sep_fwhm=self.train_cfg['sep_fwhm']['fwhm'],
                                                  nonsep_fwhm=self.train_cfg['nonsep_fwhm']['fwhm'])

    def loss(self, batch_inputs_orig: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
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

        # losses = dict()
        orig_sample_combine_train = False
        if orig_sample_combine_train:
            x = self.extract_feat(torch.cat([batch_sampled_inputs, batch_inputs]))
            temp_batch_data_samples = deepcopy(batch_data_samples)
        else:
            x = self.extract_feat(batch_sampled_inputs)

        # warp bboxes to the resample image
        ori_shape = (batch_inputs_orig.shape[2:])
        filter_invalid=False
        if filter_invalid:
            batch_transformed_gt_bboxes, keeps = warp_bboxes(batch_gt_bboxes_orig, interpolators, ori_shape, filter_invalid=True)
            for i in range(len(batch_data_samples)):
                # batch_data_samples[i].gt_instances.bboxes = batch_transformed_gt_bboxes[i]
                batch_data_samples[i].gt_instances.set_field(batch_transformed_gt_bboxes[i], 'bboxes')
                orig_labels = batch_data_samples[i].gt_instances.labels
                batch_data_samples[i].gt_instances.set_field(orig_labels[keeps[i]], 'labels')
        else:
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
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                              2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
            # 保存绘制好的图像到指定文件夹
            # output_path = '/root/autodl-tmp/img_ori_with_gt.jpg'
            # cv2.imwrite(output_path, image)

            plt.imshow(image)
            plt.show()


        # x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)

        mag_weight = self.train_cfg['saliency_loss_weight']['weight']
        small_th = self.train_cfg['saliency_loss_weight']['small_threshold']
        medium_th = self.train_cfg['saliency_loss_weight']['medium_threshold']
        large_th = self.train_cfg['saliency_loss_weight']['large_threshold']
        saliency_losses = saliency_loss(saliency_list, grid, uniform_grid, batch_gt_bboxes_orig,
                                        ori_shape, mag_weight, small_th, medium_th, large_th)
        losses.update(saliency_losses)


        return losses

    def predict(self,
                batch_inputs_orig: Tensor,
                batch_data_samples: SampleList,
                min_ratio=None,
                rescale: bool = True) -> SampleList:

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



        # x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=False)

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

        # # ######################## 可视化原图结果 ###########################
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

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)

        return batch_data_samples
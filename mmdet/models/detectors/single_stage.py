# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector


@MODELS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
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
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)

        show_gt = False
        if show_gt:
            import os
            import numpy as np
            import copy
            import torch
            import cv2
            from torchvision import transforms
            import matplotlib.pyplot as plt

            # # ######################### show orig batch gt ###########################
            # batch_inputs = batch_data_samples[0].batch_inputs
            # img_visualize_ori = (denormalizer(batch_inputs[0])).byte().contiguous()
            image = batch_inputs[0].cpu().permute(1, 2, 0).numpy()

            # image = img_visualize_ori.permute(1, 2, 0).cpu().numpy()
            # output_path = '/root/autodl-tmp/img_ori.jpg'
            # cv2.imwrite(output_path, image)
            # # ######################### 可视化 ###########################
            # image = cv2.imread('/root/autodl-tmp/img_ori.jpg')
            # if len(batch_gt_bboxes[0])>2:
            #     clusters = generate_clusters_for_bounding_boxes(batch_gt_bboxes[0].cpu(), batch_inputs_orig[0])
            # cluster_bounding_boxes(batch_gt_bboxes[0].cpu(), batch_inputs_orig[0])
            # 假设box_list是包含检测框的列表，每个框都是一个元组 (x1, y1, x2, y2)
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            plt.imshow(image)
            plt.show()

            # Create a figure and a set of subplots
            fig, ax = plt.subplots()
            # Display the image
            ax.imshow(image)
            batch_gt_bboxes = [copy.deepcopy(data_sample.gt_instances.bboxes) for data_sample in batch_data_samples]
            box_list = batch_gt_bboxes[0].cpu().numpy()
            # 遍历每个检测框并在图像上绘制
            for box in box_list:
                # x1, y1, x2, y2 = box
                # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                #               2)  # 在图像上绘制矩形框，(0, 255, 0) 是绿色，2 是线宽度
                # Create a rectangle patch with the xyxy format
                rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1,
                                         edgecolor='g', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)
            # 保存绘制好的图像到指定文件夹
            # output_path = '/root/autodl-tmp/img_ori_with_gt.jpg'
            # cv2.imwrite(output_path, image)
            plt.imshow(image)
            plt.show()

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
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
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)


        # # ######################## 可视化原图结果 ###########################
        import cv2
        import matplotlib.pyplot as plt
        import torch
        from torchvision import transforms

        show=False
        if show:
        # if show and batch_input_orig_shape != (800, 800):
            mean = torch.tensor([0., 0., 0.])
            std = torch.tensor([255., 255., 255.])
            MEAN = [-mean / std for mean, std in zip(mean, std)]
            STD = [1 / std for std in std]
            denormalizer = transforms.Normalize(mean=MEAN, std=STD)
            image = denormalizer(batch_data_samples[0].ori_inputs).byte().contiguous()
            image = image.permute(1, 2, 0).cpu().numpy()
            output_path = '/root/autodl-tmp/img_ori.jpg'
            cv2.imwrite(output_path, image)
            image = cv2.imread('/root/autodl-tmp/img_ori.jpg')

            plt.imshow(image)
            plt.show()
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

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

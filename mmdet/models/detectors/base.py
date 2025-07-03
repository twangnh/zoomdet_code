# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import torch
from mmengine.model import BaseModel
from torch import Tensor
import mmcv
import numpy as np
import time

from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from ..utils import samplelist_boxtype2tensor
import torch.nn.functional as F

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

from copy import deepcopy

class BaseDetector(BaseModel, metaclass=ABCMeta):
    """Base class for detectors.

    Args:
       data_preprocessor (dict or ConfigDict, optional): The pre-process
           config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
       init_cfg (dict or ConfigDict, optional): the config to control the
           initialization. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    # TODO: these properties need to be carefully handled
    # for both single stage & two stage detectors
    @property
    def with_shared_head(self) -> bool:
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self) -> bool:
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if self.__class__.__name__ == 'RezoomedFasterRCNN' or self.__class__.__name__ == 'RetinaNetZoom':
            ## 原图(ori_shape)放缩(以scale factor缩小)到img_shape(被800x1333窗限制的最大大小),然后pad到pad_shape/batch_input_shape
            min_ratio = 10
            for i in range(len(data_samples)):
                if min(data_samples[i].scale_factor) < min_ratio:
                    min_ratio = max(data_samples[i].scale_factor)
            ## as max clamp min ratio, this step stores the original batch shape for inference, so that boxes are mapped to original input image shape
            batch_orig_shape_for_inference = (int(np.ceil(inputs.shape[2] / min_ratio)),
                                           int(np.ceil(inputs.shape[3] / min_ratio)))
            ## uncomment to prevent very large original images
            min_ratio = max(0.3, min_ratio)
            # batch_shape = (int(np.ceil(inputs.shape[2] / min_ratio / 32) * 32),
            #                           int(np.ceil(inputs.shape[3] / min_ratio / 32) * 32))
            batch_shape = (int(np.ceil(inputs.shape[2] / min_ratio)),
                                      int(np.ceil(inputs.shape[3] / min_ratio)))
            # print(min_ratio)
            batch_images_list = []
            for i in range(len(data_samples)):
                # 放大至新画布大小
                image = F.interpolate(data_samples[i].ori_inputs.unsqueeze(0),
                                      size=((round(data_samples[i].img_shape[0] / min_ratio),
                                             round(data_samples[i].img_shape[1] / min_ratio))),
                                      mode='bilinear', align_corners=True)

                data_samples[i].gt_instances_orig_maybelarger_due2batch = deepcopy(data_samples[i].gt_instances)
                data_samples[i].gt_instances_orig_maybelarger_due2batch.bboxes = data_samples[i].gt_instances.bboxes / min_ratio

                # 计算 padding 大小
                pad_height = max(batch_shape[0] - image.shape[2], 0)
                pad_width = max(batch_shape[1] -image.shape[3], 0)

                # 计算 padding 参数
                padding = (0, pad_width, 0, pad_height)

                # 执行 padding
                batch_images_list.append(F.pad(image, padding))

            hr_inputs = torch.cat(batch_images_list, dim=0)

            if mode == 'loss':
                data_samples[0].set_field(inputs, 'batch_inputs')
                data_samples[0].set_field(batch_orig_shape_for_inference, 'batch_orig_shape_for_inference')

                return self.loss(hr_inputs, data_samples)
            elif mode == 'predict':
                data_samples[0].set_field(inputs, 'batch_inputs')
                data_samples[0].set_field(batch_orig_shape_for_inference, 'batch_orig_shape_for_inference')
                return self.predict(hr_inputs, data_samples, min_ratio)
            elif mode == 'tensor':
                return self._forward(inputs, data_samples)
            else:
                raise RuntimeError(f'Invalid mode "{mode}". '
                                   'Only supports loss, predict and tensor mode')

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, tuple]:
        """Calculate losses from a batch of inputs and data samples."""
        pass

    @abstractmethod
    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        pass

    @abstractmethod
    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        pass

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: InstanceList) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

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
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        samplelist_boxtype2tensor(data_samples)
        return data_samples

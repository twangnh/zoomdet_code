# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS



@HOOKS.register_module()
class GaussianBlurHook(Hook):

    priority = 'VERY_HIGH'

    def __init__(self,
                 blur_by_epoch: int = 2) -> None:
        self.blur_by_epoch = blur_by_epoch

    def before_train_epoch(self, runner) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if (runner.epoch + 1) > self.blur_by_epoch:
            runner.model.module.train_cfg['gaussian_probability']['p'] = 1
            # runner.model.module.train_cfg['saliency_loss_weight']['weight'] = 10
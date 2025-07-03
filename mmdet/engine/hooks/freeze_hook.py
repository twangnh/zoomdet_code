# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS
from typing import Callable, Dict, List, Optional, Sequence, Union

DATA_BATCH = Optional[Union[dict, tuple, list]]


@HOOKS.register_module()
class FreezeHook(Hook):

    priority = 'VERY_HIGH'

    def __init__(self,
                 freeze_by_epoch: int = 2) -> None:
        self.freeze_by_epoch = freeze_by_epoch

    def before_train_epoch(self, runner) -> None:
        """Save the checkpoint and synchronize buffers after each epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if (runner.epoch + 1) > self.freeze_by_epoch:
            runner.model.module.train_cfg['detach_detector_loss']['switch'] = 1
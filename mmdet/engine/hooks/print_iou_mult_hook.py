# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet.datasets.samplers import TrackImgSampler
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample, TrackDataSample

from mmengine.dist import is_main_process
from mmengine.logging import MMLogger

@HOOKS.register_module()
class PrintIouMultHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def after_val_epoch(self, runner: Runner, metrics) -> None:
        logger: MMLogger = MMLogger.get_current_instance()

        if hasattr(runner.model, 'module'):
            test_cfg = runner.model.module.test_cfg
        else:
            test_cfg = runner.model.test_cfg
        if is_main_process():
            logger.info("global multiple is: {}".format(test_cfg['multiple_sum']['sum'] / (test_cfg['box_num_sum']['sum'] + 1e-5)))
            logger.info("small multiple is: {}".format(test_cfg['multiple_sum']['small_sum'] / (test_cfg['box_num_sum']['small_sum'] + 1e-5)))
            logger.info("medium multiple is: {}".format(test_cfg['multiple_sum']['medium_sum'] / (test_cfg['box_num_sum']['medium_sum'] + 1e-5)))
            logger.info("large multiple is: {}".format(test_cfg['multiple_sum']['large_sum'] / (test_cfg['box_num_sum']['large_sum'] + 1e-5)))
            logger.info("max multiple is: {}".format(test_cfg['multiple_sum']['max_multiple']))
            logger.info("more than 2 multiple sum: {}".format(test_cfg['box_num_sum']['multiple_more_than_2']))
            logger.info('\n')

            logger.info('sum {}'.format(test_cfg['iou_box_num_sum']['sum']))
            logger.info("mean IOU is: {}".format(test_cfg['iou_sum']['sum'] / (test_cfg['iou_box_num_sum']['sum'] + 1e-5)))
            logger.info("small mean IOU is: {}".format(test_cfg['iou_sum']['small_sum'] / (test_cfg['iou_box_num_sum']['small_sum']+1e-5)))
            logger.info("medium mean IOU is: {}".format(test_cfg['iou_sum']['medium_sum'] / (test_cfg['iou_box_num_sum']['medium_sum']+1e-5)))
            logger.info("large mean IOU is: {}".format(test_cfg['iou_sum']['large_sum'] / (test_cfg['iou_box_num_sum']['large_sum']+1e-5)))

    def after_test(self, runner: Runner) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if hasattr(runner.model, 'module'):
            test_cfg = runner.model.module.test_cfg
        else:
            test_cfg = runner.model.test_cfg
        if is_main_process():
            logger.info("global multiple is: {}".format(test_cfg['multiple_sum']['sum'] / (test_cfg['box_num_sum']['sum'] + 1e-5)))
            logger.info("small multiple is: {}".format(test_cfg['multiple_sum']['small_sum'] / (test_cfg['box_num_sum']['small_sum'] + 1e-5)))
            logger.info("medium multiple is: {}".format(test_cfg['multiple_sum']['medium_sum'] / (test_cfg['box_num_sum']['medium_sum'] + 1e-5)))
            logger.info("large multiple is: {}".format(test_cfg['multiple_sum']['large_sum'] / (test_cfg['box_num_sum']['large_sum'] + 1e-5)))
            logger.info("max multiple is: {}".format(test_cfg['multiple_sum']['max_multiple']))
            logger.info("more than 2 multiple sum: {}".format(test_cfg['box_num_sum']['multiple_more_than_2']))
            logger.info('\n')

            logger.info('sum {}'.format(test_cfg['iou_box_num_sum']['sum']))
            logger.info("mean IOU is: {}".format(test_cfg['iou_sum']['sum'] / (test_cfg['iou_box_num_sum']['sum'] + 1e-5)))
            logger.info("small mean IOU is: {}".format(test_cfg['iou_sum']['small_sum'] / (test_cfg['iou_box_num_sum']['small_sum']+1e-5)))
            logger.info("medium mean IOU is: {}".format(test_cfg['iou_sum']['medium_sum'] / (test_cfg['iou_box_num_sum']['medium_sum']+1e-5)))
            logger.info("large mean IOU is: {}".format(test_cfg['iou_sum']['large_sum'] / (test_cfg['iou_box_num_sum']['large_sum']+1e-5)))

# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import time
from collections import defaultdict

import numpy as np
import torch.multiprocessing as mp
from mmengine.logging import MMLogger
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


class COCOevalexsmall(COCOeval):
    def summarize(self, logger):
        """Compute and display summary metrics for evaluation results.

        Note this function can *only* be applied on the default parameter
        setting
        """

        def _summarize(logger, ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'  # noqa
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [
                i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
            ]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            # print(
            #     iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
            #                 mean_s))
            logger.info(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                            mean_s))
            return mean_s

        def _summarizeDets(logger):
            stats = []
            stats.append(_summarize(logger, 1, maxDets=self.params.maxDets[-1]))
            stats.append(
                _summarize(logger, 1, iouThr=.5, maxDets=self.params.maxDets[-1]))
            stats.append(
                _summarize(logger, 1, iouThr=.75, maxDets=self.params.maxDets[-1]))
            for area_rng in ('small', 'medium', 'large', 'ex_small'):
                stats.append(
                    _summarize(
                        logger, 1, areaRng=area_rng, maxDets=self.params.maxDets[-1]))
            for max_det in self.params.maxDets:
                stats.append(_summarize(logger, 0, maxDets=max_det))
            for area_rng in ('small', 'medium', 'large', 'ex_small'):
                stats.append(
                    _summarize(
                        logger, 0, areaRng=area_rng, maxDets=self.params.maxDets[-1]))
            stats = np.array(stats)
            return stats

        def _summarizeKps():
            stats = np.zeros((10, ))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize(logger)
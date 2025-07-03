import torch
import torch.nn.functional as F
import math
import cv2
import time

def kl_divergence_loss(input_tensor, target_tensor):
    input_log_softmax = F.log_softmax(input_tensor, dim=1)
    target_softmax = F.softmax(target_tensor, dim=1)
    kl_loss = F.kl_div(input_log_softmax, target_softmax, reduction='batchmean')
    return kl_loss

def foreground_align_uniform_loss(grid, batch_transformed_gt_bboxes, batch_gt_bboxes, ori_shapes, rz_shape, fa_weight):
    fa_loss_dict = {}
    fa_loss = 0
    fa_loss_all = torch.tensor(0, dtype=torch.float32).to('cuda')

    for i in range(grid.shape[0]):
        ori_img_width = ori_shapes[i][1]
        ori_img_height = ori_shapes[i][0]
        for box_id in range(batch_gt_bboxes[i].shape[0]):
            x1_u, y1_u, x2_u, y2_u = batch_gt_bboxes[i].tensor[box_id, :]
            x1, y1, x2, y2 = batch_transformed_gt_bboxes[i][box_id, :]

            a = (x1_u / ori_img_width) * 2 - 1
            b = (x2_u / ori_img_width) * 2 - 1
            c = (y1_u / ori_img_height) * 2 - 1
            d = (y2_u / ori_img_height) * 2 - 1
            indices = torch.nonzero(
                (grid[0, :, :, 0] >= a) & (grid[0, :, :, 0] <= b) & (grid[0, :, :, 1] >= c) & (grid[0, :, :, 1] <= d))

            if indices.size(0) > 0:
                y_max = torch.max(indices[:, 0])
                y_min = torch.min(indices[:, 0])
                x_max = torch.max(indices[:, 1])
                x_min = torch.min(indices[:, 1])

                if (y_max != y_min) & (x_max != x_min):

                    offset_x = (grid[i, int(y_min):int(y_max), int(x_min+1):int(x_max), 0] \
                               - grid[i, int(y_min):int(y_max), int(x_min):int(x_max-1), 0]) * rz_shape[1]

                    offset_y = (grid[i, int(y_min+1):int(y_max), int(x_min):int(x_max), 1] \
                               - grid[i, int(y_min):int(y_max-1), int(x_min):int(x_max), 1]) * rz_shape[0]

                    if (offset_x.size(0) > 0) & (offset_y.size(0) > 0) & \
                            (offset_x.size(1) > 0) & (offset_y.size(1) > 0):
                    # #     uniform_tensor_x = torch.full((int(y_max - y_min), int(x_max - x_min - 1)), 1,
                    # #                                   dtype=torch.float32).to('cuda')
                    # #     uniform_tensor_y = torch.full((int(y_max - y_min - 1), int(x_max - x_min)), 1,
                    # #                                   dtype=torch.float32).to('cuda')
                    # #     loss_x = kl_divergence_loss(offset_x, uniform_tensor_x)
                    # #     loss_y = kl_divergence_loss(offset_y, uniform_tensor_y)
                    # #     loss_2 = loss_x + loss_y
                    # # else:
                    # #     loss_2 = torch.tensor(0, dtype=torch.float32).to('cuda')
                    #
                        ######## version_2 ############
                        mean_x = torch.mean(offset_x)
                        loss_x = torch.mean(torch.abs(offset_x - mean_x))
                        mean_y = torch.mean(offset_y)
                        loss_y = torch.mean(torch.abs(offset_y - mean_y))

                        loss_2 = loss_x + loss_y
                    else:
                        loss_2 = torch.tensor(0, dtype=torch.float32).to('cuda')

                    fa_loss += loss_2

        fa_loss_all += fa_loss / batch_gt_bboxes[i].shape[0]
        fa_loss = 0

    fa_loss_dict['fa_uniform_loss'] = fa_weight * fa_loss_all

    return fa_loss_dict
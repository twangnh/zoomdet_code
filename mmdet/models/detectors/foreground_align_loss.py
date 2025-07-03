import torch
import torch.nn.functional as F
import math
import cv2
import time


def offset_flip_loss(saliency_list, batch_transformed_gt_bboxes, batch_gt_bboxes, ori_shape, fa_weight, mag_weight):
    """
    predicted_grid: Tensor of shape [batch, H, W, 2] containing grid offsets.
    bounding_boxes: List of Tensors, each of shape [num_boxes, 4] containing normalized
                    coordinates (x_min, y_min, x_max, y_max) for each image in the batch.
    """
    loss = torch.tensor([0.]).to('cuda')
    for i, boxes in enumerate(batch_gt_bboxes):
        current_grid = saliency_list[i].permute(1,2,0)
        # boxes[:,]
        for box in boxes:
            # Convert normalized coords to pixel indices
            x_min = int(box[0] / ori_shape[1] * current_grid.shape[1])
            y_min = int(box[1] / ori_shape[0] * current_grid.shape[0])
            x_max = int(box[2] / ori_shape[1] * current_grid.shape[1])
            y_max = int(box[3] / ori_shape[0] * current_grid.shape[0])
            # bh = y_max - y_min
            # bw = x_max - x_min
            # x_min = x_min - int(bw * 0.2) if x_min - int(bw * 0.2) >= 0 else 0
            # y_min = y_min - int(bh * 0.2) if x_min - int(bh * 0.2) >= 0 else 0
            # x_max = x_max + int(bw * 0.2) if x_max + int(bw * 0.2) <= current_grid.shape[1] else current_grid.shape[1]
            # y_max = y_max + int(bh * 0.2) if y_max + int(bh * 0.2) <= current_grid.shape[0] else current_grid.shape[0]
            # x_min, , x_max, y_max = x_min - int(bw * 0.2), \
            #
            #     y_min - int(bh * 0.2), x_max + int(bw * 0.2), y_max + int(bh * 0.2)
            #


            if y_max<=y_min+1 or x_max<=x_min+1:
                continue
            # Extract the region of the grid corresponding to the current box
            region = current_grid[y_min:y_max, x_min:x_max]

            # Calculate symmetry loss for this region
            W_region = region.shape[1]
            region_flipped_hor = region.flip(dims=[1])  # Flipping W dimension
            half_width = W_region // 2
            left_half_x = region[:, :half_width, 0]
            right_half_x = region_flipped_hor[:, :half_width, 0]
            left_half_y = region[:, :half_width, 1]
            right_half_y = region_flipped_hor[:, :half_width, 1]
            loss += torch.mean((left_half_x + right_half_x).pow(2)) + torch.mean((left_half_y - right_half_y).pow(2))


            # Vertical Symmetry Loss
            H_region = region.shape[0]
            region_flipped_ver = region.flip(dims=[0])  # Flipping H dimension
            half_height = H_region // 2
            top_half_x = region[:half_height, :, 0]
            bottom_half_x = region_flipped_ver[:half_height, :, 0]
            top_half_y = region[:half_height, :, 1]
            bottom_half_y = region_flipped_ver[:half_height, :, 1]
            loss += torch.mean((top_half_x - bottom_half_x).pow(2)) + torch.mean((top_half_y + bottom_half_y).pow(2))

    # Normalize loss by number of bounding boxes
    total_boxes = sum([boxes.shape[0] for boxes in batch_gt_bboxes])
    if total_boxes > 0:
        loss /= total_boxes

    offset_flip_loss_dict = {}
    offset_flip_loss_dict['offset_flip_loss'] = loss*0.1
    if torch.isnan(loss):
        print('except')

    return offset_flip_loss_dict

def foreground_align_loss(grid, batch_transformed_gt_bboxes, batch_gt_bboxes, ori_shapes, rz_shape, fa_weight, mag_weight):
    loss_dict = {}
    fa_loss_all = torch.tensor([0], dtype=torch.float32).to('cuda')
    # mag_loss_all = torch.tensor([0], dtype=torch.float32).to('cuda')

    for i in range(grid.shape[0]):
        ori_img_width = ori_shapes[i][1]
        ori_img_height = ori_shapes[i][0]
        box_count = 0
        fa_loss = 0
        mag_loss = 0
        for box_id in range(batch_transformed_gt_bboxes[i].shape[0]):
            x1_u, y1_u, x2_u, y2_u = batch_gt_bboxes[i].tensor[box_id, :]

            a = (x1_u / ori_img_width) * 2 - 1
            b = (x2_u / (ori_img_width - 1)) * 2 - 1
            c = (y1_u / ori_img_height) * 2 - 1
            d = (y2_u / (ori_img_height - 1)) * 2 - 1
            indices = torch.nonzero(
                (grid[0, :, :, 0] >= a) & (grid[0, :, :, 0] <= b) & (grid[0, :, :, 1] >= c) & (grid[0, :, :, 1] <= d))

            if indices.size(0) > 0:
                y_max = torch.max(indices[:, 0])
                y_min = torch.min(indices[:, 0])
                x_max = torch.max(indices[:, 1])
                x_min = torch.min(indices[:, 1])

                if ((x_max - x_min) > 0) & ((y_max - y_min) > 0):
                    box_count += 1

                    mean_x = torch.mean(grid[i, int(y_min):int(y_max), int(x_min):int(x_max), 0], dim=0)
                    var_x = grid[i, int(y_min):int(y_max), int(x_min):int(x_max), 0] - mean_x.unsqueeze(0)
                    loss_x = torch.sum(torch.abs(var_x)) / (y_max - y_min)

                    mean_y = torch.mean(grid[i, int(y_min):int(y_max), int(x_min):int(x_max), 1], dim=1)
                    var_y = grid[i, int(y_min):int(y_max), int(x_min):int(x_max), 1] - mean_y.unsqueeze(1)
                    loss_y = torch.sum(torch.abs(var_y)) / (x_max - x_min)

                    # if (y_max - y_min) * (x_max - x_min) / ((x2_u - x1_u) * (y2_u - y1_u) * (rz_shape[0] / ori_img_height) * (
                    #             rz_shape[1] / ori_img_width)) < 2:
                    #     left_loss = torch.mean(2 - torch.clamp_min(a - grid[i, :, :, 0], min=0))
                    #     right_loss = torch.mean(2 - torch.clamp_min(grid[i, :, :, 0] - b, min=0))
                    #     up_loss = torch.mean(2 - torch.clamp_min(c - grid[i, :, :, 1], min=0))
                    #     down_loss = torch.mean(2 - torch.clamp_min(grid[i, :, :, 1] - d, min=0))
                    #     mag_loss += left_loss + right_loss + up_loss + down_loss

                    # mag_loss += -torch.log(torch.sigmoid((area_after / area_uniform) - 1))
                    # mag_loss += torch.max(torch.tensor([0], dtype=torch.float32).to('cuda'),
                    #                       (0.73 - torch.sigmoid(area_after / area_uniform - 1)))

                    fa_loss += (loss_x + loss_y)

        if fa_loss > 0:
            fa_loss_all += fa_loss / box_count
            # mag_loss_all += mag_loss / box_count

    loss_dict['fa_loss'] = fa_weight * fa_loss_all / grid.shape[0]
    # loss_dict['mag_loss'] = mag_weight * mag_loss_all / grid.shape[0]

    return loss_dict
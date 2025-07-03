import torch
import torch.nn.functional as F
import math
import cv2

def aspect_ratio_loss(grid, weight):
    aspect_ratio_loss_dict = {}
    ar_loss = 0
    for i in range(grid.shape[0]):
        mean_x = torch.mean(grid[i, :, :, 0], dim=0)
        var_x = grid[i, :, :, 0] - mean_x.unsqueeze(0)
        loss_x = torch.sum(torch.abs(var_x)) / grid.shape[1]

        mean_y = torch.mean(grid[i, :, :, 1], dim=1)
        var_y = grid[i, :, :, 1] - mean_y.unsqueeze(1)
        loss_y = torch.sum(torch.abs(var_y)) / grid.shape[0]

        loss_2 = loss_x + loss_y
        ar_loss += loss_2

    aspect_ratio_loss_dict['ar_loss'] = weight * ar_loss

    return aspect_ratio_loss_dict
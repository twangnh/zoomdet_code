import torch
import torch.nn.functional as F
import math
import cv2
import time

# def torch_log_power(x, M=4.0, b=3.0, a=0):
#     return torch.clamp((torch.log((M+a)/(x+a))), min=0.)**b

def torch_log_power(x, M=10.0, b=3, a=2):
    return torch.clamp((torch.log(M/(x+a))), min=0.)**b


def enlarge_mask(masks):
    # Check if the input mask is valid
    if masks.dim() != 3:
        raise ValueError("The masks should have the shape [H, W, N]")

    # Convert the masks to binary in case they're not
    masks = (masks > 0).float()

    # Prepare to collect enlarged masks
    # enlarged_masks = torch.zeros_like(masks)

    for i in range(masks.size(2)):  # Iterate over the masks
        mask = masks[:, :, i]

        # Find the indices of the non-zero pixels
        indices = mask.nonzero(as_tuple=True)

        if indices[0].numel() == 0:  # No mask found, skip
            continue

        # Calculate the bounding box of the current masked area
        min_y, max_y = indices[0].min(), indices[0].max()
        min_x, max_x = indices[1].min(), indices[1].max()

        # Calculate the size of the bounding box
        height = max_y - min_y + 1
        width = max_x - min_x + 1

        area = height * width
        if area <= 1024:
            enlargement_factor = 3.0
        elif area > 1024 and area <= 9216:
            enlargement_factor = 1.0
        elif area > 9216:
            enlargement_factor = 1.0

        # Calculate enlargement amount
        enlarge_height = int(height * enlargement_factor - height)
        enlarge_width = int(width * enlargement_factor - width)

        # Update the bounding box coordinates based on the enlargement
        min_y = max(0, min_y - enlarge_height // 2)
        max_y = min(mask.size(0) - 1, max_y + enlarge_height // 2)
        min_x = max(0, min_x - enlarge_width // 2)
        max_x = min(mask.size(1) - 1, max_x + enlarge_width // 2)

        # Update the mask for this item
        masks[min_y:max_y + 1, min_x:max_x + 1, i] = 1

    return masks

def map_area2weight(x):
    return 1 - 0.9 * (x - 1) / 1023

def saliency_loss(saliency_list, grid, uniform_grid, ori_gt_bboxes, ori_shapes, saliency_weight, small_th, medium_th, large_th, saliency_loss_mode='box_wise_loss'):
    ## box_wise_loss, global_loss, global_loss_sml
    saliency_loss_dict = {}
    saliency_reg_loss_dict = {}
    saliency_loss = 0
    device = grid.device
    h_ori, w_ori = ori_shapes[0], ori_shapes[1]
    # yc = torch.linspace(0, h_ori, h_ori).to(device)
    # xc = torch.linspace(0, w_ori, w_ori).to(device)
    # yc, xc = torch.meshgrid(yc, xc)
    # xc = xc.unsqueeze(-1)
    # yc = yc.unsqueeze(-1)

    h_det, w_det = grid.shape[1], grid.shape[2]
    yc = torch.linspace(0, h_det, h_det).to(device)
    xc = torch.linspace(0, w_det, w_det).to(device)
    yc, xc = torch.meshgrid(yc, xc)
    xc = xc.unsqueeze(-1)
    yc = yc.unsqueeze(-1)

    scaling_factor = h_ori/h_det
    ori_gt_bboxes = [o/scaling_factor for o in ori_gt_bboxes]

    for i in range(grid.shape[0]):
        x1, y1, x2, y2 = ori_gt_bboxes[i].permute(1, 0).split(1)
        # 扩展维度，以便进行广播
        x1 = x1.unsqueeze(1)
        y1 = y1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        y2 = y2.unsqueeze(1)

        # 检查是否在目标框范围内
        gt_mask = ((xc >= x1) & (xc <= x2) & (yc >= y1) & (yc <= y2)) * 1.
        # area_indice = (x2 - x1) * (y2 - y1) < 9216
        # if torch.any(area_indice):
        #     gt_mask = gt_mask[:, :, area_indice.squeeze(0).squeeze(0)]

            # # ######################## 可视化 ##########################
            # saliency_learned = (saliency_gt.unsqueeze(0) * 255.).clamp(0, 255).byte().contiguous()
            # image_3 = saliency_learned.permute(1, 2, 0).cpu().numpy()
            # # 保存绘制好的图像到指定文件夹
            # image_3 = cv2.resize(image_3, tuple((rz_shape[1], rz_shape[0])))
            # output_path = '/root/autodl-tmp/saliency_gt.jpg'
            # cv2.imwrite(output_path, image_3)

        keep_inds = gt_mask.sum(dim=(0,1))!=0
        gt_mask = gt_mask[:, :, keep_inds]
        x1, y1, x2, y2 = x1[:, :, keep_inds], y1[:, :, keep_inds], x2[:, :, keep_inds], y2[:, :, keep_inds]

        # gt_mask = enlarge_mask(gt_mask)

        small_indice = (x2 - x1) * (y2 - y1) <= 1024
        medium_indice = ((x2 - x1) * (y2 - y1) > 1024) & ((x2 - x1) * (y2 - y1) <= 9216)
        large_indice = (x2 - x1) * (y2 - y1) > 9216

        if saliency_loss_mode == 'box_wise_loss':
            sampled_saliency = F.grid_sample(gt_mask.permute(2, 0, 1).unsqueeze(0), grid[i].unsqueeze(0),
                                             padding_mode='reflection')
            # uniform_saliency = F.grid_sample(gt_mask.permute(2, 0, 1).unsqueeze(0), uniform_grid,
            #                                  padding_mode='reflection')

            uniform_saliency = gt_mask.permute(2,0,1).unsqueeze(0)

            sampled_point_count = sampled_saliency.sum(dim=(2, 3))
            uniform_point_count = uniform_saliency.sum(dim=(2, 3))
            magnification = sampled_point_count / uniform_point_count

            loss_all = []
            saliency_pred = saliency_list[i].permute(1, 2, 0)
            mag_reg_loss = saliency_pred.abs().mean()
            # mag_reg_loss_m = torch.tensor([0.]).to('cuda')
            # mag_reg_loss_l = torch.tensor([0.]).to('cuda')

            if torch.any(small_indice):
                # mag_loss_s = torch.max(-torch.log(torch.sigmoid(magnification[small_indice.squeeze(0)]-1)) - small_th,
                #                                                  torch.tensor([0.]).to('cuda'))
                mag_loss_s = torch_log_power(magnification[small_indice.squeeze(0)])

                # weight = map_area2weight(uniform_point_count[small_indice.squeeze(0)])
                # mag_loss_s = mag_loss_s * weight

                loss_all.append(mag_loss_s)

                if torch.isnan(mag_loss_s).any() or torch.isinf(mag_loss_s).any():
                    print('except')


            # if torch.any(medium_indice):
            #     # mag_loss_m = torch.max(-torch.log(torch.sigmoid(magnification[medium_indice.squeeze(0)]-1)) - medium_th,
            #     #                                   torch.tensor([0.]).to('cuda'))
            #
            #     mag_loss_m = torch_log_power(magnification[medium_indice.squeeze(0)])
            #     loss_all.append(mag_loss_m)
            #
            #     if torch.isnan(mag_loss_m).any() or torch.isinf(mag_loss_m).any():
            #         print('except')
            #
            #
            # if torch.any(large_indice):
            #     # mag_loss_l = torch.max(-torch.log(torch.sigmoid(magnification[large_indice.squeeze(0)]-1)) - large_th,
            #     #                                   torch.tensor([0.]).to('cuda'))
            #
            #     mag_loss_l = torch_log_power(magnification[large_indice.squeeze(0)])
            #     loss_all.append(mag_loss_l)
            #
            #     if torch.isnan(mag_loss_l).any() or torch.isinf(mag_loss_l).any():
            #         print('except')

            # if torch.any(medium_indice):
            #
            #     gt_mask_medium = gt_mask[:, :, medium_indice.squeeze(0).squeeze(0)].permute(2, 0, 1).unsqueeze(0)
            #     gt_mask_interp = F.interpolate(gt_mask_medium, size=tuple(saliency_pred.shape[:2]), mode='bilinear', align_corners=True)
            #     gt_mask_interp = gt_mask_interp.squeeze(0).sum(0).bool()  ## sum over all gt
            #     saliency_pred_foreground = saliency_pred[gt_mask_interp]
            #     mag_reg_loss_m += saliency_pred_foreground.abs().mean()
            #
            # if torch.any(large_indice):
            #
            #     gt_mask_large = gt_mask[:, :, large_indice.squeeze(0).squeeze(0)].permute(2,0,1).unsqueeze(0)
            #     gt_mask_interp = F.interpolate(gt_mask_large, size=tuple(saliency_pred.shape[:2]), mode='bilinear', align_corners=True)
            #     gt_mask_interp = gt_mask_interp.squeeze(0).sum(0).bool() ## sum over all gt
            #     saliency_pred_foreground = saliency_pred[gt_mask_interp]
            #     mag_reg_loss_l += saliency_pred_foreground.abs().mean()

            if len(loss_all)>0:
                saliency_loss += torch.mean(torch.cat(loss_all))
            else:
                saliency_loss = torch.tensor([0.]).to('cuda')

            # saliency_loss += torch.mean(mag_loss_s)
            # saliency_loss += torch.mean(torch.cat((mag_loss_s, mag_loss_m)))

            # if saliency_loss>0.04:
            #     print('small th is {}'.format(small_th))
            #     print('med th is {}'.format(medium_th))
            #     print('large th is {}'.format(large_th))

        elif saliency_loss_mode == 'global_loss':

            gt_mask_combine = (gt_mask.sum(-1)>=1).float()

            sampled_saliency_ = F.grid_sample(gt_mask_combine.unsqueeze(0).unsqueeze(0), grid[i].unsqueeze(0),
                                              padding_mode='reflection')
            uniform_saliency_ = F.grid_sample(gt_mask_combine.unsqueeze(0).unsqueeze(0), uniform_grid,
                                              padding_mode='reflection')
            sampled_point_count_ = sampled_saliency_.sum(dim=(2, 3))
            uniform_point_count_ = uniform_saliency_.sum(dim=(2, 3))
            magnification_ = sampled_point_count_ / uniform_point_count_

            # mag_loss = torch.max(-torch.log(torch.sigmoid(magnification_ - 1)) - large_th,
            #                        torch.tensor([0.]).to('cuda'))
            mag_loss = torch_log_power(magnification_)
            saliency_loss += mag_loss

        elif saliency_loss_mode == 'global_loss_sml':
            loss_all = []
            if torch.any(small_indice):
                gt_mask_s = gt_mask[:,:, small_indice.squeeze(0).squeeze(0)]
                gt_mask_s_combine = (gt_mask_s.sum(-1) >= 1).float()
                sampled_saliency_ = F.grid_sample(gt_mask_s_combine.unsqueeze(0).unsqueeze(0), grid[i].unsqueeze(0),
                                                  padding_mode='reflection')
                # uniform_saliency_ = F.grid_sample(gt_mask_s.sum(-1).unsqueeze(0).unsqueeze(0), uniform_grid,
                #                                   padding_mode='reflection')
                uniform_saliency_ = gt_mask_s_combine.unsqueeze(0).unsqueeze(0)

                sampled_point_count_ = sampled_saliency_.sum(dim=(2, 3))
                uniform_point_count_ = uniform_saliency_.sum(dim=(2, 3))

                magnification_ = sampled_point_count_ / uniform_point_count_

                # mag_loss_s = torch.max(-torch.log(torch.sigmoid(magnification_ - 1)) - small_th,
                #                                                  torch.tensor([0.]).to('cuda'))

                mag_loss_s = torch_log_power(magnification_)
                loss_all.append(mag_loss_s)

                if torch.isnan(mag_loss_s).any() or torch.isinf(mag_loss_s).any():
                    print('except')

            if torch.any(medium_indice):
                gt_mask_m = gt_mask[:,:, medium_indice.squeeze(0).squeeze(0)]
                gt_mask_m_combine = (gt_mask_m.sum(-1) >= 1).float()
                sampled_saliency_ = F.grid_sample(gt_mask_m_combine.unsqueeze(0).unsqueeze(0), grid[i].unsqueeze(0),
                                                      padding_mode='reflection')
                # uniform_saliency_ = F.grid_sample(gt_mask_m.sum(-1).unsqueeze(0).unsqueeze(0), uniform_grid,
                #                                   padding_mode='reflection')
                uniform_saliency_ = gt_mask_m_combine.unsqueeze(0).unsqueeze(0)

                sampled_point_count_ = sampled_saliency_.sum(dim=(2, 3))
                uniform_point_count_ = uniform_saliency_.sum(dim=(2, 3))

                magnification_ = sampled_point_count_ / uniform_point_count_

                # mag_loss_m = torch.max(-torch.log(torch.sigmoid(magnification_ - 1)) - medium_th,
                #                                                  torch.tensor([0.]).to('cuda'))

                mag_loss_m = torch_log_power(magnification_)
                loss_all.append(mag_loss_m)

                if torch.isnan(mag_loss_m).any() or torch.isinf(mag_loss_m).any():
                    print('except')

            if torch.any(large_indice):
                gt_mask_l = gt_mask[:,:, large_indice.squeeze(0).squeeze(0)]
                gt_mask_l_combine = (gt_mask_l.sum(-1) >= 1).float()
                sampled_saliency_ = F.grid_sample(gt_mask_l_combine.unsqueeze(0).unsqueeze(0), grid[i].unsqueeze(0),
                                                  padding_mode='reflection')
                # uniform_saliency_ = F.grid_sample(gt_mask_l.sum(-1).unsqueeze(0).unsqueeze(0), uniform_grid,
                #                                   padding_mode='reflection')

                uniform_saliency_ = gt_mask_l_combine.unsqueeze(0).unsqueeze(0)

                sampled_point_count_ = sampled_saliency_.sum(dim=(2, 3))
                uniform_point_count_ = uniform_saliency_.sum(dim=(2, 3))
                magnification_ = sampled_point_count_ / uniform_point_count_

                # mag_loss_l = torch.max(-torch.log(torch.sigmoid(magnification_ - 1)) - large_th,
                #                                                  torch.tensor([0.]).to('cuda'))

                mag_loss_l = torch_log_power(magnification_)
                loss_all.append(mag_loss_l)

                if torch.isnan(mag_loss_l).any() or torch.isinf(mag_loss_l).any():
                    print('except')

            if len(loss_all)>0:
                saliency_loss += torch.mean(torch.cat(loss_all))
            else:
                saliency_loss = torch.tensor([0.]).to('cuda')

    saliency_loss_dict['saliency_loss'] = saliency_weight * 0.2 * saliency_loss / grid.shape[0]
    # saliency_loss_dict['saliency_reg_loss'] = 0.002*mag_reg_loss

    return saliency_loss_dict
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .saliency_network import saliency_network_resnet18, fov_simple
import random
from torch_cluster import nearest, knn
from torchvision import transforms
import cv2
import time
from math import floor, ceil

def time_synchronized(t1=None, m=None):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.time()
    if m is not None and t1 is not None:
        print('timing {} is {}'.format(m, t2 - t1))
    return t2

def make1DGaussian(size, fwhm=3, center=None):
    """ Make a 1D gaussian kernel.

    size is the length of the kernel,
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = np.arange(0, size, 1, dtype=np.float)

    if center is None:
        center = size // 2

    return np.exp(-4*np.log(2) * (x-center)**2 / fwhm**2)

def make2DGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class NearestInterpolator_Torch(nn.Module):

    def __init__(self, points, values):
        super(NearestInterpolator_Torch, self).__init__()
        self.points = points
        self.values = values

    def __call__(self, points_q):
        try:
            nn_torch = nearest(points_q, self.points)
        except:
            print('xxxxx')
        return self.values[nn_torch]

    # def __call__(self, points_q, n_nearest = 2):
    #     knn_torch = knn(self.points, points_q, n_nearest)
    #     sample_inds = knn_torch[1, :].view(-1, n_nearest)
    #     return self.values[sample_inds].mean(-1)

class GridGenerator(nn.Module):
    def __init__(self, sep_fwhm=250, nonsep_fwhm=10, grid_shape=(640, 640), separable=True,
                 attraction_fwhm=10, anti_crop=True, **kwargs):
        super(GridGenerator, self).__init__()
        self.using_gt = False
        self.saliency_network = saliency_network_resnet18()
        self.grid_shape = grid_shape
        self.padding_size = 10
        # self.sep_padding_size = 500
        # self.total_shape = tuple(
        #     dim+2*self.padding_size
        #     for dim in self.grid_shape
        # )
        self.padding_mode = 'reflect' if anti_crop else 'replicate'
        self.separable = separable
        self.sep_padding_size = 650
        self.sep_filter = make1DGaussian(
            2 * self.sep_padding_size + 1, fwhm=sep_fwhm)
        self.sep_filter = torch.FloatTensor(self.sep_filter).unsqueeze(0) \
            .unsqueeze(0).cuda()


        # self.sep_total_shape = tuple(
        #     dim + 2 * self.sep_padding_size
        #     for dim in self.grid_shape
        # )
        #
        # self.P_basis_x = torch.zeros(self.sep_total_shape[1])
        # for i in range(self.sep_total_shape[1]):
        #     self.P_basis_x[i] = \
        #         (i - self.sep_padding_size) / (self.grid_shape[1] - 1.0)
        # self.P_basis_y = torch.zeros(self.sep_total_shape[0])
        # for i in range(self.sep_total_shape[0]):
        #     self.P_basis_y[i] = \
        #         (i - self.sep_padding_size) / (self.grid_shape[0] - 1.0)


        self.filter = make2DGaussian(
            2 * self.padding_size + 1, fwhm=nonsep_fwhm)
        self.filter = torch.FloatTensor(self.filter) \
            .unsqueeze(0).unsqueeze(0).cuda()

        # self.total_shape = tuple(
        #     dim + 2 * self.padding_size
        #     for dim in self.grid_shape
        # )

        # self.P_basis = torch.zeros(2, *self.total_shape)
        # for k in range(2):
        #     for i in range(self.total_shape[0]):
        #         for j in range(self.total_shape[1]):
        #             self.P_basis[k, i, j] = k * (i - self.padding_size) / (self.grid_shape[0] - 1.0) + (1.0 - k) * (
        #                     j - self.padding_size) / (self.grid_shape[1] - 1.0)  # noqa: E501

    # def deformable_grid_generate(self, saliency, rz_shape, device):
    #     uniform_grid_y, uniform_grid_x = torch.meshgrid(torch.arange(saliency.shape[2], device=device)+0.5,
    #                                         torch.arange(saliency.shape[3], device=device)+0.5)
    #
    #     uniform_grid_x = uniform_grid_x.float()
    #     uniform_grid_y = uniform_grid_y.float()
    #
    #     # offset_confidence = torch.sigmoid(saliency[:, 2, :, :])
    #     final_grid = torch.zeros(saliency.shape[0], 2, saliency.shape[2], saliency.shape[3],  device=device)
    #     # final_grid[:, 0, :, :] = (uniform_grid_x + saliency[:, 0, :, :] * offset_confidence) / (saliency.shape[3] - 1)
    #     # final_grid[:, 1, :, :] = (uniform_grid_y + saliency[:, 1, :, :] * offset_confidence) / (saliency.shape[2] - 1)
    #     final_grid[:, 0, :, :] = (uniform_grid_x + saliency[:, 0, :, :]) / (saliency.shape[3] - 1)
    #     final_grid[:, 1, :, :] = (uniform_grid_y + saliency[:, 1, :, :]) / (saliency.shape[2] - 1)
    #
    #
    #     uniform_grid = torch.cat((uniform_grid_x.unsqueeze(0) / (saliency.shape[3] - 1),
    #                               uniform_grid_y.unsqueeze(0) / (saliency.shape[2] - 1))).unsqueeze(0)
    #     # uniform_grid = torch.cat((2 * uniform_grid_x.unsqueeze(0) / saliency.shape[3] - 1,
    #     #                           2 * uniform_grid_y.unsqueeze(0) / saliency.shape[2] - 1)).unsqueeze(0)
    #
    #
    #     # final_grid = torch.clamp(final_grid * 2 - 1, min=-1, max=1)
    #     # uniform_grid = torch.clamp(uniform_grid * 2 - 1, min=-1, max=1)
    #
    #     final_grid = F.interpolate(final_grid, size=rz_shape, mode='bilinear', align_corners=True)
    #     uniform_grid = F.interpolate(uniform_grid, size=rz_shape, mode='bilinear', align_corners=True)
    #
    #     return final_grid.permute(0, 2, 3, 1), uniform_grid.permute(0, 2, 3, 1)

    def deformable_grid_generate(self, saliency, rz_shape, device):
        uniform_grid_y, uniform_grid_x = torch.meshgrid(torch.arange(rz_shape[0], device=device)+0.5,
                                            torch.arange(rz_shape[1], device=device)+0.5)

        uniform_grid_x = uniform_grid_x.float()
        uniform_grid_y = uniform_grid_y.float()

        # offset_confidence = torch.sigmoid(saliency[:, 2, :, :])
        final_grid = torch.zeros(saliency.shape[0], 2, rz_shape[0], rz_shape[1],  device=device)
        saliency = F.interpolate(saliency, size=rz_shape, mode='bilinear', align_corners=True)
        # final_grid[:, 0, :, :] = (uniform_grid_x + saliency[:, 0, :, :] * offset_confidence) / (saliency.shape[3] - 1)
        # final_grid[:, 1, :, :] = (uniform_grid_y + saliency[:, 1, :, :] * offset_confidence) / (saliency.shape[2] - 1)
        # final_grid[:, 0, :, :] = (uniform_grid_x + saliency[:, 0, :, :]) / (saliency.shape[3] - 1)
        # final_grid[:, 1, :, :] = (uniform_grid_y + saliency[:, 1, :, :]) / (saliency.shape[2] - 1)
        final_grid[:, 0, :, :] = 2 * (uniform_grid_x + saliency[:, 0, :, :]) / saliency.shape[3] - 1
        final_grid[:, 1, :, :] = 2 * (uniform_grid_y + saliency[:, 1, :, :]) / saliency.shape[2] - 1


        # uniform_grid = torch.cat((uniform_grid_x.unsqueeze(0) / (saliency.shape[3] - 1),
        #                           uniform_grid_y.unsqueeze(0) / (saliency.shape[2] - 1))).unsqueeze(0)
        uniform_grid = torch.cat((2 * uniform_grid_x.unsqueeze(0) / saliency.shape[3] - 1,
                                  2 * uniform_grid_y.unsqueeze(0) / saliency.shape[2] - 1)).unsqueeze(0)

        return final_grid.permute(0, 2, 3, 1), uniform_grid.permute(0, 2, 3, 1)

    def forward(self, x, rz_shape):
        # low_shape = (int(ori_shape[0] / 8), int(ori_shape[1] / 8))
        # x_low = nn.AdaptiveAvgPool2d(low_shape)(x)
        # start_time = time_synchronized()
        saliency = self.saliency_network(x)
        # saliency*=0
        # end_time = time_synchronized(start_time, 'saliency_network')
        device = x.device
        # start_time_1 = time_synchronized()
        grid, uniform_grid = self.deformable_grid_generate(saliency, rz_shape, device)
        # end_time = time_synchronized(start_time_1, 'generate_grid')

        # start_time_2 = time_synchronized()
        # (N, output_shape, output_shape, 2) to (2, N, output_shape, output_shape)
        grid_reorder = grid.permute(3, 0, 1, 2).detach()
        # (N, 800*800)
        # if self.training:
        u_cor = ((grid_reorder[0, :, :, :] + 1) / 2).view(
            grid_reorder.shape[1], -1)
        v_cor = ((grid_reorder[1, :, :, :] + 1) / 2).view(
            grid_reorder.shape[1], -1)

        ##改到0 H-1范围， 原来是0.5 H-0.5范围
        u_cor, v_cor = u_cor-1/grid_reorder.shape[3]/2., v_cor-1/grid_reorder.shape[2]/2.

        # (800*800)
        x_cor = torch.arange(0, grid_reorder.shape[3], device=grid_reorder.device).unsqueeze(0).expand(
            (grid_reorder.shape[2], grid_reorder.shape[3])).reshape(-1)
        # (N, 800*800)
        x_cor = x_cor.unsqueeze(0).expand(u_cor.shape[0], -1).float()

        y_cor = torch.arange(0, grid_reorder.shape[2], device=grid_reorder.device).unsqueeze(-1).expand(
            (grid_reorder.shape[2], grid_reorder.shape[3])).reshape(-1)
        y_cor = y_cor.unsqueeze(0).expand(u_cor.shape[0], -1).float()

        # x_cor, y_cor = x_cor+0.5, y_cor+0.5
        # v:h, u:w
        # (N, 1, 800*800)
        u_cor, v_cor = u_cor.unsqueeze(1), v_cor.unsqueeze(1)
        # every line in u is (1, u_cor, v_cor)
        # every line in v is (0, u_cor, v_cor)
        u = torch.cat([torch.zeros(u_cor.shape, device=u_cor.device, dtype=u_cor.dtype), u_cor, v_cor], dim=1)
        v = torch.cat([torch.ones(u_cor.shape, device=u_cor.device, dtype=u_cor.dtype), u_cor, v_cor], dim=1)

        # (N, 3, 2*800*800) -> (N, 2*800*800, 3)
        points = torch.cat([u, v], dim=2).transpose(2, 1).float()

        # range from (0,1)
        # y_cor /= grid.shape[1]
        # x_cor /= grid.shape[2]

        # (N, 2*800*800)
        values = torch.cat([x_cor, y_cor], dim=1)

        # range from (0,1)
        # points[:, :, 1:] /= torch.Tensor([rz_shape[1], rz_shape[0]]).to(points.device)

        b, n, dim = points.shape
        # (N, 2*800*800)
        if b == 1:
            batch_inds = torch.zeros((1, n)).to(points.device)
        else:
            batch_inds = torch.arange(b).unsqueeze(1).expand(-1, n).to(points.device)
        # (2*800*800*N, 4), every column means (img_id, u or v, u_cor, v_cor)
        batch_points = torch.cat([batch_inds.view(-1).unsqueeze(1), points.contiguous().view(-1, 3)], dim=1)

        interpolator_all = NearestInterpolator_Torch(batch_points, values.view(-1))
        # end_time = time_synchronized(start_time_2, 'generate_interpolator')

        return grid, uniform_grid, interpolator_all, saliency
        # else:
        #     return grid, saliency

def make_grid_generator(sep_fwhm, nonsep_fwhm):
    model = GridGenerator(sep_fwhm=sep_fwhm, nonsep_fwhm=nonsep_fwhm)
    # for name, param in model.named_parameters():
    #     param.requires_grad = False   # freeze for the first m epoch

    return model

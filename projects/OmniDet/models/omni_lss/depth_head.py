from tkinter import W
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


class DepthNet(nn.Module):
    def __init__(self,
                 in_channel: int,
                 layer_nums: Sequence[int] = [2, 2, 2],
                 num_filters: Sequence[int] = [64, 128, 256],
                 layer_strides: Sequence[int] = [2, 2, 2],
                 **kwargs):
        super().__init__()

        in_channels = [in_channel, *num_filters[:-1]]

        # deconv
        num_up_in_channels = num_filters[1:]
        num_up_in_channels.reverse()
        num_up_filters = num_filters[:-1]
        num_up_filters.reverse()
        up_strides = layer_strides[1:]
        up_strides.reverse()

        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for i in range(len(layer_nums)):
            cur_layers = [
                nn.Conv3d(in_channels[i], num_filters[i],
                          3, layer_strides[i], 1, bias=False),
                nn.ReLU(inplace=True)
            ]
            for _ in range(layer_nums[i]):
                cur_layers.extend([
                    nn.Conv3d(num_filters[i],
                              num_filters[i], 3, 1, 1, bias=False),
                    nn.ReLU(inplace=True)
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

        for i in range(len(layer_nums)-1):
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose3d(
                    num_up_in_channels[i], num_up_filters[i],
                    kernel_size=3, stride=up_strides[i], padding=1, bias=False
                ),
                nn.ReLU()
            ))

        self.depth_head = nn.ConvTranspose3d(
            num_up_filters[-1], 1, kernel_size=3, stride=2, padding=1, bias=False
        )
    
    def forward(self, x):
        x_down = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)  # 256
            if i < len(self.blocks) - 1:
                x_down.append(x)

        x_down.reverse()  # 128, 64
        x_down_size = [x.size() for x in x_down]
        for i in range(len(self.deblocks)):
            x = self.deblocks[i][0](
                x, output_size=x_down_size[i])  # transpose conv
            x = self.deblocks[i][1:](x)
            x = x + x_down[i]
        dd, dh, dw = x.size()[2:]
        output_size = torch.Size([2 * dd, 2 * dh, 2 * dw])
        x = self.depth_head(x, output_size=output_size)  # BN, 1, D, H, W
        x = x.squeeze().softmax(dim=1)  # BN, D, H, W
        return x


@MODELS.register_module()
class OmniDepthHead(nn.Module):
    def __init__(self,
                 dbound: Tuple[float, float, float],
                 feature_size: Tuple[int, int],
                 padding_size: Tuple[int, int],
                 img_key: str = 'cam_fisheye',
                 elevation_range: Tuple[float, float] = (-torch.pi / 4, torch.pi / 4),
                 in_channel: int = 96,
                 layer_nums: Sequence[int] = [2, 2, 2],
                 num_filters: Sequence[int] = [64, 128, 256],
                 layer_strides: Sequence[int] = [2, 2, 2],
                 loss_depth: dict = dict(type='mmdet.SmoothL1Loss', reduction='mean', loss_weight=0.2),
                 **kwargs):
        super().__init__()

        min_depth, max_depth, depth_interval = dbound
        self.dbound = dbound
        self.register_buffer('depth_bins', torch.arange(
            min_depth, max_depth, depth_interval))

        self.img_key = img_key
        self.feature_size = feature_size
        self.padding_size = padding_size
        self.elevation_range = elevation_range
        self.depthnet = DepthNet(
            in_channel, layer_nums, num_filters, layer_strides)

        self.loss_depth = MODELS.build(loss_depth)

    def feature_padding(self, x):
        # Compute the padding size for height and width
        pad_h = self.padding_size[0] - self.feature_size[0]
        pad_w = self.padding_size[1] - self.feature_size[1]
        
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Padding size must be larger than or equal to feature size.")
        
        # Pad in (height, width) dimension
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Apply padding using F.pad (pad format: [left, right, top, bottom])
        x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return x_padded

    def feature_cropping(self, x):
        # Compute the cropping size for height and width
        pad_h = self.padding_size[0] - self.feature_size[0]
        pad_w = self.padding_size[1] - self.feature_size[1]
        
        if pad_h < 0 or pad_w < 0:
            raise ValueError("Padding size must be larger than or equal to feature size.")
        
        # Calculate cropping indices
        crop_top = pad_h // 2
        crop_bottom = self.padding_size[0] - crop_top - self.feature_size[0]
        crop_left = pad_w // 2
        crop_right = self.padding_size[1] - crop_left - self.feature_size[1]

        # Perform cropping using slicing
        x_cropped = x[:, :, crop_top:-crop_bottom if crop_bottom > 0 else None, crop_left:-crop_right if crop_right > 0 else None]
        return x_cropped
    
    def equirectangular_projection(self, ori_points, lidar2cam):
        '''
            Attention: This function is not implemented for any data augmentation.
            points (list): batch of points, each point is a tensor of shape (num_points, 3).
            lidar2cam (tensor): lidar2cam matrix, shape (B, num_cam, 4, 4).
        '''
        def cartesian_to_spherical(x, y, z):
            r = torch.sqrt(x**2 + y**2 + z**2)
            theta = torch.arctan2(x, z)  # Azimuth
            phi = torch.arctan2(y, torch.sqrt(x**2 + z**2))  # Elevation
            return r, theta, phi
        
        B, num_cam, _, _ = lidar2cam.shape
        
        height, width = self.feature_size
        total_depth_maps = []

        # phi_min, phi_max = -np.pi / 2, np.pi / 2  # 完整仰角范围
        # 根据height动态调整仰角范围
        # phi_range = (phi_max - phi_min) * (height / (width / 2))  # width/2是全高度时的基准
        # phi_min, phi_max = -phi_range / 2, phi_range / 2  # 新仰角范围

        phi_min, phi_max = self.elevation_range
        for b in range(B):
            points_b = ori_points[b].clone()
            points_b = torch.cat([points_b, torch.ones(points_b.shape[0], 1).to(points_b.device)], dim=1)
            batch_depth_maps = []

            for n in range(num_cam):
                proj_matrix = lidar2cam[b, n]
                points = torch.matmul(proj_matrix, points_b.T).T
                r, theta, phi = cartesian_to_spherical(
                    points[:, 0], points[:, 1], points[:, 2])
                
                valid_mask = (phi >= phi_min) & (phi <= phi_max)
                r, theta, phi = r[valid_mask], theta[valid_mask], phi[valid_mask]
                u = ((theta + torch.pi) / (2 * torch.pi) * width).long()
                v = ((phi - phi_min) / (phi_max - phi_min) * height).long()

                # u = ((theta + torch.pi) / (2 * torch.pi) * width).long()
                # v = ((phi + np.pi / 2) / np.pi * height).long()

                u = torch.clamp(u, 0, width - 1)
                v = torch.clamp(v, 0, height - 1)

                depth_map = torch.full((height, width), float('inf')).to(points.device)
                depth_map[v, u] = torch.minimum(depth_map[v, u], r)
                depth_map[depth_map == float('inf')] = 0
                batch_depth_maps.append(depth_map)
            
            batch_depth_maps = torch.stack(batch_depth_maps)
            total_depth_maps.append(batch_depth_maps)
        
        total_depth_maps = torch.stack(total_depth_maps)
        return total_depth_maps

    def depth_regression(self, depth_prob):
        '''
            Args:
                depth_prob (torch.Tensor): The features of the image. Shape: (BN, D, H, W)
            return:
                depth_map (torch.Tensor): The depth map. Shape: (BN, H, W)
        '''
        depth_map = torch.sum(depth_prob * self.depth_bins.view(1, -1, 1, 1), dim=1)
        return depth_map
    
    def get_fov_mask(self, img_feat):
        '''
            Args:
                img_feat (torch.Tensor): The features of the image. Shape: (BN, C, D, H, W)
            return:
                mask (torch.Tensor): The mask of the fov. Shape: (H, W)
        '''
        mask = img_feat[0].sum(0).sum(0) != 0
        return mask
    
    def loss(self,
             img_feat: torch.Tensor,
             points: torch.Tensor,
             data_samples: list,):
        '''
            Args:
                img_feat (): Features in a batch.
                points (): The points in a batch.
                data_samples (List[:obj:`Det3DDataSample`]): The Data
                    Samples. It usually includes information such as
                    `gt_instance_3d`.
        '''
        batch_input_metas = [item.metainfo for item in data_samples]
        lidar2cam = []

        for i, meta in enumerate(batch_input_metas):
            lidar2cam.append(meta[self.img_key]['lidar2cam'])

        lidar2cam = img_feat.new_tensor(np.asarray(lidar2cam))
        B, N = lidar2cam.shape[:2]
        gt_depth_maps = self.equirectangular_projection(points, lidar2cam)  # B, N, H, W
        # gH, gW = gt_depth_maps.shape[2:]
        # gt_depth_maps = gt_depth_maps.view(B*N, 1, gH, gW)
        # gt_depth_maps = self.feature_padding(gt_depth_maps)
        # gt_depth_maps = gt_depth_maps.view(B, N, self.padding_size[0], self.padding_size[1])


        B, N, D, pH, pW, C = img_feat.shape
        img_feat = img_feat.view(B * N, D, pH, pW, C).permute(0, 4, 1, 2, 3) # BN, C, D, pH, pW
        img_feat = self.feature_padding(img_feat)
        depth_prob = self.depthnet(img_feat)  # BN, D, H, W
        pred_depth_maps = self.depth_regression(depth_prob).unsqueeze(1)
        pred_depth_maps = self.feature_cropping(pred_depth_maps)
        pred_depth_maps = pred_depth_maps.view(B, N, pH, pW)

        fov_mask = self.get_fov_mask(img_feat)
        total_loss = 0
        for b in range(B):
            for n in range(N):
                pred_depth = pred_depth_maps[b, n]
                gt_depth = gt_depth_maps[b, n]
                valid_mask = (gt_depth > self.dbound[0]) & (gt_depth < self.dbound[1])  # [0.5, 48.5]
                valid_mask = valid_mask & fov_mask
                pred_depth = pred_depth[valid_mask]
                gt_depth = gt_depth[valid_mask]
                loss = self.loss_depth(pred_depth, gt_depth)
                total_loss += loss
        
        total_loss /= (B * N)
        # import matplotlib.pyplot as plt
        # img_show = img_feat[0].sum(0).sum(0).detach().cpu().numpy()
        # plt.figure('img')
        # plt.imshow(img_show, cmap='jet')
        # plt.show()
        # depth_show = depth_maps[0].detach().cpu().numpy()
        # direction = ['front', 'left', 'right', 'rear']
        # for i in range(4):
        #     plt.figure(direction[i])
        #     plt.imshow(depth_show[i], cmap='jet')
        # plt.show()
        return dict(loss_depth=total_loss)

    def forward(self,
                x: torch.Tensor,
                points: torch.Tensor,
                data_samples: list):
        
        pass

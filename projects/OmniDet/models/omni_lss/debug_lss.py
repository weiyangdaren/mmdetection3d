from typing import Tuple

import torch
from torch import nn

from mmdet3d.registry import MODELS
from .fisheye_lss import gen_dx_bx, BaseViewTransform, BaseDepthTransform
from .ops import bev_pool
from ...utils import draw_scenes_v2



class DepthLSSTransformDebug(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        """Compared with `LSSTransform`, `DepthLSSTransform` adds sparse depth
        information from lidar points into the inputs of the `depthnet`."""
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 96, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1),
        )
        self.depthnet1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    
    def depth_map_to_depth_prob(self, depth_map):
        '''
        depth_map: (N, 1, H, W)
        '''

        # Generate depth levels (D,)
        depth_levels = torch.arange(
            self.dbound[0], self.dbound[1], self.dbound[2], device=depth_map.device)
        D = depth_levels.shape[0]  # Number of depth levels
    
        # Compute the closest depth level index for each pixel in depth_map
        # Shape of depth_map: (N, 1, H, W)
        # Shape of depth_levels: (1, D, 1, 1) for broadcasting
        depth_levels = depth_levels.view(1, D, 1, 1)
        depth_indices = torch.argmin(torch.abs(depth_map - depth_levels), dim=1)  # Shape (N, H, W)
        
        # Create one-hot encoding based on depth_indices
        N, H, W = depth_map.shape[0], depth_map.shape[2], depth_map.shape[3]
        depth_prob = torch.zeros((N, D, H, W), device=depth_map.device)
        depth_prob.scatter_(1, depth_indices.unsqueeze(1), 1)
        
        depth_prob[depth_prob==0] = 0.000  # Avoid division by zero


        return depth_prob
    
    def depth_prob_to_depth_map(self, depth_prob):
        '''
        depth_prob: (N, D, H, W)
        '''

        # Generate depth levels (D,)
        depth_levels = torch.arange(
            self.dbound[0], self.dbound[1], self.dbound[2], device=depth_prob.device)
        D = depth_levels.shape[0]

        # Ensure depth_prob is of shape (N, D, H, W)
        assert depth_prob.shape[1] == D, "Depth levels and probability dimensions do not match"
        
        # Reshape depth_levels for broadcasting: (1, D, 1, 1)
        depth_levels = depth_levels.view(1, D, 1, 1)
        
        # Compute the weighted sum along the depth dimension
        depth_map = torch.sum(depth_prob * depth_levels, dim=1, keepdim=True)  # Shape (N, 1, H, W)
        
        return depth_map

    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        depth_prob = self.depth_map_to_depth_prob(d)
        _depth_map = self.depth_prob_to_depth_map(depth_prob)

        # debug visualization
        import matplotlib.pyplot as plt
        for i in range(d.shape[0]):
            plt.figure(f'depth_{i}')
            plt.imshow(d[i, 0].detach().cpu().numpy(), cmap='jet')
  
        x = x.view(B * N, C, fH, fW)

        # d = self.dtransform(d)
        # x = torch.cat([depth_prob, x], dim=1)
        # depth = x[:, :self.D].softmax(dim=1)
        x = self.depthnet1(x)
        depth = depth_prob
        x = depth.unsqueeze(1) * x.unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW)
        x = x.permute(0, 1, 3, 4, 5, 2)
        return x
    
    def forward(
        self,
        img,
        points,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3].clone()
        post_rots = img_aug_matrix[..., :3, :3].clone()
        post_trans = img_aug_matrix[..., :3, 3].clone()
        camera2lidar_rots = camera2lidar[..., :3, :3].clone()
        camera2lidar_trans = camera2lidar[..., :3, 3].clone()

        batch_size = len(points)
        depth = torch.zeros(batch_size, img.shape[1], 1,
                            *self.feature_size).to(points[0].device)

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]
            cur_lidar2camera = lidar2camera[b]
            cur_cam2img = cam_intrinsic[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0))

            # debug visualization
            # points = cur_coords.transpose(1, 0).cpu().numpy()
            # draw_scenes_v2(points=points)

            # lidar2image
            downsample_ratio = 8
            cur_cam2img[:2, :2] /= downsample_ratio
            cur_lidar2image = torch.matmul(cur_cam2img, cur_lidar2camera)

            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            # cur_img_aug_matrix[:, :3, :3] /= downsample_ratio
            # cur_img_aug_matrix[:, :3, 3] /= downsample_ratio
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += (cur_img_aug_matrix[:, :3, 3]/downsample_ratio).reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = ((cur_coords[..., 0] < self.feature_size[0])
                      & (cur_coords[..., 0] >= 0)
                      & (cur_coords[..., 1] < self.feature_size[1])
                      & (cur_coords[..., 1] >= 0))
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth = depth.to(masked_dist.dtype)
                depth[b, c, 0, masked_coords[:, 0],
                      masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        # debug
        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x
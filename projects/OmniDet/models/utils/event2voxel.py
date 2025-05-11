from typing import Any, List, Dict
import numpy as np
import torch
import torch.nn.functional as F

from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape


@TRANSFORMS.register_module()
class Event2Voxel(BaseTransform):
    def __init__(self,
                 pixel_size: List[float],
                 time_window: int,
                 resolution: List[int],
                 max_timestamp: int,
                 num_points: int,
                 num_features: int,
                 max_voxels=20000):
        print(max_timestamp/ time_window)

        self.voxel_size = [*pixel_size, time_window]
        self.point_cloud_range = [0, 0, 0, *resolution, max_timestamp]
        self.num_features = num_features
        self.voxelizer = VoxelizationByGridShape(point_cloud_range=self.point_cloud_range,
                                                 max_num_points=num_points,
                                                 voxel_size=self.voxel_size,
                                                 max_voxels=max_voxels)

    def transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        points = data['points']
        res_voxels, res_coors, res_num_points = self.voxelizer(points)  
        return res_voxels, res_coors, res_num_points


if __name__ == '__main__':
    event2voxel = Event2Voxel(
        pixel_size=[2, 2], 
        time_window=1_000_000, # 0.1ms
        resolution=[1280, 720], 
        max_timestamp=100_000_000, # 10ms  # carla use nanoseconds as timestamp in dvsevents
        num_points=1, 
        num_features=4,
        max_voxels=1_000_000)  

    data_path = './data/CarlaCollection/train-Town01_Opt-ClearNoon-2024_09_23_11_44_45/ego0/dvs_camera_front/00026981_xytp.npz'
    with np.load(data_path) as data:
        loaded_events = data["dvs_events"]

    x = torch.from_numpy(loaded_events['x'].copy().astype(np.float32))
    y = torch.from_numpy(loaded_events['y'].copy().astype(np.float32))
    t = torch.from_numpy(loaded_events['t'].copy().astype(np.float32)) 
    pol = torch.from_numpy(loaded_events['pol'].copy().astype(np.float32))  
    
    t = t - t[0]
    t_mask = t < 100_000_000
    events = torch.stack([x, y, t, pol], dim=1)
    events = events[t_mask]
    data = {'points': events}

    pts_voxels, pts_coords, pts_num_points = [], [], []
    res_voxels, res_coors, res_num_points = event2voxel.transform(data)
    res_coors = F.pad(res_coors, (1, 0), mode='constant', value=0)
    
    from mmcv.ops import SparseConvTensor
    from mmdet3d.registry import MODELS
    
    pts_voxels.append(res_voxels)
    pts_coords.append(res_coors)
    pts_num_points.append(res_num_points)
    pts_voxels = torch.cat(pts_voxels, dim=0)
    pts_coords = torch.cat(pts_coords, dim=0)
    pts_num_points = torch.cat(pts_num_points, dim=0)

    print(pts_voxels.shape)
    np.save('spatial_features.npy', pts_voxels.cpu().numpy())

    voxel_encoder=dict(type='HardSimpleVFE')
    voxel_encoder = MODELS.build(voxel_encoder)
    voxel_features = voxel_encoder(pts_voxels, pts_num_points, pts_coords)

    input_sp_tensor = SparseConvTensor(voxel_features, pts_coords.int(), (101, 360, 640), 1)
    spatial_features = input_sp_tensor.dense()
    print(spatial_features.shape)
    np.save('spatial_features.npy', spatial_features.cpu().numpy())


    pts_path = './data/CarlaCollection/train-Town01_Opt-ClearNoon-2024_09_23_11_44_45/ego0/lidar/00026981.bin'
    pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)
    pts = torch.from_numpy(pts)

    pts_voxelizer = VoxelizationByGridShape(point_cloud_range=[-50, -50, -5, 50, 50, 3],
                                            max_num_points=1,
                                            voxel_size=[0.1, 0.1, 0.1],
                                            max_voxels=1_000_000)
    pts_voxels, pts_coords, pts_num_points = [], [], []
    res_voxels, res_coors, res_num_points = pts_voxelizer(pts)
    res_coors = F.pad(res_coors, (1, 0), mode='constant', value=0)
    pts_voxels.append(res_voxels)
    pts_coords.append(res_coors)
    pts_num_points.append(res_num_points)
    pts_voxels = torch.cat(pts_voxels, dim=0)
    pts_coords = torch.cat(pts_coords, dim=0)
    pts_num_points = torch.cat(pts_num_points, dim=0)

    #voxel encoder
    voxel_encoder=dict(type='HardSimpleVFE')
    voxel_encoder = MODELS.build(voxel_encoder)
    voxel_features = voxel_encoder(pts_voxels, pts_num_points, pts_coords)


    input_sp_tensor = SparseConvTensor(voxel_features, pts_coords.int(), (81, 1000, 1000), 1)
    spatial_features = input_sp_tensor.dense()
    print(spatial_features.shape)



    # from mmdet3d.registry import MODELS
    # voxel_encoder=dict(type='HardSimpleVFE')
    # middle_encoder=dict(
    #     type='SparseEncoder',
    #     in_channels=4,
    #     sparse_shape=[41, 1600, 1408],
    #     order=('conv', 'norm', 'act'))
    # voxel_encoder = MODELS.build(voxel_encoder)
    # middle_encoder = MODELS.build(middle_encoder)
    
    print('########')
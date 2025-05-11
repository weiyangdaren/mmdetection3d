from typing import Optional, Union, List, Tuple
import cv2
from matplotlib.artist import get
import numpy as np

from mmcv.transforms.base import BaseTransform
from mmdet3d.registry import TRANSFORMS
from ocamcamera import OcamCamera


@TRANSFORMS.register_module()
class MultiViewFisheyePerspectiveProjection(BaseTransform):
    """Multi-view fisheye perspective projection.

    Args:
        image_size (Tuple[int, int]): The size of the output image.
            Default: (400, 400).
        perspective_fov (float): The field of view of the perspective camera.
            Default: 110.0.
        num_views (int): The number of views. Default: 2.
        camera_orientation (List[float]): The orientation of the camera.
            Default: [-55, 55].
        ocam_path (str): The path to the ocam camera calibration file.
            Default: 'data/CarlaCollection/calib_results.txt'.
        ocam_fov (float): The field of view of the ocam camera.
            Default: 220.0.
    """

    def __init__(self,
                 image_size: Tuple[int, int] = (400, 400),
                 perspective_fov: float = 110.0,
                 num_imgs_of_per_view: int = 2,
                 camera_orientation: List[float] = [-55, 55],
                 ocam_path: str = 'data/CarlaCollection/calib_results.txt',
                 ocam_fov: float = 220.0) -> None:
        self.image_size = image_size
        self.perspective_fov = perspective_fov
        self.num_imgs = num_imgs_of_per_view
        self.camera_orientation = camera_orientation
        assert len(camera_orientation) == self.num_imgs, \
            f'camera_orientation should have {self.num_imgs} elements, ' \
            f'but got {len(camera_orientation)}.'
        
        self.omni_ocam = OcamCamera(filename=ocam_path, fov=ocam_fov)

    @staticmethod
    def generate_perspective_map(ocam, img, W=500, H=500, fov_deg=110, yaw_deg=0):
        fov_rad = np.radians(fov_deg)
        yaw_rad = np.radians(yaw_deg)

        # Compute focal length from FoV and image width
        f = (W / 2) / np.tan(fov_rad / 2)

        # Create image plane in camera frame (centered at 0,0,z)
        x = np.linspace(-W / 2, W / 2 - 1, W)
        y = np.linspace(-H / 2, H / 2 - 1, H)
        x_grid, y_grid = np.meshgrid(x, y)

        # These are 3D points on the image plane at z = f
        rays = np.stack([x_grid, y_grid, np.full_like(x_grid, f)], axis=-1)  # shape: (H, W, 3)

        # Rotate camera by yaw angle around y-axis
        R_yaw = np.array([
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
        ])
        rays_rotated = rays @ R_yaw.T  # shape: (H, W, 3)

        # Reshape to (3, N) for ocam.world2cam
        point3D = rays_rotated.reshape(-1, 3).T
        mapx, mapy = ocam.world2cam(point3D)
        mapx = mapx.reshape(H, W).astype(np.float32)
        mapy = mapy.reshape(H, W).astype(np.float32)

        # Sample fisheye image
        out = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return out, R_yaw

    @staticmethod
    def get_camera_intrinsics(img_height, img_width, view_fov):
        calibration = np.eye(4, dtype=np.float32)
        calibration[0, 2] = img_width / 2.0
        calibration[1, 2] = img_height / 2.0
        calibration[0, 0] = img_width / (2.0 * np.tan(view_fov * np.pi / 360.0))
        calibration[1, 1] = calibration[0, 0]
        return calibration
        
    
    def transform(self, results):
        ori_imgs = results['cam_fisheye']['img']
        new_imgs = list()
        lidar2cam, cam2lidar, lidar2img, cam2img = list(), list(), list(), list()
        # import matplotlib.pyplot as plt
        for i, img in enumerate(ori_imgs):
            # plt.figure('ori_img_{}'.format(i))
            # plt.imshow(img.astype(np.uint8))
            for j in range(self.num_imgs):
                w, h = self.image_size
                new_img, _R_yaw = self.generate_perspective_map(
                    self.omni_ocam, img, w, h, self.perspective_fov,  self.camera_orientation[j])
                new_imgs.append(new_img)
                R_yaw = np.eye(4)
                R_yaw[:3, :3] = _R_yaw
                _lidar2cam = np.dot(R_yaw, results['cam_fisheye']['lidar2cam'][i])
                lidar2cam.append(_lidar2cam)
                _cam2lidar = np.linalg.inv(_lidar2cam)
                cam2lidar.append(_cam2lidar)

                _cam2img = self.get_camera_intrinsics(h, w, self.perspective_fov)
                cam2img.append(_cam2img)
                _lidar2img = np.dot(_cam2img, _lidar2cam)
                lidar2img.append(_lidar2img)

        #         plt.figure('new_img_{}_{}'.format(i, j))
        #         plt.imshow(new_img.astype(np.uint8))
        # plt.show()

        lidar2cam = np.stack(lidar2cam, axis=0)
        cam2lidar = np.stack(cam2lidar, axis=0)
        lidar2img = np.stack(lidar2img, axis=0)
        cam2img = np.stack(cam2img, axis=0)
        results['cam_fisheye']['img'] = new_imgs
        results['cam_fisheye']['lidar2cam'] = lidar2cam
        results['cam_fisheye']['cam2lidar'] = cam2lidar
        results['cam_fisheye']['lidar2img'] = lidar2img
        results['cam_fisheye']['cam2img'] = cam2img
        results['cam_fisheye']['img_shape'] = self.image_size
        results['cam_fisheye']['ori_shape'] = self.image_size
        results['cam_fisheye']['pad_shape'] = self.image_size

        # print('new_imgs', len(new_imgs))

        return results
                


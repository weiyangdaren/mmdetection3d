import mmengine
import numpy as np
import torch
import re
import os
from tqdm import tqdm
from typing import List, Dict, Callable
from pathlib import Path
from functools import partial
from collections import defaultdict, OrderedDict

import datetime
import open3d as o3d
from copy import deepcopy
from mmcv.ops import points_in_boxes_cpu
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import ext.chamfer.chamfer_ext as chamfer_ext
from evendar.dataset.carla_calib import get_matrix, parse_sensor_transform


def get_label_remap():
    _origin_labels = {
        0: 'Terrain',
        25: 'Ground',
        26: 'Ground',
        27: 'Ground',
        1: 'Roads',
        2: 'Sidewalks',
        3: 'Buildings',
        4: 'Buildings',
        5: 'Guardrail',
        6: 'Poles',
        7: 'Poles',
        8: 'Poles',
        9: 'Vegetation',
        10: 'Terrain',
        12: 'Pedestrian',
        13: 'Cyclist',
        14: 'Car',
        15: 'Truck',
        16: 'Bus',
        17: 'Misc',
        18: 'Cyclist',
        19: 'Cyclist',
        20: 'Misc',
        21: 'Misc',
        22: 'Misc',
        11: 'Background',
        23: 'Background',
        24: 'Roads',
        28: 'Guardrail',
        29: 'Misc',
    }

    _unique_labels = []
    for k, v in _origin_labels.items():
        if v not in _unique_labels:
            _unique_labels.append(v)

    semantics_label_remap = {
        origin_label: new_label
        for origin_label, category in _origin_labels.items()
        for new_label, unique_cate in enumerate(_unique_labels)
        if category == unique_cate
    }

    boxes_label_remap = {
        'Car': 10,
        'Van': 11,
        'Truck': 11,
        'Bus': 12,
        'Pedestrian': 8,
        'Cyclist': 9
    }

    new_semantic_labels = {v: k for k, v in enumerate(_unique_labels)}
    # return label_remap, new_semantic_labels
    return semantics_label_remap, boxes_label_remap, new_semantic_labels


class BoundedQueue:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = OrderedDict()

    def update(self, key, value):
        if key in self.data:
            self.data.move_to_end(key)
        self.data[key] = value
        if len(self.data) > self.capacity:
            self.data.popitem(last=False)

    def get(self, key):
        return self.data.get(key, None)


class FastPointSampler:
    def __init__(self,
                 target_num: int,
                 distance_range: List[float],
                 distance_level: int = 100,
                 decay_func: str = "linear"):
        self.target_num = target_num
        self.d_min, self.d_max = distance_range
        self.distance_level = distance_level
        self.decay_func = self.get_decay_func(decay_func)

    def get_decay_func(self, decay_func: str) -> Callable[[np.ndarray], np.ndarray]:
        """根据给定的类型返回对应的衰减函数"""
        d_min, d_max = self.d_min, self.d_max
        if decay_func == "none":
            return None
        if decay_func == "linear":
            k = 1.0 / (d_max - d_min)
            return lambda d: 1 - k * (d - d_min)
        elif decay_func == "quad":
            k = 1.0 / (d_max - d_min) ** 2
            return lambda d: 1 - k * (d - d_min) ** 2
        elif decay_func == "log":
            alpha_log = 1.0
            return lambda d: np.log(alpha_log * (d_max - d) + 1) / np.log(alpha_log * (d_max - d_min) + 1)
        elif decay_func == "exp":
            beta_exp = 0.11
            return lambda d: np.exp(-beta_exp * (d - d_min))
        elif decay_func == "gaussian":
            sigma_gaussian = (d_max - d_min) / 4
            return lambda d: np.exp(-((d - d_min) ** 2) / (2 * sigma_gaussian ** 2))
        else:
            raise ValueError(f"Unknown decay function: {decay_func}")

    def get_drop_probabilities(self, points: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(points[:, :3], axis=1)
        bins = np.linspace(self.d_min, self.d_max, self.distance_level + 1)
        distribution, _ = np.histogram(distances, bins=bins)
        expected_distribution = self.target_num / self.distance_level
        probabilities = 1 - \
            np.clip(expected_distribution / (distribution + 1e-6), 0, 1)

        # Fit probabilities to a curve
        bin_centers = (bins[:-1] + bins[1:]) / 2
        curve_fit = np.interp(distances, bin_centers, probabilities)
        probabilities = np.clip(curve_fit, 0, 1)
        return probabilities

    def sample(self, points: np.ndarray) -> np.ndarray:
        if self.decay_func is None:
            return points

        distances = np.linalg.norm(points[:, :3], axis=1)
        # probs = self.decay_func(distances)
        probs = self.get_drop_probabilities(points)
        rand_vals = np.random.uniform(0, 1, size=len(points))
        mask = probs - rand_vals < 0
        sampled_points = points[mask]

        # 调整点数接近 target_num
        # while len(sampled_points) > self.target_num:
            # sampled_points = sampled_points[np.random.choice(len(sampled_points), self.target_num, replace=False)]

        return sampled_points


def parser_file_path(root_dir, max_neighor=100, neighbor_range=None, distance_threshold=None, single_scene_name=None):
    """
    parser the file path of the semantic lidar

    Args:
        root_dir (str): the root directory of the dataset
        max_neighor (int): the maximum number of the neighbor frames
        neighbor_range (List): a list contains the neighbor range
        distance_threshold (List): a list contains the distance threshold
        single_scene_name (str): single scene name for debug
    Returns:
        dict: a dictionary contains the file path of the semantic lidar
    """
    if neighbor_range is None:
        neighbor_range = [-100, 100]

    if distance_threshold is None:
        distance_threshold = [0.3, 5.0]  # min, max

    def add_neighbor_frame(frame_idx, frame_idx_with_loc, frame_idx_list, neighbor_frame_paths, neighbor_count):
        pass

    def parser_scene_file_path(root_dir, scene_name):
        print(f"Prepare {scene_name}...")

        semantic_transform_file = root_dir / scene_name / \
            'ego0' / 'semantic_lidar' / 'sensor_metadata.txt'

        frame_idx_with_loc = parse_sensor_transform(semantic_transform_file)
        frame_idx_list = list(frame_idx_with_loc.keys())
        scene_file_paths = defaultdict(dict)
        for frame_idx in frame_idx_list:
            label_path = root_dir / scene_name / \
                'ego0' / 'label_3d' / f'{frame_idx}.json'
            lidar_path = root_dir / scene_name / \
                'ego0' / 'semantic_lidar' / f'{frame_idx}.bin'

            current_location = deepcopy(
                frame_idx_with_loc[frame_idx]['location'])

            # get the neighbor frames via distance
            neighbor_frame_paths = []
            neighbor_count = 0

            # current to backward
            for i in range(-1, neighbor_range[0], -1):  # from zero to the max
                neighbor_frame_idx = f"{(int(frame_idx) + i):08d}"
                if neighbor_frame_idx not in frame_idx_list:
                    continue

                # Calculate the distance between the current frame and the neighbor frame
                neighbor_location = frame_idx_with_loc[neighbor_frame_idx]['location']
                distance = np.linalg.norm(
                    np.array(current_location) - np.array(neighbor_location))

                # Add the neighbor frame only if the distance is within the threshold
                if distance < distance_threshold[0] or distance > distance_threshold[1]:
                    continue

                neighbor_label_path = root_dir / scene_name / \
                    'ego0' / 'label_3d' / f'{neighbor_frame_idx}.json'
                neighbor_lidar_path = root_dir / scene_name / \
                    'ego0' / 'semantic_lidar' / f'{neighbor_frame_idx}.bin'
                neighbor_frame_paths.append({
                    'frame_idx': neighbor_frame_idx,
                    'label_path': neighbor_label_path,
                    'lidar_path': neighbor_lidar_path
                })

                current_location = neighbor_location
                neighbor_count += 1
                if neighbor_count >= max_neighor:
                    break

            # current to forward
            neighbor_count = 0
            current_location = deepcopy(
                frame_idx_with_loc[frame_idx]['location'])
            for i in range(1, neighbor_range[1], 1):  # from zero to the max
                neighbor_frame_idx = f"{(int(frame_idx) + i):08d}"
                if neighbor_frame_idx not in frame_idx_list:
                    continue

                # Calculate the distance between the current frame and the neighbor frame
                neighbor_location = frame_idx_with_loc[neighbor_frame_idx]['location']
                distance = np.linalg.norm(
                    np.array(current_location) - np.array(neighbor_location))

                # Add the neighbor frame only if the distance is within the threshold
                if distance < distance_threshold[0] or distance > distance_threshold[1]:
                    continue

                neighbor_label_path = root_dir / scene_name / \
                    'ego0' / 'label_3d' / f'{neighbor_frame_idx}.json'
                neighbor_lidar_path = root_dir / scene_name / \
                    'ego0' / 'semantic_lidar' / f'{neighbor_frame_idx}.bin'
                neighbor_frame_paths.append({
                    'frame_idx': neighbor_frame_idx,
                    'label_path': neighbor_label_path,
                    'lidar_path': neighbor_lidar_path
                })

                current_location = neighbor_location
                neighbor_count += 1
                if neighbor_count >= max_neighor:
                    break

            scene_file_paths[frame_idx] = {
                'label_path': label_path,
                'lidar_path': lidar_path,
                'neighbor_frames': neighbor_frame_paths
            }
        return scene_name, scene_file_paths

    total_file_paths = defaultdict(dict)
    scene_names = [x.stem for x in root_dir.iterdir()
                   if x.suffix == '.json']
    # mauanlly set the scene names for multi-processing, fuck multiprocess package
    scene_names = scene_names[36:72]
    if single_scene_name:
        scene_names = [single_scene_name]

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = executor.map(lambda scene: parser_scene_file_path(
            root_dir, scene), scene_names)
        # futures = {executor.submit(parser_scene_file_path, root_dir, scene_name): scene_name for scene_name in scene_names}
    for scene_name, scene_file_paths in results:
        total_file_paths[scene_name] = scene_file_paths

    print("Finish parsing all file paths...")
    return total_file_paths


class OccGenerator:
    def __init__(self,
                 root_dir: str,
                 scene_name: str,
                 file_paths: Dict[str, Dict],
                 voxel_size: List[float] = [0.4, 0.4, 0.4],
                 pc_range: List[float] = [-40, -40, -2.4,
                                          40, 40, 4.0],  # 0 is the current frame
                 whole_scene_to_mesh: bool = False,
                 with_semantics: bool = False,
                 save_as_grid: bool = False,
                 max_points: int = 500000,
                 sample_points_func: str = 'none',
                 ):
        self.root_dir = Path(root_dir)
        self.scene_name = scene_name
        self.file_paths = file_paths

        self.voxel_size = np.array(voxel_size)
        self.pc_range = np.array(pc_range)
        self.whole_scene_to_mesh = whole_scene_to_mesh
        self.with_semantics = with_semantics
        self.save_as_grid = save_as_grid
        grid_size = (self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size
        self.grid_size = np.round(grid_size).astype(np.int32)

        self.lidar_cache = BoundedQueue(200)
        self.label_cache = BoundedQueue(200)
        self.point_sampler = FastPointSampler(
            target_num=max_points,
            distance_range=[0, 40],
            decay_func=sample_points_func)

        self.sem_label_remap, self.box_label_remap, self.semantics_label = get_label_remap()
        self.transform = self.load_transform()

    def load_transform(self):
        """
        Load the transformation including translation and rotation of the semantic lidar

        Returns:
            dict: a dictionary contains the transformation of the semantic lidar

        """

        if (self.root_dir / 'semantic_lidar_transform.pkl').exists():
            return mmengine.load(self.root_dir / 'semantic_lidar_transform.pkl')[self.scene_name]

        semantic_lidar_transform = defaultdict(dict)
        scene_names = [x.stem for x in self.root_dir.iterdir()
                       if x.suffix == '.json']
        for scene_name in scene_names:
            semantic_transform_file = self.root_dir / scene_name / \
                'ego0' / 'semantic_lidar' / 'sensor_metadata.txt'
            sensor_transforms = parse_sensor_transform(semantic_transform_file)
            semantic_lidar_transform[scene_name] = sensor_transforms
        mmengine.dump(semantic_lidar_transform, self.root_dir /
                      'semantic_lidar_transform.pkl')
        return semantic_lidar_transform[self.scene_name]

    def load_pc_from_file(self, file_path):
        """
        Load a point cloud from a file.

        Args:
            file_path (str): The path to the file containing the point cloud.

        Returns:
            open3d.geometry.PointCloud: The loaded point cloud.
        """
        if self.lidar_cache.get(file_path) is not None:
            return deepcopy(self.lidar_cache.get(file_path))

        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        points[:, -1] = np.vectorize(self.sem_label_remap.get)(points[:, -1])
        self.lidar_cache.update(file_path, points)
        return points

    def load_label(self, file_path):
        """
        Load the labels from a file.

        Args:
            file_path (str): The path to the file containing the labels.

        Returns:
            numpy.ndarray: The loaded labels.
        """
        if self.label_cache.get(file_path) is not None:
            return deepcopy(self.label_cache.get(file_path))

        labels = mmengine.load(file_path)
        # with open(file_path, 'r') as f:
        #     labels = json.load(f)

        gt_boxes = []
        gt_names = []
        gt_ids = []
        for label in labels:
            gt_boxes.append(label['bounding_box'])
            gt_names.append(self.box_label_remap[label['class']])
            gt_ids.append(label['id'])

        gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1, 7)
        gt_names = np.array(gt_names)
        gt_ids = np.array(gt_ids)
        gt_boxes = self.enlarge_box3d(gt_boxes, gt_names)
        gt_boxes = self.boxes_to_bottom_center(gt_boxes)
        self.label_cache.update(file_path, (gt_boxes, gt_names, gt_ids))
        return gt_boxes, gt_names, gt_ids

    @staticmethod
    def enlarge_box3d(boxes3d, boxes_names, extra_width=(0.4, 0.4, 0.4)):
        """
        Args:
            boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            extra_width: [extra_x, extra_y, extra_z]

        Returns:

        """
        large_boxes3d = deepcopy(boxes3d)
        large_boxes3d[:, 3:6] += np.array(extra_width)[None, :]

        # enlarge the pedestrian box
        ped_mask = boxes_names == 8
        large_boxes3d[ped_mask, 3:6] += 0.4

        # cyc_mask = boxes_names == 9
        # large_boxes3d[cyc_mask, 3:6] += 0.2

        return large_boxes3d

    @staticmethod
    def boxes_to_bottom_center(boxes3d):
        """
        Args:
            boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        Returns:

        """
        bottom_center_boxes3d = deepcopy(boxes3d)
        bottom_center_boxes3d[:, 2] -= boxes3d[:, 5] / 2
        return bottom_center_boxes3d

    @staticmethod
    def separate_object_and_background(points, boxes, boxes_name, box_label_remap=None, ego_range=None, use_kdtree_optimization=False):
        """
        Enhanced point cloud segmentation directly using numerical semantic labels for object optimization

        Args:
            points (np.ndarray): [N,4] Point cloud data (x, y, z, semantic_label)
            boxes (np.ndarray): [M,7] 3D bounding boxes in format (x, y, z, dx, dy, dz, yaw)
            boxes_name (np.ndarray/list): [M] Numerical semantic labels for each bounding box
            ego_range (list): Ego vehicle range filter [x_range, y_range, z_range]
            use_kdtree_optimization (bool): Use KDTree optimization for point cloud segmentation

        Returns:
            object_points_list (list): [M] List of optimized point clouds for each object (elements shape: [K,4])
            static_points (np.ndarray): Static background point cloud
        """

        points_tensor = torch.from_numpy(
            points[:, :3]).float().unsqueeze(0)  # (1, N, 3)
        boxes_tensor = torch.from_numpy(
            boxes).float().unsqueeze(0)           # (1, M, 7)
        # geom mask [1, N, M]
        # print(points_tensor.shape, boxes_tensor.shape)
        points_tensor = deepcopy(points_tensor)
        boxes_tensor = deepcopy(boxes_tensor)
        points_in_boxes = points_in_boxes_cpu(points_tensor, boxes_tensor)

        sem_labels = points[:, 3].astype(int)
        boxes_sem = np.asarray(boxes_name).astype(int)

        object_points_list = []  # List of object point clouds
        if box_label_remap is None:
            # Dynamic points mask
            dynamic_mask = np.zeros(len(points), dtype=bool)
        # KDTree for fast search
        if use_kdtree_optimization:
            kdtree = cKDTree(points[:, :3])
        for obj_idx in range(points_in_boxes.shape[-1]):
            # mask of points in the current box
            geom_mask = points_in_boxes[0, :, obj_idx].bool().numpy()
            # semantic label of the current box
            expected_sem = boxes_sem[obj_idx]

            # Filter points by geometry and semantics
            sem_mask = (sem_labels == expected_sem)
            valid_mask = geom_mask & sem_mask

            if use_kdtree_optimization:
                # Search radius for KDTree query
                # use maximum box dimension for search radius
                search_radius = boxes[obj_idx, 3:].max()
                box_center = boxes[obj_idx, :3]

                # Query points in the search radius
                neighbor_indices = kdtree.query_ball_point(
                    box_center, r=search_radius)
                neighbor_indices = np.array(neighbor_indices).flatten()

                # Filter points by semantics
                neighbor_sem_mask = (
                    sem_labels[neighbor_indices] == expected_sem)
                valid_neighbors = neighbor_indices[neighbor_sem_mask]
                valid_mask[valid_neighbors] = True

            object_points = points[valid_mask]
            object_points_list.append(object_points)
            if box_label_remap is None:
                dynamic_mask |= valid_mask

        if box_label_remap is None:
            static_mask = ~dynamic_mask
            static_points = points[static_mask]
        else:
            # to ensure the static points are not contains the object points that are not in the boxes
            box_labels = set([x for x in box_label_remap.values()])
            static_mask = np.ones(len(points), dtype=bool)
            for i, label in enumerate(box_labels):
                static_mask[points[:, 3] == label] = False
            static_points = points[static_mask]

        if ego_range is not None:
            ego_mask = (
                (np.abs(points[:, 0]) > ego_range[0]) |
                (np.abs(points[:, 1]) > ego_range[1]) |
                (np.abs(points[:, 2]) > ego_range[2])
            )
            static_mask &= ego_mask

        return object_points_list, static_points

    @staticmethod
    def lidar_to_other_lidar(pcd, source_transform, target_transform):
        """
        Transform a point cloud from one LiDAR sensor to another.

        Args:
            pcd (numpy.ndarray): The input point cloud to transform.
            source_transform (dict): The transformation matrix of the source LiDAR sensor.
            target_transform (dict): The transformation matrix of the target LiDAR sensor.

        Returns:
            numpy.ndarray: The transformed point cloud.
        """

        if pcd.shape[1] == 4:
            pcd_xyz = pcd[:, :3]
            pcd_sem = pcd[:, 3]
        else:
            pcd_xyz = pcd

        source2world = get_matrix(**source_transform)
        target2world = get_matrix(**target_transform)

        world2target = np.linalg.inv(target2world)
        source2target = np.dot(world2target, source2world)
        pcd_xyz = np.dot(source2target[:3, :3],
                         pcd_xyz.T).T + source2target[:3, 3]

        if pcd.shape[1] == 4:
            return np.concatenate([pcd_xyz, pcd_sem[:, None]], axis=1)
        else:
            return pcd_xyz

    @staticmethod
    def mask_points_out_of_range(points, pc_range):
        """
        Masks points that are outside the specified range.

        Args:
            points (numpy.ndarray): The input points to mask.
            pc_range (list): The range to keep points within.

        Returns:
            numpy.ndarray: The masked points.
        """
        mask = (points[:, 0] > pc_range[0]) & (points[:, 0] < pc_range[3]) \
            & (points[:, 1] > pc_range[1]) & (points[:, 1] < pc_range[4]) \
            & (points[:, 2] > pc_range[2]) & (points[:, 2] < pc_range[5])
        return points[mask]

    @staticmethod
    def points_to_voxel(points, voxel_size, pc_range, grid_size):
        points[:, :3] = (points[:, :3] - pc_range[:3]) / voxel_size
        points[:, :3] = np.floor(points[:, :3])
        points = points.astype(np.int32)

        voxel = np.zeros(grid_size, dtype=np.float32)
        voxel[points[:, 0], points[:, 1], points[:, 2]] = 1

        _x = np.linspace(0, grid_size[0] - 1, grid_size[0])
        _y = np.linspace(0, grid_size[1] - 1, grid_size[1])
        _z = np.linspace(0, grid_size[2] - 1, grid_size[2])
        xx, yy, zz = np.meshgrid(_x, _y, _z, indexing='ij')
        meshgrid = np.stack([xx, yy, zz], axis=-1)
        voxel_fg = meshgrid[voxel == 1]
        voxel_fg[:, :3] = (voxel_fg[:, :3]+0.5) * \
            voxel_size + pc_range[:3]
        voxel_fg = voxel_fg.astype(np.float32)
        return voxel_fg

    @staticmethod
    def assign_semantics(voxel, pts_with_sem, voxel_size, pc_range):
        x = torch.from_numpy(voxel).cuda().unsqueeze(0).float()
        y = torch.from_numpy(pts_with_sem[:, :3]).cuda().unsqueeze(0).float()
        d1, d2, idx1, idx2 = chamfer_ext.forward(x, y)
        indices = idx1[0].cpu().numpy()

        voxel_sem = pts_with_sem[:, 3][np.array(indices)]
        voxel_with_sem = np.concatenate([voxel, voxel_sem[:, None]], axis=1)

        # to voxel coordinates
        voxel_with_sem[:, :3] = (
            voxel_with_sem[:, :3] - pc_range[:3]) / voxel_size
        voxel_with_sem[:, :3] = np.floor(voxel_with_sem[:, :3])
        voxel_with_sem = voxel_with_sem.astype(np.int32)
        return voxel_with_sem

    @staticmethod
    def convert_voxel_to_grid(voxel, grid_size, semantics_label=None):
        """
        Convert voxel data to a grid representation.

        Args:
            voxel (numpy.ndarray): The voxel data (N*3 or N*4, with optional semantic labels).
            grid_size (tuple): The size of the grid (H, W, D).
            semantics_label (dict, optional): Mapping of semantic labels.

        Returns:
            dict: A dictionary containing the 'semantics' grid and 'mask_lidar' grid.
        """
        free_label = len(semantics_label) if semantics_label is not None else 0
        grid = np.ones(grid_size, dtype=np.float32) * free_label
        mask_lidar = np.zeros(grid_size, dtype=np.uint8)

        # Extract voxel coordinates
        coords = voxel[:, :3].astype(np.int32)

        # Mark occupied cells in the mask. May be we dont need this
        # mask_lidar[coords[:, 0], coords[:, 1], coords[:, 2]] = 1

        # Assign semantic labels if applicable
        semantics = voxel[:, 3].astype(np.int32)
        grid[coords[:, 0], coords[:, 1], coords[:, 2]] = semantics

        return {'voxel': voxel, 'semantics': grid}

    @staticmethod
    def place_obj_points(seq_infos, obj_ids_zoo, obj_names_zoo, obj_points_zoo):
        gt_boxes = seq_infos[0]['gt_boxes']  # the boxes in the current frame
        locs = gt_boxes[:, :3]
        dims = gt_boxes[:, 3:6]
        rots = gt_boxes[:, 6]

        obj_xyz = []
        obj_sem = []

        for i, obj_id in enumerate(seq_infos[0]['obj_id']):
            for j, obj_query in enumerate(obj_ids_zoo):
                if obj_id == obj_query:
                    pts_xyz = obj_points_zoo[obj_query][:, :3]
                    Rot = Rotation.from_euler('z', rots[i], degrees=False)
                    rotated_pts_xyz = Rot.apply(pts_xyz)
                    pts_xyz = rotated_pts_xyz + locs[i]

                    if pts_xyz.shape[0] > 8:
                        pts_in_box = points_in_boxes_cpu(torch.from_numpy(pts_xyz[None, :, :]),
                                                         torch.from_numpy(gt_boxes[i][None, None, :]))
                        pts_xyz = pts_xyz[pts_in_box[0, :, 0].bool()]

                    obj_xyz.append(pts_xyz)
                    obj_sem.append(
                        np.ones(pts_xyz.shape[0]) * obj_names_zoo[j])

        if len(obj_xyz) == 0:
            return None

        obj_xyz = np.concatenate(obj_xyz, axis=0)
        obj_sem = np.concatenate(obj_sem, axis=0)
        obj_points = np.concatenate([obj_xyz, obj_sem[:, None]], axis=1)
        return obj_points

    @staticmethod
    def prepare_obj_points(seq_infos):
        """
        Args:
            seq_infos (list): a list contains the information of the frames

        Returns:
            list: a list contains the object ids
            list: a list contains the object names
            dict: a dictionary contains the object points, 
                  each object points is a numpy.ndarray, shape (N, 4)
        """

        obj_ids_zoo = []
        obj_names_zoo = []
        for info in seq_infos:
            for i, obj_id in enumerate(info['obj_id']):
                if obj_id not in obj_ids_zoo and info['obj_points'][i].shape[0] > 0:
                    obj_ids_zoo.append(obj_id)
                    obj_names_zoo.append(info['gt_names'][i])

        # move and rotate the object points in all frames to the each object center,
        # so we can complete each object point cloud
        obj_points_zoo = defaultdict(list)
        for obj_query in obj_ids_zoo:
            for info in seq_infos:
                for i, obj_id in enumerate(info['obj_id']):
                    if obj_query == obj_id:
                        obj_points = info['obj_points'][i]
                        if obj_points.shape[0] > 0:
                            # relative coordinates
                            obj_points = obj_points[:,
                                                    :3] - info['gt_boxes'][i][:3]
                            Rot = Rotation.from_euler(
                                'z', -info['gt_boxes'][i][6], degrees=False)
                            rotated_obj_points = Rot.apply(obj_points)
                            obj_points_zoo[obj_query].append(
                                rotated_obj_points)
            obj_points_zoo[obj_query] = np.concatenate(
                obj_points_zoo[obj_query], axis=0)
        return obj_ids_zoo, obj_names_zoo, obj_points_zoo

    @staticmethod
    def create_mesh_from_map(pcd, depth, n_threads, min_density=None):
        """
        Generates a 3D mesh from a point cloud using the Poisson surface reconstruction method.

        Args:
            pcd (open3d.geometry.PointCloud): The input point cloud to reconstruct the surface from.
            depth (int): The depth of the octree used for surface reconstruction. Higher values result in finer details.
            n_threads (int): The number of threads to use for the computation.
            min_density (float, optional): The minimum density threshold for filtering vertices. Vertices with density 
                below this threshold (quantile) will be removed. If None, no filtering is applied.

        Returns:
            tuple: A tuple containing:
                - mesh (open3d.geometry.TriangleMesh): The reconstructed 3D mesh.
                - densities (numpy.ndarray): The density values of the vertices in the mesh.
        """
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        #     pcd, depth=depth, n_threads=n_threads
        # )

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=10,  
            n_threads=32,
            linear_fit=True
        )

        # Post-process the mesh
        if min_density:
            vertices_to_remove = densities < np.quantile(
                densities, min_density)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        mesh.compute_vertex_normals()

        return mesh, densities

    @staticmethod
    def preprocess_cloud(pcd, max_nn=20, normals=None):
        """
        Preprocess a point cloud by optionally estimating and orienting normals.

        Args:
            pcd (open3d.geometry.PointCloud): The input point cloud to preprocess.
            max_nn (int, optional): Maximum number of nearest neighbors to use for normal estimation. Defaults to 20.
            normals (bool, optional): If True, normals will be estimated and oriented towards the camera location. Defaults to None.

        Returns:
            open3d.geometry.PointCloud: The preprocessed point cloud with optionally estimated and oriented normals.
        """
        cloud = deepcopy(pcd)
        if normals:
            params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
            cloud.estimate_normals(params)
            cloud.orient_normals_towards_camera_location()

        return cloud

    def scene_to_mesh(self, points):
        """
        Args:
            points (numpy.ndarray): the points of the scene, shape (N, 3)
        """
        # create mesh from the whole scene
        pts_ori = o3d.geometry.PointCloud()
        with_norm = o3d.geometry.PointCloud()
        pts_ori.points = o3d.utility.Vector3dVector(points[:, :3])
        # import time
        # time1 = time.time()
        _with_norm = self.preprocess_cloud(
            pts_ori, max_nn=20, normals=True)
        # time2 = time.time()
        # print(f"Preprocess cloud time: {time2 - time1:.2f}s")

        with_norm.points = _with_norm.points
        with_norm.normals = _with_norm.normals
        mesh, _ = self.create_mesh_from_map(
            with_norm, depth=10, n_threads=32, min_density=0.1)
        # time3 = time.time()
        # print(f"Create mesh from map time: {time3 - time2:.2f}s")


        new_points = np.asarray(mesh.vertices, dtype=np.float32)
        return new_points

    def prepare_single_frame(self, frame_idx):
        """
        prepare the single frame

        Args:
            frame_idx (str): the index of the frame

        Returns:
            list: a list contains the information of the frame
            numpy.ndarray: the points of the frame, shape (N, 4)
        """

        seq_infos = []

        label_path = self.file_paths[frame_idx]['label_path']
        lidar_path = self.file_paths[frame_idx]['lidar_path']

        gt_boxes, gt_names, gt_ids = self.load_label(label_path)
        lidar = self.load_pc_from_file(lidar_path)
        target_transform = self.transform[frame_idx]
        # list of object points, background points (numpy ndarray)
        obj_points, bg_points = self.separate_object_and_background(
            lidar, gt_boxes, gt_names, self.box_label_remap)

        seq_infos.append({
            'frame_idx': frame_idx,
            'obj_id': gt_ids,
            'obj_points': obj_points,
            'bg_points': bg_points,
            'transform': target_transform,
            'gt_boxes': gt_boxes,
            'gt_names': gt_names
        })

        neighbor_frames = self.file_paths[frame_idx]['neighbor_frames']
        merge_points = [bg_points]

        def process_neighbor_frame(neighbor_frame):
            neighbor_frame_idx = neighbor_frame['frame_idx']
            neighbor_label_path = neighbor_frame['label_path']
            neighbor_lidar_path = neighbor_frame['lidar_path']

            neighbor_gt_boxes, neighbor_gt_names, neighbor_gt_ids = self.load_label(
            neighbor_label_path)
            neighbor_lidar = self.load_pc_from_file(neighbor_lidar_path)
            neighbor_transform = self.transform[neighbor_frame_idx]

            neighbor_obj_points, neighbor_bg_points = self.separate_object_and_background(
            neighbor_lidar, neighbor_gt_boxes, neighbor_gt_names, self.box_label_remap)

            # Transform the neighbor lidar to the current lidar
            transformed_points = self.lidar_to_other_lidar(
            neighbor_bg_points, neighbor_transform, target_transform)

            return {
            'seq_info': {
                'frame_idx': neighbor_frame_idx,
                'obj_id': neighbor_gt_ids,
                'obj_points': neighbor_obj_points,
                'bg_points': transformed_points,
                'transform': neighbor_transform,
                'gt_boxes': neighbor_gt_boxes,
                'gt_names': neighbor_gt_names
            },
            'transformed_points': transformed_points
            }

        with ThreadPoolExecutor(max_workers=32) as executor:
            results = list(executor.map(process_neighbor_frame, neighbor_frames))

        for result in results:
            merge_points.append(result['transformed_points'])
            seq_infos.append(result['seq_info'])

        # merge the points
        merge_points = np.concatenate(merge_points, axis=0)
        return seq_infos, merge_points

    def process_single_frame(self, frame_idx):
        # import time
        # time1 = time.time()
        seq_infos, ori_points = self.prepare_single_frame(frame_idx)
        # time2 = time.time()
        # print(f"Prepare frame {frame_idx} time: {time2 - time1:.2f}s")

        obj_ids_zoo, obj_names_zoo, obj_points_zoo = self.prepare_obj_points(
            seq_infos)
        # time3 = time.time()
        # print(f"Prepare object points time: {time3 - time2:.2f}s")

        obj_points = self.place_obj_points(
            seq_infos, obj_ids_zoo, obj_names_zoo, obj_points_zoo)
        if obj_points is not None:
            ori_points = np.concatenate([ori_points, obj_points], axis=0)
        # time4 = time.time()
        # print(f"Place object points time: {time4 - time3:.2f}s")

        ori_points = self.point_sampler.sample(ori_points)
        # time5 = time.time()
        # print(f"Sample points time: {time5 - time4:.2f}s")

        pts_xyz = ori_points[:, :3]
        if self.whole_scene_to_mesh:
            pts_xyz = self.scene_to_mesh(pts_xyz)
        # time6 = time.time()
        # print(f"Scene to mesh time: {time6 - time5:.2f}s")

        pts_xyz = self.mask_points_out_of_range(
            pts_xyz, self.pc_range)
        ori_points = self.mask_points_out_of_range(
            ori_points, self.pc_range)

        voxel_fg = self.points_to_voxel(
            pts_xyz,
            voxel_size=self.voxel_size,
            pc_range=self.pc_range,
            grid_size=self.grid_size)  # (N, 3)

        if self.with_semantics:
            voxel_fg = self.assign_semantics(
                voxel_fg,
                pts_with_sem=ori_points,
                voxel_size=self.voxel_size,
                pc_range=self.pc_range,)  # (N, 4)

        if self.with_semantics and self.save_as_grid:
            occ_grid = self.convert_voxel_to_grid(
                voxel_fg,
                grid_size=self.grid_size,
                semantics_label=self.semantics_label)
        else:
            occ_grid = {'voxel': voxel_fg}

        return occ_grid

    def process(self):
        save_dir = self.root_dir / self.scene_name / 'ego0' / 'label_occ'
        save_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(self.file_paths.items(),
                    desc=f"Processing scene {self.scene_name}")
        for frame_idx, _ in pbar:
            pbar.set_description(
                f"Processing frame {frame_idx}")
            occ_grid = self.process_single_frame(frame_idx)
            save_path = save_dir / f'{frame_idx}.npz'
            np.savez_compressed(save_path, **occ_grid)


def process_single_scene(
        root_dir,
        scene_name,
        file_paths,
        voxel_size=[0.4, 0.4, 0.4],
        pc_range=[-40, -40, -2.4, 40, 40, 4.0],
        whole_scene_to_mesh=False,
        with_semantics=False,
        save_as_grid=False,):

    pid = os.getpid()
    print(f"[PID {pid}] Processing scene {scene_name}.")
    generator = OccGenerator(
        root_dir=root_dir,
        scene_name=scene_name,
        file_paths=file_paths,
        voxel_size=voxel_size,
        pc_range=pc_range,
        whole_scene_to_mesh=whole_scene_to_mesh,
        with_semantics=with_semantics,
        save_as_grid=save_as_grid,
        max_points=2000000,
        sample_points_func='gaussian')
    generator.process()

    return scene_name


def check_not_finished(log_file, scene_name_list):
    """
    Check if the processing of all scenes is finished.

    Args:
        log_file (str): The path to the log file.
        scene_name_list (list): A list of scene names.

    Returns:
        scene_name_list (list): A list of scene names that not in log file.
    """
    log_file = Path(log_file)
    if not log_file.exists():
        return scene_name_list

    with open(log_file, 'r') as f:
        lines = f.readlines()

    finished_scenes = set(line.split("Finish processing scene ")[-1].rsplit(
        '.', 1)[0].strip() for line in lines if "Finish processing scene" in line)
    return [scene_name for scene_name in scene_name_list if scene_name not in finished_scenes]


def process(
        root_dir,
        voxel_size=[0.4, 0.4, 0.4],
        pc_range=[-40, -40, -2.4, 40, 40, 4.0],
        whole_scene_to_mesh=False,
        with_semantics=False,
        save_as_grid=False,
        num_workers=32,
        log_file=None):

    file_path_with_scene = parser_file_path(Path(root_dir), max_neighor=100, neighbor_range=[-250, 250],
        distance_threshold=[0.2, 5.0])
    scene_name_list = list(file_path_with_scene.keys())
    new_scene_name_list = check_not_finished(
        log_file, scene_name_list) if log_file else scene_name_list

    print(f"Start processing {len(new_scene_name_list)} scenes...")

    for scene_name in new_scene_name_list:
        process_single_scene(
            root_dir=root_dir,
            scene_name=scene_name,
            file_paths=file_path_with_scene[scene_name],
            voxel_size=voxel_size,
            pc_range=pc_range,
            whole_scene_to_mesh=whole_scene_to_mesh,
            with_semantics=with_semantics,
            save_as_grid=save_as_grid)

        if log_file is not None:
            with open(log_file, 'a') as f:
                f.write(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Finish processing scene {scene_name}.\n")

        print(f"Finish processing scene {scene_name}.")

    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     results = {executor.submit(process_single_scene, root_dir, scene_name,
    #                                file_path_with_scene[scene_name], voxel_size, pc_range,
    #                                whole_scene_to_mesh, with_semantics, save_as_grid): scene_name for scene_name in scene_name_list}

    #     # Trigger execution by iterating over results
    #     for future in as_completed(results):
    #         scene_name = results[future]
    #         try:
    #             future.result()
    #             print(f"Finish processing scene {scene_name}.")
    #         except Exception as e:
    #             print(f"Error processing scene {scene_name}: {e}")

    # print("Finish processing all scenes.")


def process_debug(root_dir):
    # process single scene for debug
    single_scene_name = 'train-Town01_Opt-ClearNoon-2024_09_23_11_44_45'
    neighbor_range = [-250, 250]

    file_paths = parser_file_path(Path(
        root_dir), max_neighor=100, neighbor_range=neighbor_range,
        distance_threshold=[0.2, 5.0], single_scene_name=single_scene_name)[single_scene_name]

    new_file_paths = dict()
    count = 0
    for frame_idx, file_path in file_paths.items():
        new_file_paths[frame_idx] = file_path
    
    frame_idx_with_neighbor = []
    for frame_idx, _ in new_file_paths.items():
        neighbor_frames = [x['frame_idx'] for x in new_file_paths[frame_idx]['neighbor_frames']]
        frame_idx_with_neighbor.append({
            frame_idx: neighbor_frames
        })
    # print(frame_idx_with_neighbor)


    voxel_size = [0.4, 0.4, 0.4]
    pc_range = [-40, -40, -2.4, 40, 40, 4.0]
    whole_scene_to_mesh = True
    with_semantics = True
    save_as_grid = True

    occ_generator = OccGenerator(
        root_dir=root_dir,
        scene_name=single_scene_name,
        file_paths=new_file_paths,
        voxel_size=voxel_size,
        pc_range=pc_range,
        whole_scene_to_mesh=whole_scene_to_mesh,
        with_semantics=with_semantics,
        save_as_grid=save_as_grid,
        max_points=2000000,
        sample_points_func='gaussian')

    occ_generator.process()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    root_dir = './data/CarlaCollection'
    process(
        root_dir=root_dir,
        voxel_size=[0.4, 0.4, 0.4],
        pc_range=[-40, -40, -2.4, 40, 40, 4.0],
        whole_scene_to_mesh=True,
        with_semantics=True,
        save_as_grid=True,
        num_workers=2,
        log_file='./log.log'
    )

    # process_debug(root_dir)

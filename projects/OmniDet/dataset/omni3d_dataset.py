from typing import Callable, List, Union, Optional, Dict
from pathlib import Path

import numpy as np

# from mmengine.dataset import BaseDataset
from mmdet3d.registry import DATASETS
from mmdet3d.datasets import Det3DDataset
from mmdet3d.structures import LiDARInstance3DBoxes


@DATASETS.register_module()
class Omni3DDataset(Det3DDataset):
    METAINFO = {
        'classes':
        ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist'),
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Dict],
                 test_mode: bool = False,
                #  modality: Dict[str, Union[bool, List[str]]] = None,
                #  cam_names: Optional[Dict[str, List[str]]] = None,
                 metainfo: Optional[Dict[str, str]] = None,
                 **kwargs):
        # self.use_lidar = modality.get('use_lidar', False)
        # self.use_sensor = modality.get('use_cam_sensor', [])
        # self.cam_names = cam_names
        # for sensor in self.use_sensor:
        #     assert sensor in self.cam_names.keys(
        #     ), f'{sensor} not in {self.cam_names.keys()}'
        # self.load_eval_anns = kwargs.get('load_eval_anns', False)

        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            metainfo=metainfo,
            pipeline=pipeline,
            test_mode=test_mode,
            **kwargs)

    def parse_anno_info(self, info: dict) -> dict:
        ann_info = dict()
        if len(info['annos']['gt_names']) > 0:
            gt_bboxes_3d = LiDARInstance3DBoxes(info['annos']['gt_boxes'])
            ann_info['gt_bboxes_3d'] = gt_bboxes_3d
            ann_info['gt_labels_3d'] = np.vectorize(
                self.metainfo['categories'].get)(info['annos']['gt_names'])
        else:
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)
        info.pop('annos')
        return ann_info

    def parse_data_info(self, info: dict) -> Union[List[dict], dict]:
        # ret_info = dict()
        # if self.use_lidar:
        #     lidar_points = dict()
        #     lidar_points['num_features'] = info['pc_info']['num_features']
        #     lidar_points['lidar_path'] = Path(
        #         self.data_root) / info['pc_info'].pop('pts_path')
        #     ret_info['lidar_points'] = lidar_points

        # cam_instances = []
        # for sensor in self.use_sensor:
        #     for cam_id in self.cam_names[sensor]:
        #         img_info = info['cam_info'][sensor][cam_id]
        #         img_info['cam_path'] = Path(
        #             self.data_root) / img_info['cam_path']
        #         cam_instances.append(img_info)
        # if len(cam_instances) > 0:
        #     ret_info['cam_instances'] = cam_instances

        # info['lidar_points'] = info.pop('pc_info')
        info['pc_info']['lidar_path'] = Path(
            self.data_root) / info['pc_info'].pop('pts_path')
        info['pc_info']['semantic_lidar_path'] = Path(
            self.data_root) / info['pc_info'].pop('semantic_pts_path')

        for cam_type, cam_list in info['cam_info'].items():
            for cam_name, cam_info in cam_list.items():
                cam_info['cam_path'] = Path(self.data_root) / cam_info['cam_path']
                if cam_type == 'cam_dvs':
                    cam_info['cam_npz_path'] = Path(
                        self.data_root) / cam_info['cam_npz_path']

        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_anno_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_anno_info(info)

        return info

    # def prepare_data(self, index: int) -> Union[dict, None]:
    #     ori_input_dict = self.get_data_info(index)
    #     input_dict = copy.deepcopy(ori_input_dict)

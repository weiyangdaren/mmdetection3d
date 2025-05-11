from typing import Any, Dict

from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS



@TRANSFORMS.register_module()
class RandomSensorDrop(BaseTransform):
    def __init__(self, drop_params: dict, drop_gt: bool = False, cam_type: str = 'cam_fisheye'):
        super().__init__()

        self.mode = drop_params.get('mode', 'num')
        self.value = drop_params.get('value', 1)
        self.drop_gt = drop_gt
        self.cam_type = cam_type
        
        # 验证参数合法性
        assert self.mode in ['num', 'name', 'id'], \
            f"Invalid mode: {self.mode}, must be one of ['num', 'name', 'id']"
        if self.mode == 'num':
            assert isinstance(self.value, int) and self.value > 0, \
                "For 'num' mode, value must be positive integer"
    
    def _get_drop_indices(self, sensor_list: list) -> list:
        """确定要丢弃的传感器索引"""
        if self.mode == 'num':
            num_drop = min(self.value, len(sensor_list))
            return random.sample(range(len(sensor_list)), num_drop)
            
        elif self.mode == 'name':
            return [i for i, name in enumerate(sensor_list) if name in self.value]
            
        elif self.mode == 'id':
            return [i for i in self.value if i < len(sensor_list)]
        
        return []

    def transform(self, results: dict) -> dict:
        """主转换函数"""
        # 获取当前摄像头列表
        cam_info = results['cam_info'][self.cam_type]
        sensor_names = list(cam_info.keys())
        
        # 确定要丢弃的索引
        drop_indices = self._get_drop_indices(sensor_names)
        keep_indices = [i for i in range(len(sensor_names)) if i not in drop_indices]
        
        # 如果没有保留任何传感器，返回原始数据
        if not keep_indices:
            return results
        
        # 更新摄像头信息
        new_cam_info = {}
        for idx in keep_indices:
            cam_name = sensor_names[idx]
            new_cam_info[cam_name] = cam_info[cam_name]
        results['cam_info'][self.cam_type] = new_cam_info

        # 更新对应的数据字段
        if self.cam_type in results:
            cam_data = results[self.cam_type]
            for key in ['img_path', 'lidar2cam', 'cam2lidar', 'lidar2img']:
                if key in cam_data:
                    if isinstance(cam_data[key], np.ndarray):
                        cam_data[key] = cam_data[key][keep_indices]
                    else:
                        cam_data[key] = [cam_data[key][i] for i in keep_indices]
            
            # 特殊处理图像数据
            if 'img' in cam_data:
                img = cam_data['img']
                if isinstance(img, list):
                    cam_data['img'] = [img[i] for i in keep_indices]
                else:
                    cam_data['img'] = img[..., keep_indices]
            
            # 更新视图数量
            cam_data['num_views'] = len(keep_indices)
        
        # # 处理GT标注（假设GT按传感器索引存储）
        # if self.drop_gt and 'gt_bboxes_3d' in results:
        #     results['gt_bboxes_3d'] = [results['gt_bboxes_3d'][i] for i in keep_indices]
        #     if 'gt_labels_3d' in results:
        #         results['gt_labels_3d'] = [results['gt_labels_3d'][i] for i in keep_indices]
        
        return results
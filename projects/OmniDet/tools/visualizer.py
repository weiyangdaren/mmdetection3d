import json
import torch
import numpy as np
from pathlib import Path

from mmdet3d.visualization import Det3DLocalVisualizer
from mmdet3d.structures import LiDARInstance3DBoxes


def read_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_label(path):
    assert Path(path).exists(), 'label file not exists: %s' % path
    labels = json.load(open(path, 'r'))
    gt_boxes = []
    gt_names = []
    gt_mesh = []
    gt_ids = []
    for label in labels:
        gt_boxes.append(label['bounding_box'])
        gt_names.append(label['class'])
        gt_mesh.append(label['name'])
        gt_ids.append(label['id'])
    return (np.array(gt_boxes, dtype=np.float32).reshape(-1, 7), 
            np.array(gt_names), 
            np.array(gt_mesh), 
            np.array(gt_ids))


token = 'train-Town01_Opt-ClearNoon-2024_09_23_11_44_45.ego0.00026981'
scene, ego, frame_id = token.split('.')
lidar_path = Path('data/CarlaCollection') / scene / ego / 'lidar' / f'{frame_id}.bin'

# lidar_path = 'data/CarlaCollection/train-Town01_Opt-ClearNoon-2024_09_23_11_44_45/ego0/lidar/00027081.bin'
label_path = str(lidar_path).replace('lidar', 'label_3d').replace('bin', 'json')
print(read_label(label_path)[0])

visualizer = Det3DLocalVisualizer()
points = read_bin(lidar_path)

# set point cloud in visualizer
visualizer.set_points(points)
bboxes_3d = LiDARInstance3DBoxes(torch.tensor(read_label(label_path)[0]), origin=(0.5, 0.5, 0.5))
# Draw 3D bboxes
visualizer.draw_bboxes_3d(bboxes_3d, bbox_color=(255, 0, 0))
visualizer.show()

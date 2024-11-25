from collections import OrderedDict
from pathlib import Path
from concurrent import futures as futures

import mmcv
import mmengine
import numpy as np
import json
from PIL import Image
from collections import defaultdict

import sys
sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from tools.omni_utils import LoadCarlaText, TransformUtils
from configs.dataset_cfg import CATEGORIES

def _read_imageset_file(path):
    assert Path(path).exists(), f'{path} does not exist'
    split_list = json.load(open(path, 'r'))
    return split_list


def _read_label(path):
    assert path.exists(), 'label file not exists: %s' % path
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
    return np.array(gt_boxes, dtype=np.float32).reshape(-1, 7), np.array(gt_names), np.array(gt_mesh), np.array(gt_ids)


def create_sensor_transform(data_path, sample_list, num_workers=4):
    def process_single_sensor(scene_ego_sensor_pair):
        scene, ego, sensor = scene_ego_sensor_pair
        sensor_transform_file = data_path / scene / ego / sensor / 'sensor_metadata.txt'
        sensor_transform = LoadCarlaText.load_sensor_transform(
            sensor_transform_file) if sensor_transform_file.exists() else None
        return (scene, ego), {sensor: sensor_transform}

    import concurrent.futures as futures
    scene_ego_set = {(item['scene_name'], item['vehicle_name'])
                     for item in sample_list}
    scene_ego_sensor_pair_list = []
    for scene, ego in scene_ego_set:
        sensor_dirs = [x for x in (
            data_path / scene / ego).iterdir() if x.is_dir()]
        for sensor_dir in sensor_dirs:
            if (sensor_dir / 'sensor_metadata.txt').exists():
                scene_ego_sensor_pair_list.append(
                    (scene, ego, sensor_dir.name))

    transform_dict = {}
    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        for (scene_ego, sensor_dict) in executor.map(process_single_sensor, scene_ego_sensor_pair_list):
            if scene_ego not in transform_dict:
                transform_dict[scene_ego] = {}
            transform_dict[scene_ego].update(sensor_dict)
    return transform_dict


def get_omni3d_info(root_path,
                    num_worker=8,
                    sample_list=None,
                    sensor_transform=None):
    root_path = Path(root_path)
    cam_rgb = ['rgb_camera_front', 'rgb_camera_left',
               'rgb_camera_right', 'rgb_camera_rear']
    cam_nusc = ['nu_rgb_camera_front', 'nu_rgb_camera_front_left', 'nu_rgb_camera_front_right',
                'nu_rgb_camera_rear_right', 'nu_rgb_camera_rear_left', 'nu_rgb_camera_rear']
    cam_fisheye = ['fisheye_camera_front', 'fisheye_camera_left',
                   'fisheye_camera_right', 'fisheye_camera_rear']
    cam_dvs = ['dvs_camera_front', 'dvs_camera_left',
               'dvs_camera_right', 'dvs_camera_rear']
    total_cam = {'cam_rgb': cam_rgb, 'cam_nusc': cam_nusc,
                 'cam_fisheye': cam_fisheye, 'cam_dvs': cam_dvs}
    view_fov = {'cam_rgb': 98.5, 'cam_nusc': 90,
                'cam_fisheye': 220, 'cam_dvs': 104.7}
    from ocamcamera import OcamCamera

    def map_func(token):
        scene_name, vehicle_name, weather, frame_id, prev_id, next_id = token['scene_name'], token[
            'vehicle_name'], token['weather'], token['frame_id'], token['prev_id'], token['next_id']

        info = {}
        token = f'{scene_name}.{vehicle_name}.{frame_id}'
        info['token'] = token
        info['metainfo'] = {'scene_name': scene_name, 'vehicle_name': vehicle_name,
                            'weather': weather, 'frame_id': frame_id, 'prev_id': prev_id, 'next_id': next_id}
        pc_info = {'num_features': 4, 'pts_path': f'{scene_name}/{vehicle_name}/lidar/{frame_id}.bin',
                   'semantic_pts_path': f'{scene_name}/{vehicle_name}/semantic_lidar/{frame_id}.bin'}
        info['pc_info'] = pc_info

        sensor_transform_info = sensor_transform[(scene_name, vehicle_name)]
        lidar_transform = sensor_transform_info['lidar'].get(frame_id, None)
        cam_info = {}
        for cam_type, cam_list in total_cam.items():
            cam_info[cam_type] = defaultdict(dict)
            for cam_name in cam_list:
                if cam_type != 'cam_dvs':
                    cam_path = f'{scene_name}/{vehicle_name}/{cam_name}/{frame_id}.png'
                    cam_info[cam_type][cam_name].update({'cam_path': cam_path})
                else:
                    cam_path = f'{scene_name}/{vehicle_name}/{cam_name}/{frame_id}_xytp.png'
                    cam_info[cam_type][cam_name].update({'cam_path': cam_path})
                    cam_npz_path = f'{scene_name}/{vehicle_name}/{cam_name}/{frame_id}_xytp.npz'
                    cam_info[cam_type][cam_name].update(
                        {'cam_npz_path': cam_npz_path})

                # get extrinsic
                cam_transform = sensor_transform_info[cam_name].get(
                    frame_id, None)
                sensor2lidar = TransformUtils.get_sensor_to_sensor_matrix(
                    cam_transform, lidar_transform, use_right_handed_coordinate=True)
                img2lidar = TransformUtils.convert_img_to_3d_coord(
                    sensor2lidar)
                lidar2sensor = TransformUtils.get_sensor_to_sensor_matrix(
                    lidar_transform, cam_transform, use_right_handed_coordinate=True)
                lidar2img = TransformUtils.convert_3d_to_img_coord(
                    lidar2sensor)
                cam_info[cam_type][cam_name].update(
                    {'sensor2lidar': sensor2lidar, 'img2lidar': img2lidar, 'lidar2sensor': lidar2sensor, 'lidar2img': lidar2img})

                # get intrinsic
                cam_path = root_path / cam_path
                image = Image.open(cam_path).convert("RGB")
                width, height = image.size  # Get width and height
                cam_info[cam_type][cam_name].update(
                    {'image_size': (width, height)})
                if cam_type != 'cam_fisheye':
                    intrinsic = TransformUtils.get_camera_intrinsics(
                        height, width, view_fov[cam_type])
                    cam_info[cam_type][cam_name].update(
                        {'cam_intrinsic': intrinsic})
                else:
                    ocam_path = root_path / 'calib_results.txt'
                    ocam = OcamCamera(filename=ocam_path,
                                      fov=view_fov[cam_type])
                    cam_info[cam_type][cam_name].update(
                        {'cam_intrinsic': ocam})
        info['cam_info'] = cam_info

        # get label
        label_path = root_path / scene_name / \
            vehicle_name/'label_3d'/(frame_id+'.json')
        gt_boxes, gt_names, gt_mesh, gt_ids = _read_label(Path(label_path))
        annos = {}
        annos['gt_boxes'] = gt_boxes
        annos['gt_names'] = gt_names
        annos['gt_mesh'] = gt_mesh
        annos['gt_ids'] = gt_ids
        info['annos'] = annos
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, sample_list)

    return list(image_infos)


def create_omni3d_infos(root_path,
                        info_prefix='omni3d',
                        version='2hz-mini',
                        save_path=None,
                        workers=8):
    metainfo = {
        'categories': CATEGORIES,
        'dataset': 'omni3d', 
        'version': version
    }

    root_path = Path(root_path)
    imageset_folder = root_path / ('ImageSets' + '-' + version)
    train_img_ids = _read_imageset_file(imageset_folder / 'train.json')
    val_img_ids = _read_imageset_file(imageset_folder / 'val.json')

    transform_path = imageset_folder / 'omni3d_sensor_transform.pkl'
    if not transform_path.exists():
        total_img_ids = train_img_ids + val_img_ids
        transform_dict = create_sensor_transform(root_path, total_img_ids)
        mmengine.dump(transform_dict, imageset_folder /
                      'omni3d_sensor_transform.pkl')
    else:
        transform_dict = mmengine.load(transform_path)

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(imageset_folder)
    else:
        save_path = Path(save_path)

    omni3d_infos_train = get_omni3d_info(root_path,
                                         workers,
                                         sample_list=train_img_ids,
                                         sensor_transform=transform_dict)
    infos_to_save = {'data_list': omni3d_infos_train, 'metainfo': metainfo}
    mmengine.dump(infos_to_save, save_path / f'{info_prefix}_infos_train.pkl')

    omni3d_infos_val = get_omni3d_info(root_path,
                                       workers,
                                       sample_list=val_img_ids,
                                       sensor_transform=transform_dict)
    infos_to_save = {'data_list': omni3d_infos_val, 'metainfo': metainfo}
    mmengine.dump(infos_to_save, save_path / f'{info_prefix}_infos_val.pkl')
    print('Done!')


if __name__ == '__main__':
    root_path = 'data/CarlaCollection'
    info_prefix = 'omni3d'
    version = '2hz-mini'
    save_path = None
    workers = 32
    create_omni3d_infos(root_path,
                        info_prefix=info_prefix,
                        version=version,
                        save_path=save_path,
                        workers=workers)

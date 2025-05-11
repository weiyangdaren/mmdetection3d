import numpy as np
import re
from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ocamcamera import OcamCamera


def get_matrix(location, rotation):
    """
    Creates matrix from carla transform.
    """
    x, y, z = location[0], location[1], location[2]
    roll, pitch, yaw = rotation[0], rotation[1], rotation[2]

    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))
    matrix = np.eye(4)
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix


def get_camera_intrinsics(img_height, img_width, view_fov):
    calibration = np.eye(4, dtype=np.float32)
    calibration[0, 2] = img_width / 2.0
    calibration[1, 2] = img_height / 2.0
    calibration[0, 0] = img_width / (2.0 * np.tan(view_fov * np.pi / 360.0))
    calibration[1, 1] = calibration[0, 0]
    return calibration

def lidar_coord_to_cam_coord(matrix):
    #           ^ z                . z
    #           |                 /
    #           |       to:      +-------> x
    #           |  . x           |
    #           | /              |
    # y <-------+                v y
    M = np.array([[ 0, -1,  0,  0 ],
                  [ 0,  0, -1,  0 ],
                  [ 1,  0,  0,  0 ],
                  [ 0,  0,  0,  1 ]])
    return np.dot(M, matrix) # first transform to camera position, then to axis of camera coord

def cam_coord_to_lidar_coord(matrix):
    #    . z                      ^ z
    #   /                         |
    #  +-------> x  to:           |
    #  |                          |  . x
    #  |                          | /
    #  v y              y <-------+
    M = np.array([[ 0,  0,  1,  0 ],
                  [-1,  0,  0,  0 ],
                  [ 0, -1,  0,  0 ],
                  [ 0,  0,  0,  1 ]])
    return np.dot(matrix, M)  # first transform to axis of lidar coord, then to lidar position


def parse_sensor_transform(filename):
    sensor_transforms = {}
    pattern = re.compile(r"""
        frame=(\d+),\s+
        .*?
        Transform\(
            Location\(x=([-.\d]+),\s+y=([-.\d]+),\s+z=([-.\d]+)\),\s+
            Rotation\(pitch=([-.\d]+),\s+yaw=([-.\d]+),\s+roll=([-.\d]+)\)
        \)
    """, re.X)

    with open(filename, 'r') as f:
        data = f.read()
        # data = f.readline()  # only first line
        matches = pattern.findall(data)
        for match in matches:
            frame_id = match[0].zfill(8)
            x = float(match[1])
            y = -float(match[2])
            z = float(match[3])
            location = [x, y, z]  # transform carla coord to lidar coord system, x, -y, z
            pitch = -float(match[4])
            yaw = -float(match[5])
            roll = float(match[6])
            # transform carla coord to lidar coord system, roll, -pitch, -yaw
            rotation = [roll, pitch, yaw]

            sensor_transforms[frame_id] = {
                'location': location,
                'rotation': rotation
            }
    return sensor_transforms


class OmniCalib:
    VIEW_FOV = {
        'cam_rgb': 98.5, 'cam_nusc': 90, 'cam_fisheye': 220, 'cam_dvs': 104.7
    }
    def __init__(self, cam_type, sensor_transform, lidar_transform, img_height, img_width, ocam_path=None):
        sensor2world = get_matrix(
            sensor_transform['location'], sensor_transform['rotation'])
        lidar2world = get_matrix(
            lidar_transform['location'], lidar_transform['rotation'])
        world2sensor = np.linalg.inv(sensor2world)
        world2lidar = np.linalg.inv(lidar2world)
        sensor2lidar = np.dot(world2lidar, sensor2world)
        lidar2sensor = np.dot(world2sensor, lidar2world)

        self.cam2lidar = cam_coord_to_lidar_coord(sensor2lidar)
        self.lidar2cam = lidar_coord_to_cam_coord(lidar2sensor)
        
        if cam_type != 'cam_fisheye':
            self.cam2img = get_camera_intrinsics(
                img_height, img_width, self.VIEW_FOV[cam_type])
            self.lidar2img = np.dot(self.cam2img, self.lidar2cam)
        else:
            self.cam2img = OcamCamera(
                filename=ocam_path, fov=self.VIEW_FOV[cam_type])
            self.lidar2img = np.eye(4)


def load_transform(data_root):
    sensor_list = [x.stem for x in data_root.iterdir(
    ) if x.is_dir() and (x / 'sensor_metadata.txt').exists()]
    sensor_transforms = {}
    for sensor in sensor_list:
        sensor_transform = parse_sensor_transform(
            data_root / sensor / 'sensor_metadata.txt')
        sensor_transforms[sensor] = sensor_transform
    return sensor_transforms


def read_img(img_path):
    return np.array(Image.open(img_path).convert("RGB")), Image.open(img_path).size


def read_depth(depth_path):
    depth_image = Image.open(depth_path)
    depth_array = np.array(depth_image, dtype=np.float32)  # 转为 numpy 数组
    depth_array_normalized = depth_array[:, :, 0] + depth_array[:,
                                                                :, 1] * 256.0 + depth_array[:, :, 2] * 256.0 * 256.0
    depth_array_normalized /= (256.0 * 256.0 * 256.0 - 1)
    depth_in_meters = depth_array_normalized * 1000
    return depth_in_meters, depth_image.size


def read_lidar(lidar_path):
    return np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]

def read_label(label_path):
    import json
    from mmdet3d.structures import LiDARInstance3DBoxes
    labels = json.load(open(label_path, 'r'))
    gt_boxes = []
    for label in labels:
        gt_boxes.append(label['bounding_box'])
    gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1, 7)
    return LiDARInstance3DBoxes(gt_boxes, origin=(0.5, 0.5, 0.5))

def project_points_to_image(points, lidar2img, img_width, img_height):
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    points_img = lidar2img @ points_homogeneous.T  # (4, N)
    points_img = points_img.T  # (N, 4)
    points_img[:, 0] /= points_img[:, 2]  # x / z
    points_img[:, 1] /= points_img[:, 2]  # y / z
    valid_mask = points_img[:, 2] > 0
    points_img = points_img[valid_mask]

    u, v = points_img[:, 0], points_img[:, 1]
    valid_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)

    return u[valid_mask], v[valid_mask]


def visualize_projection(img, u, v):

    # 绘制点云投影
    for x, y in zip(u, v):
        cv2.circle(img, (int(x), int(y)), radius=2,
                   color=(0, 255, 0), thickness=-1)

    # 显示结果
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Projected Point Cloud on Image")
    plt.show()


def draw_3d_boxes(image, imgfov_pts_2d, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding boxes projected onto a 2D image.

    Args:
        image_path (str): Path to the image file.
        imgfov_pts_2d (np.ndarray): n*8*2 array containing 2D coordinates of the vertices of 3D bounding boxes.
        color (tuple): RGB color for the bounding box lines.
        thickness (int): Thickness of the bounding box lines.

    Returns:
        np.ndarray: Image with bounding boxes drawn on it.
    """

    imgfov_pts_2d = np.array(imgfov_pts_2d, dtype=np.int32)  # Ensure integer coordinates

    # Define the edges of a 3D box (connect the 8 vertices)
    box_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Draw each bounding box
    for bbox in imgfov_pts_2d:
        for edge in box_edges:
            pt1 = tuple(bbox[edge[0]])  # Start point of the edge
            pt2 = tuple(bbox[edge[1]])  # End point of the edge
            cv2.line(image, pt1, pt2, color, thickness)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Projected 3D boxes on Image")
    plt.show()

    return image


def project_depth_to_points(depth, cam2img, cam2lidar, img_width, img_height):
    """  
    将深度图像转换为3D点云坐标  

    参数:  
    - depth: 深度图像，numpy数组，形状为 (height, width)  
    - cam2img: 从相机到图像的内参矩阵，4x4 numpy数组  
    - cam2lidar: 从相机到激光雷达的变换矩阵，4x4 numpy数组  
    - img_width: 图像宽度  
    - img_height: 图像高度  

    返回:  
    - points: 3D点云坐标，形状为 (N, 3)，其中N是有效点的数量  
    """
    # 创建网格坐标
    x, y = np.meshgrid(np.arange(img_width), np.arange(img_height))

    # 展平坐标和深度
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth_values = depth.reshape(-1)

    # 只保留深度值有效的点
    valid_mask = depth_values > 0
    x = x[valid_mask]
    y = y[valid_mask]
    depth_values = depth_values[valid_mask]

    # 根据 cam2img 内参计算相机坐标系下的 3D 点
    # 使用内参矩阵构造齐次坐标
    Z = depth_values
    # cam2img[0, 2] = cx; cam2img[0, 0] = fx
    X = (x - cam2img[0, 2]) * Z / cam2img[0, 0]
    # cam2img[1, 2] = cy; cam2img[1, 1] = fy
    Y = (y - cam2img[1, 2]) * Z / cam2img[1, 1]

    # 构建齐次坐标
    camera_points = np.column_stack([X, Y, Z, np.ones_like(Z)])

    # 使用 cam2lidar 变换到 LiDAR 坐标系
    lidar_points = (cam2lidar @ camera_points.T).T

    # 返回 3D 点云坐标（去掉齐次坐标的第四维）
    return lidar_points[:, :3]


def mask_points_by_range(points, point_range):
    valid_mask = (points[:, 0] > point_range[0]) & (points[:, 0] < point_range[3]) & \
                 (points[:, 1] > point_range[1]) & (points[:, 1] < point_range[4]) & \
                 (points[:, 2] > point_range[2]) & (
                     points[:, 2] < point_range[5])
    return points[valid_mask]


def proj_lidar_bbox3d_to_img(bboxes_3d,
                             lidar2img) -> np.ndarray:
    """Project the 3D bbox on 2D plane.

    Args:
        bboxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bbox in lidar coordinate
            system to visualize.
        input_meta (dict): Meta information.
    """
    corners_3d = bboxes_3d.corners.cpu().numpy()
    num_bbox = corners_3d.shape[0]
    pts_4d = np.concatenate(
        [corners_3d.reshape(-1, 3),
         np.ones((num_bbox * 8, 1))], axis=-1)

    pts_2d = pts_4d @ lidar2img.T

    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=1e5)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    imgfov_pts_2d = pts_2d[..., :2].reshape(num_bbox, 8, 2)

    return imgfov_pts_2d


if __name__ == '__main__':
    data_root = Path('data/test_data')
    cam_name = 'nu_rgb_camera_front_left'
    frame_id = '00026981'
    sensor_transforms = load_transform(data_root)
    lidar_transform = sensor_transforms['lidar']

    # ====================== points to img ======================
    sensor_transform = sensor_transforms[cam_name]
    img, (img_w, img_h) = read_img(data_root / cam_name / f'{frame_id}.png')
    lidar = read_lidar(data_root / 'lidar' / f'{frame_id}.bin')
    omni_calib = OmniCalib('cam_nusc', sensor_transform[frame_id], lidar_transform[frame_id], img_h, img_w)
    u, v = project_points_to_image(lidar, omni_calib.lidar2img, img_w, img_h)
    visualize_projection(img, u, v)

    gt_boxes = read_label(data_root / 'label_3d' / f'{frame_id}.json')
    imgfov_pts_2d = proj_lidar_bbox3d_to_img(gt_boxes, omni_calib.lidar2img)

    img = draw_3d_boxes(img, imgfov_pts_2d, color=(255, 0, 0))

    print('############')

    # ====================== depth to points ======================

    cam_name = 'depth_camera_left'
    depth, (depth_w, depth_h) = read_depth(
        data_root / cam_name / f'{frame_id}.png')
    
    plt.imshow(depth)
    plt.show()

    sensor_transform = sensor_transforms[cam_name]
    omni_calib = OmniCalib(
        'cam_nusc', sensor_transform[frame_id], lidar_transform[frame_id], depth_h, depth_w)

    points = project_depth_to_points(
        depth, omni_calib.cam2img, omni_calib.cam2lidar, depth_w, depth_h)
    points = mask_points_by_range(points, [-50, -50, -5, 50, 50, 5])
    points_color = np.zeros((points.shape[0], 3))
    points_color[:, 1] = 1  # green

    lidar = read_lidar(data_root / 'lidar' / f'{frame_id}.bin')
    lidar_color = np.zeros((lidar.shape[0], 3))
    lidar_color[:, 0] = 1  # read

    import open3d_vis_utils as V
    show_pts = np.vstack((points, lidar))
    show_color = np.vstack((points_color, lidar_color))
    V.draw_scenes_v2(show_pts, point_colors=show_color)
    # V.draw_scenes_v2(points, point_colors=points_color)

import re
import numpy as np
import torch
from scipy.spatial.transform import Rotation


class LoadCarlaText(object):
    '''
    Note that the coordinate system in CARLA is different from the one in PCDet.
    All the coordinates in CARLA are in left-handed coordinate system, while the coordinates in PCDet are in right-handed coordinate system.
    '''
    @staticmethod
    def load_ego_transform(ego_transform_file):
        ego_transforms = {}

        pattern = re.compile(r"""
            (\d+): \s+                         # frame ID
            Transform\(Location\(x=([-.\d]+), \s+ y=([-.\d]+), \s+ z=([-.\d]+)\), \s+  # location (x, y, z)
            Rotation\(pitch=([-.\d]+), \s+ yaw=([-.\d]+), \s+ roll=([-.\d]+)\)\)        # rotation (pitch, yaw, roll)
        """, re.X)

        with open(ego_transform_file, 'r') as f:
            data = f.read()
            matches = pattern.findall(data)
            
            for match in matches:
                frame_id = match[0]
                x = float(match[1])
                y = float(match[2])
                z = float(match[3])
                location = [x, y, z]  # transform to lidar coord system
                pitch = float(match[4])
                yaw = float(match[5])
                roll = float(match[6])
                rotation = [roll, pitch, yaw]  # transform to lidar coord system
                
                ego_transforms[frame_id] = {
                    'location': location,
                    'rotation': rotation
                }
        return ego_transforms
    
    @staticmethod
    def load_sensor_transform(sensor_transform_file):
        sensor_transforms = {}
    
        pattern = re.compile(r"""
            frame=(\d+),\s+
            .*?
            Transform\(
                Location\(x=([-.\d]+),\s+y=([-.\d]+),\s+z=([-.\d]+)\),\s+
                Rotation\(pitch=([-.\d]+),\s+yaw=([-.\d]+),\s+roll=([-.\d]+)\)
            \)
        """, re.X)
        
        with open(sensor_transform_file, 'r') as f:
            data = f.read()
            matches = pattern.findall(data)
            for match in matches:
                frame_id = match[0].zfill(8)
                x = float(match[1])
                y = float(match[2])
                z = float(match[3])
                location = [x, y, z]  # transform to lidar coord system
                pitch = float(match[4])
                yaw = float(match[5])
                roll = float(match[6])
                rotation = [roll, pitch, yaw]  # transform to lidar coord system, roll, pitch, yaw

                sensor_transforms[frame_id] = {
                    'location': location,
                    'rotation': rotation
                }
        return sensor_transforms

    @staticmethod
    def load_gnss_data(gnss_file):
        gnss_data = {}
        pattern = re.compile(r"""
            GnssMeasurement\(frame=(\d+),\s+ # frame
            timestamp=([\d.]+),\s+            # timestamp
            lat=([\d.-]+),\s+                 # lat
            lon=([\d.-]+),\s+                 # lon
            alt=([\d.-]+)\),\s+               # alt
            Transform\(Location\(x=([\d.-]+),\s+ y=([\d.-]+),\s+ z=([\d.-]+)\),\s+  # location (x, y, z)
            Rotation\(pitch=([\d.-]+),\s+ yaw=([\d.-]+),\s+ roll=([\d.-]+)\)\)      # rotation (pitch, yaw, roll)
        """, re.X)

        with open(gnss_file, 'r') as file:
            data = file.read()
            matches = pattern.findall(data)
            
            for match in matches:
                frame_id = match[0].zfill(8)
                timestamp = float(match[1])
                lat = float(match[2])
                lon = float(match[3])
                alt = float(match[4])

                x = float(match[5])
                y = float(match[6])
                z = float(match[7])
                location = [x, y, z]  # transform to lidar coord system
                pitch = float(match[8])
                yaw = float(match[9])
                roll = float(match[10])
                rotation = [roll, pitch, yaw]  # transform to lidar coord system

                gnss_data[frame_id] = {
                    'timestamp': timestamp,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt,
                    'location': location,
                    'rotation': rotation
                }
        return gnss_data
    
    @staticmethod
    def load_imu_data(imu_file):
        imu_data = {}

        pattern = re.compile(r"""
            IMUMeasurement\(frame=(\d+),\s+                # frame
            timestamp=([\d.]+),\s+                          # timestamp
            accelerometer=Vector3D\(x=([-.\d]+), \s+ y=([-.\d]+), \s+ z=([-.\d]+)\),\s+   # accelerometer (x, y, z)
            gyroscope=Vector3D\(x=([-.\d]+), \s+ y=([-.\d]+), \s+ z=([-.\d]+)\),\s+       # gyroscope (x, y, z)
            compass=([-.\d]+)\),\s+                         # compass
            Transform\(Location\(x=([-.\d]+), \s+ y=([-.\d]+), \s+ z=([-.\d]+)\),\s+      # location (x, y, z)
            Rotation\(pitch=([-.\d]+), \s+ yaw=([-.\d]+), \s+ roll=([-.\d]+)\)\)          # rotation (pitch, yaw, roll)
        """, re.X)

        with open(imu_file, 'r') as f:
            data = f.read()
            matches = pattern.findall(data)
            
            for match in matches:
                frame_id = match[0].zfill(8)
                timestamp = float(match[1])

                accelerometer_x = float(match[2])
                accelerometer_y = float(match[3])
                accelerometer_z = float(match[4])
                accelerometer = [accelerometer_x, accelerometer_y, accelerometer_z]  # transform to lidar coord system
                gyroscope_x = float(match[5])
                gyroscope_y = float(match[6])
                gyroscope_z = float(match[7])
                gyroscope = [gyroscope_x, gyroscope_y, gyroscope_z]  # transform to lidar coord system
                compass = float(match[8])

                x = float(match[9])
                y = float(match[10])
                z = float(match[11])
                location = [x, y, z]  # transform to lidar coord system
                pitch = float(match[12])
                yaw = float(match[13])
                roll = float(match[14])
                rotation = [roll, pitch, yaw]  # transform to lidar coord system

                imu_data[frame_id] = {
                    'timestamp': timestamp,
                    'accelerometer': accelerometer,
                    'gyroscope': gyroscope,
                    'compass': compass,
                    'location': location,
                    'rotation': rotation
                }
        return imu_data
    
    @staticmethod
    def load_sensor_transform_by_frame_id(sensor_transform_file, frame_id):
        sensor_transforms = LoadCarlaText.load_sensor_transform(sensor_transform_file)
        return sensor_transforms[frame_id]
    

class TransformUtils(object):
    @staticmethod
    def get_camera_intrinsics(img_height, img_width, view_fov):
        '''
        create a camera intrinsic matrix.
        '''
        calibration = np.eye(3, dtype=np.float32)
        calibration[0, 2] = img_width / 2.0
        calibration[1, 2] = img_height / 2.0
        calibration[0, 0] = img_width / (2.0 * np.tan(view_fov * np.pi / 360.0))
        calibration[1, 1] = img_width / (2.0 * np.tan(view_fov * np.pi / 360.0))  # fu == fv ???
        # calibration[0, 0] = calibration[1, 1] = 180 # ???
        return calibration

    @staticmethod
    def get_transform_matrix(transform):
        '''
        create a transformation matrix relative to world coordinate system.
        '''
        location, rotation = transform['location'], transform['rotation']
        x, y, z = location[0], location[1], location[2]
        roll, pitch, yaw = rotation[0], rotation[1], rotation[2]
        R_matrix = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=True).as_matrix()
        T_matrix = np.array([x, y, z])
        trans_matrix = np.eye(4)
        trans_matrix[:3, :3] = R_matrix
        trans_matrix[:3, 3] = T_matrix
        return trans_matrix.astype(np.float32)
    
    @staticmethod
    def get_sensor_to_sensor_matrix(sensor_transform, other_sensor_transform, use_right_handed_coordinate=False):
        '''
        return a transformation matrix from sensor to other sensor.
        use_right_handed_coordinate: the coordinate system in CARLA is different from the one in PCDet.
        CARLA: left-handed coordinate system, PCDet: right-handed coordinate system.
        '''
        sensor_to_world_matrix = TransformUtils.get_transform_matrix(sensor_transform)
        other_sensor_to_world_matrix = TransformUtils.get_transform_matrix(other_sensor_transform)
        world_to_other_sensor_matrix = np.linalg.inv(other_sensor_to_world_matrix)
        # Let A, B, and C be rotation matrices. C = np.dot(A, B) means applying rotation B first,
        # followed by rotation A, resulting in the combined rotation C.
        sensor_to_other_sensor_matrix = np.dot(world_to_other_sensor_matrix, sensor_to_world_matrix) 

        if use_right_handed_coordinate:
            R_left = sensor_to_other_sensor_matrix[:3, :3]
            T_left = sensor_to_other_sensor_matrix[:3, 3]
            R_right = R_left.T  # reverse rotation
            T_right = T_left
            T_right[1] = -T_right[1]  # reverse y-axis
            sensor_to_other_sensor_matrix[:3, :3] = R_right
            sensor_to_other_sensor_matrix[:3, 3] = T_right

        return sensor_to_other_sensor_matrix
    
    def convert_img_to_3d_coord(transform_matrix):
        '''
            transform image coordinate direction to lidar coordinate direction in right-handed coordinate system.
            image coordinate system ==> lidar coordinate system   

              / z                  z |  / x
             /                       | /
            -----> x   ===>   y <-----
            |
            | y
        '''
        rot_matrix = np.array([
            [ 0,  0,  1, 0],
            [-1,  0,  0, 0],
            [ 0, -1,  0, 0],
            [ 0,  0,  0, 1]
        ], dtype=np.float32)
    
        transform_matrix = np.dot(transform_matrix, rot_matrix)  #先旋转成LiDAR坐标系，再执行sensor的变换
        return transform_matrix
    
    def convert_3d_to_img_coord(transform_matrix):
        # 3d coordinate to image coordinate
        rot_matrix = np.array([
            [ 0, -1,  0, 0],
            [ 0,  0, -1, 0],
            [ 1,  0,  0, 0],
            [ 0,  0,  0, 1]
        ], dtype=np.float32)
    
        transform_matrix = np.dot(rot_matrix, transform_matrix)  # 先执行sensor的变换，再旋转成Image坐标系
        return transform_matrix


    @staticmethod
    def image_coord_to_lidar_coord(pts, reverse=False):
        '''
        transform image coordinate direction to lidar coordinate direction.
        image coordinate system ==> lidar coordinate system   

          / z                  z |  / x
         /                       | /
        -----> x   ===>   y <-----
        |
        | y

        pts: (N, 3) image coordinate points.
        reverse: reverse the transformation.

        '''
        Rot_matrix = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        if reverse:
            Rot_matrix = np.linalg.inv(Rot_matrix)
        
        if isinstance(pts, torch.Tensor):
            return torch.matmul(pts, torch.tensor(Rot_matrix, dtype=pts.dtype, device=pts.device))
        else:
            return np.dot(pts, Rot_matrix)
    


if __name__ == '__main__':
    transform1 = {
        'location': [0, 0, 0],
        'rotation': [0, 0, 0]
    }
    transform2 = {
        'location': [2, 0, 0],
        'rotation': [0, 0, 90]
    }

    point = np.array([2, 1, 0, 1])

    sensor_to_sensor_matrix = TransformUtils.get_sensor_to_sensor_matrix(transform1, transform2, use_right_handed_coordinate=True)
  
    new_point = np.dot(sensor_to_sensor_matrix, point.T).T
    print(new_point)
        

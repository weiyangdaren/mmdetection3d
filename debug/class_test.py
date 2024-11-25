a = ['points', 'lidar_points']

b = {
    'points': 'points',
    # 'lidar_points': 'lidar_points',
    'cam_points': 'cam_points',
    'cam_dvs_points': 'cam_dvs_points',
}

print(a & b.keys())
if a & b.keys():
    print('yes')
else:
    print('no')
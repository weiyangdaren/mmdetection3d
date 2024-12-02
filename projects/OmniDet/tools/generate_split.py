import json
import argparse
from pathlib import Path


DEFAULT_FRAME_RATE = 10


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/mnt/nas2/Dataset/CarlaCollection', help='dataset dir.')
    parser.add_argument('--frame_rate', type=int,
                        default=2, help='Sample frame rate.')
    parser.add_argument('--save_ext', type=str,
                        default='mini', help='save ext.')
    parser.add_argument('--split_train', type=float,
                        default=0.8, help='Train split ratio.')
    parser.add_argument('--cross_ratio', type=float,
                        default=0, help='Train split ratio.')
    args = parser.parse_args()
    return args


def get_split_set(frame_id_list, scene, vehicle):
    split_set = []
    for i, frame_id in enumerate(frame_id_list):
        this_set = {}
        prev_id = frame_id_list[i-1] if i > 0 else ''
        next_id = frame_id_list[i+1] if i < len(frame_id_list) - 1 else ''
        weather = scene.split('-')[2]
        this_set['scene_name'] = scene
        this_set['vehicle_name'] = vehicle
        this_set['weather'] = weather
        this_set['frame_id'] = frame_id
        this_set['prev_id'] = prev_id
        this_set['next_id'] = next_id
        split_set.append(this_set)
    return split_set


def save_split(split_samples, root_dir, save_ext, frame_rate=1, split_ratio=0.8, cross_ratio=0):

    mini_scene = ['train-Town10HD_Opt-ClearNoon-2024_10_29_19_07_42',
                  'train-Town10HD_Opt-CloudyNoon-2024_10_06_13_57_50',
                  'train-Town10HD_Opt-SoftRainNoon-2024_10_30_10_04_08']

    assert not (split_ratio == 0 and cross_ratio ==
                0), "split ratio and cross ratio cannot both be zero."
    assert not (split_ratio != 0 and cross_ratio !=
                0), "split ratio and cross ratio cannot both be non-zero."
    save_dir = root_dir / f'ImageSets-{frame_rate}hz-{save_ext}'
    save_dir.mkdir(exist_ok=True)

    interval = DEFAULT_FRAME_RATE // frame_rate
    train_split_set = []
    val_split_set = []
    test_split_set = []
    for scene in split_samples.keys():
        for vehicle in split_samples[scene].keys():
            frame_id_list = sorted(split_samples[scene][vehicle])[::interval]
            if scene not in mini_scene:
                continue

            if split_ratio != 0:
                num_train = int(len(frame_id_list) * split_ratio)
                train_id_list = frame_id_list[:num_train]
                val_id_list = frame_id_list[num_train:]
                train_split_set += get_split_set(train_id_list, scene, vehicle)
                val_split_set += get_split_set(val_id_list, scene, vehicle)
            if cross_ratio != 0:
                num_frame = int(len(frame_id_list) * cross_ratio)
                for i in range(0, len(frame_id_list), 2 * num_frame):
                    train_id_list = frame_id_list[i:i+num_frame]
                    val_id_list = frame_id_list[i+num_frame:i+2*num_frame]
                    train_split_set += get_split_set(train_id_list, scene, vehicle)
                    val_split_set += get_split_set(val_id_list, scene, vehicle)
            
            # else:
            #     test_split_set += get_split_set(frame_id_list, scene, vehicle)

    with open(save_dir / 'train.json', 'w') as f:
        json.dump(train_split_set, f, indent=4)
    with open(save_dir / 'val.json', 'w') as f:
        json.dump(val_split_set, f, indent=4)
    with open(save_dir / 'test.json', 'w') as f:
        json.dump(test_split_set, f, indent=4)


def generate_split():
    args = get_args()
    root_dir = Path(args.root_dir)
    scenes_name = [x.stem for x in root_dir.iterdir() if x.is_file()
                   and x.suffix == '.json']
    print(f"Found {len(scenes_name)} scenes.")
    total_samples = {}
    for scene in scenes_name:
        total_samples[scene] = {}
        scene_dir = root_dir / scene
        vehicle_name = [x.stem for x in scene_dir.iterdir() if x.is_dir()]
        for vehicle in vehicle_name:
            total_samples[scene][vehicle] = []
            with open(scene_dir / vehicle / 'ego_transform.txt', 'r') as f:
                for line in f:
                    frame_id = line.split(':')[0].strip()
                    total_samples[scene][vehicle].append(frame_id)

    save_split(total_samples, root_dir, args.save_ext, args.frame_rate, args.split_train, args.cross_ratio)


if __name__ == '__main__':
    generate_split()

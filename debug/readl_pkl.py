import pickle
import mmengine


def read_pkl(file_path):
    data = mmengine.load(file_path)
    return data

if __name__ == '__main__':
    file_path = 'data/CarlaCollection/ImageSets-2hz-0.5-all/omni3d_infos_train.pkl'
    data = read_pkl(file_path)
    print('#############')
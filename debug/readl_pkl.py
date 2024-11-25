import pickle

CATEGORIES = {
    "Car": 0,
    "Van": 1,
    "Truck": 2,
    "Bus": 3,
    "Pedestrian": 4,
    "Cyclist": 5,
}
version = '2hz-all'

metainfo = {
        'categories': CATEGORIES,
        'dataset': 'omni3d', 
        'version': version
    }


def modify_data(data):
    for i in range(len(data)):
        # token = data[i]['metadata'].pop('token')
        # data[i].update({'token': token})
        data[i]['token'] = data[i]['metadata'].pop('token')
        data[i]['metainfo'] = data[i].pop('metadata')
    return data


if __name__ == '__main__':
    with open('data/CarlaCollection/ImageSets-2hz-all/mmdet3d_infos_val.pkl', 'rb') as f:
        data = pickle.load(f)
    # print(data.keys())
    infos = data['infos']
    metadata = data.pop('metadata')
    infos = modify_data(infos)

    with open('data/CarlaCollection/ImageSets-2hz-all/omni3d_infos_val.pkl', 'wb') as f:
        pickle.dump({'data_list': infos, 'metainfo': metainfo}, f)
    
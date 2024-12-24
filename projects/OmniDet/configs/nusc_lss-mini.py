_base_ = ['./nusc_lss.py']

dataset_type = 'Omni3DDataset'
data_root = 'data/CarlaCollection/'
classes = ['Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
detect_range = [-48, -48, -5, 48, 48, 5]
cam_type='cam_nusc'
train_ann_file = 'ImageSets-2hz-mini/omni3d_infos_train.pkl'
val_ann_file = 'ImageSets-2hz-mini/omni3d_infos_train.pkl'

backend_args = None

train_pipeline = [
    # dict(
    #     type='LoadOmni3DMultiViewImageFromFiles',
    #     to_float32=True,
    #     color_type='color',
    #     backend_args=backend_args,
    #     load_cam_type='cam_fisheye',
    #     load_cam_names=['fisheye_camera_front', 'fisheye_camera_left',
    #                     'fisheye_camera_right', 'fisheye_camera_rear']),
    dict(
        type='LoadOmni3DMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args,
        load_cam_type=cam_type,
        load_cam_names=['nu_rgb_camera_front', 'nu_rgb_camera_front_left',
                        'nu_rgb_camera_front_right', 'nu_rgb_camera_rear',
                        'nu_rgb_camera_rear_right', 'nu_rgb_camera_rear_left']),

    # dict(
    #     type='LoadOmni3DPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=4,
    #     use_dim=3,
    #     backend_args=backend_args,
    #     load_point_type='lidar'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,),
    dict(
        type='ImageAug3D',
        final_dim=[400, 800],
        resize_lim=[0.625, 0.625],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0, 0],
        rand_flip=False,
        is_train=False,
        img_key=cam_type,),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=detect_range),
    dict(
        type='OmniPack3DDetInputs',
        keys=[cam_type, 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'cam2lidar', 'lidar2cam', 'lidar2img', 'img_aug_matrix', 'box_type_3d'],
        input_img_keys=[cam_type],)
]

test_pipeline = [
    dict(
        type='LoadOmni3DMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args,
        load_cam_type=cam_type,
        load_cam_names=['nu_rgb_camera_front', 'nu_rgb_camera_front_left',
                        'nu_rgb_camera_front_right', 'nu_rgb_camera_rear',
                        'nu_rgb_camera_rear_right', 'nu_rgb_camera_rear_left']),
    dict(
        type='ImageAug3D',
        final_dim=[400, 800],
        resize_lim=[0.625, 0.625],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False,
        img_key=cam_type,),
    dict(
        type='OmniPack3DDetInputs',
        keys=[cam_type, 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'cam2lidar', 'lidar2img', 'lidar2cam',
            'sample_idx', 'token', 'img_path', 'lidar_path', 
            'num_pts_feats', 'img_aug_matrix', 'box_type_3d',],
        input_img_keys=[cam_type],)
]


# CBGSDataset
train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=train_ann_file,
            pipeline=train_pipeline,
            test_mode=False,
            metainfo=dict(classes=classes),
        )
    )
)


# train_dataloader = dict(
#     batch_size=4,
#     num_workers=16,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=train_ann_file,
#         pipeline=train_pipeline,
#         test_mode=False,
#         metainfo=dict(classes=classes),
#     )
# )


val_dataloader = dict(
    batch_size=4,
    num_workers=16,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes),
    )
)
test_dataloader = val_dataloader


learning_rate = 0.0001
max_epochs = 24
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=max_epochs,
        end=max_epochs,
        by_epoch=True,
        eta_min_ratio=1e-4,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 1 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=int(0.4*max_epochs),
        eta_min=0.85 / 0.95,
        begin=0,
        end=0.4*max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=max_epochs-int(0.4*max_epochs),
        eta_min=1,  # 1.0 stand for no momentum
        begin=0.4*max_epochs,
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=learning_rate, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))


train_cfg = dict(type='EpochBasedTrainLoop',
                 max_epochs=max_epochs, val_interval=1)
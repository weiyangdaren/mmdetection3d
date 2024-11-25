_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.OmniDet.omni_dataset',
             'projects.OmniDet.omni_models'],
    allow_failed_imports=False)
# custom_imports = dict(imports=['projects.example_project.dummy'])

dataset_type = 'Omni3DDataset'
data_root = 'data/CarlaCollection/'
data_prefix = dict(
    pts='lidar')

backend_args = None

train_pipeline = [
    dict(
        type='LoadOmni3DMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args,
        load_cam_type='cam_nusc',
        load_cam_names=['nu_rgb_camera_front', 'nu_rgb_camera_front_left', 
                        'nu_rgb_camera_front_right', 'nu_rgb_camera_rear', 
                        'nu_rgb_camera_rear_right', 'nu_rgb_camera_rear_left']),
        # load_cam_type='cam_fisheye',
        # load_cam_names=['fisheye_camera_front', 'fisheye_camera_left', 
        #                 'fisheye_camera_right', 'fisheye_camera_rear']),
    dict(
        type='LoadOmni3DPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=3,
        backend_args=backend_args,
        load_point_type='lidar'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,),
    # dict(
    #     type='ImageAug3D',
    #     final_dim=[256, 704],
    #     resize_lim=[0.38, 0.55],
    #     bot_pct_lim=[0.0, 0.0],
    #     rot_lim=[-5.4, 5.4],
    #     rand_flip=True,
    #     is_train=True,
    #     img_key='cam_nusc',),
    dict(
        type='OmniPack3DDetInputs',
        keys=['cam_fisheye', 'lidar_points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar', 'img2lidar', 'img_aug_matrix'])
]

model = dict(
    type='OmniDet',
    modality=['img', 'lidar'],
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        in_channels=3,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 2, 4),
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet18')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5),
)

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets-2hz-all/omni3d_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
    )
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=24,
        by_epoch=True,
        eta_min=1e-6,
        convert_to_iter_based=True)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=16, val_interval=1)
# val_cfg = dict(type='ValLoop')

find_unused_parameters = False

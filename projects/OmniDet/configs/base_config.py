_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.OmniDet.dataset',
             'projects.OmniDet.models',
             'projects.OmniDet.models.omni_lss',],
    allow_failed_imports=False)
# custom_imports = dict(imports=['projects.example_project.dummy'])

dataset_type = 'Omni3DDataset'
data_root = 'data/CarlaCollection/'
# classes = ('Pedestrian', 'Cyclist', 'Car')
detect_range = [-48, -48, -3, 48, 48, 1]

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
        load_cam_type='cam_nusc',
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
        resize_lim=[0.38, 0.55],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[-5.4, 5.4],
        rand_flip=True,
        is_train=True,
        img_key='cam_nusc',),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=detect_range),
    dict(
        type='OmniPack3DDetInputs',
        keys=['cam_nusc', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'img2lidar', 'img_aug_matrix'])
]

model = dict(
    type='OmniLSS',
    extra_config=dict(
        img_key='cam_nusc',
        lidar_key='lidar_points',
        depth_supervision=False,),
    data_preprocessor=dict(
        type='OmniDet3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        img_keys=['cam_nusc'],),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        in_channels=3,
        base_channels=64,
        num_stages=4,
        strides=(2, 2, 2, 2),
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
    view_transform=dict(
        type='LSSTransform',
        in_channels=256,
        out_channels=96,
        image_size=[400, 800],
        feature_size=[50, 100],
        xbound=[-48.0, 48.0, 0.3],
        ybound=[-48.0, 48.0, 0.3],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[0.5, 48.5, 0.5],
        downsample=2),
    pts_backbone=dict(
        type='SECOND',
        in_channels=96,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[128, 128],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=6,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-48, -48, -0.6, 48, 48, -0.6],
                [-48, -48, -0.6, 48, 48, -0.6],
                [-48, -48, -0.6, 48, 48, -0.6],
                [-48, -48, -0.6, 48, 48, -0.6],
                [-36, -36, -0.6, 36, 36, -0.6],
                [-42, -42, -0.6, 42, 42, -0.6],
            ],
            sizes=[[4.50, 1.95, 1.58],
                   [5.37, 2.06, 2.26],
                   [6.95, 2.71, 2.97],
                   [10.27, 3.94, 4.25],
                   [0.38, 0.38, 1.78],
                   [1.95, 0.78, 1.60]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2),
        # model training and testing settings
        train_cfg=dict(
            assigner=[
                dict(  # for Car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(  # for Van
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(  # for Truck
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(  # for Bus
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1),
                dict(  # for Pedestrian
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.25,
                    min_pos_iou=0.25,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_thr=0.01,
            score_thr=0.1,
            min_bbox_size=0,
            nms_pre=100,
            max_num=50)),
)

train_dataloader = dict(
    batch_size=8,
    num_workers=16,
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='ImageSets-2hz-all/omni3d_infos_train.pkl',
            pipeline=train_pipeline,
            test_mode=False,
            # metainfo=dict(classes=classes),
        )
    )
)

learning_rate = 0.00005
max_epochs = 20

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
        eta_min=0.85 / 0.95,
        begin=0,
        end=int(0.8*max_epochs),
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        eta_min=1,  # 1.0 stand for no momentum
        begin=int(0.8*max_epochs),
        end=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=learning_rate, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
# val_cfg = dict(type='ValLoop')

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1))

find_unused_parameters = False

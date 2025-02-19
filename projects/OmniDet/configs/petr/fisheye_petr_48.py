_base_ = ['../default_runtime.py']

custom_imports = dict(
    imports=['projects.OmniDet.utils',
             'projects.OmniDet.dataset',
             'projects.OmniDet.models',
             'projects.OmniDet.models.omni_petr',],
    allow_failed_imports=False)
# custom_imports = dict(imports=['projects.example_project.dummy'])

dataset_type = 'Omni3DDataset'
data_root = 'data/CarlaCollection/'
classes = ['Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
voxel_size = [0.2, 0.2, 10]
ref_range = 48
detect_range = [-ref_range, -ref_range, -5, ref_range, ref_range, 5]
cam_type='cam_fisheye'
cam_fov = 220
train_ann_file = 'ImageSets-2hz-0.7-all/omni3d_infos_train.pkl'
val_ann_file = 'ImageSets-2hz-0.7-all/omni3d_infos_val.pkl'
backend_args = None

train_pipeline = [
    dict(
        type='LoadOmni3DMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args,
        load_cam_type=cam_type,
        load_cam_names=['fisheye_camera_front', 'fisheye_camera_left',
                        'fisheye_camera_right', 'fisheye_camera_rear',
                        ]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=detect_range),
    dict(
        type='OmniPack3DDetInputs',
        keys=[cam_type, 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'cam2lidar', 'lidar2cam', 'lidar2img', 'box_type_3d', 'token'],
        input_img_keys=[cam_type],)
]

test_pipeline = [
    dict(
        type='LoadOmni3DMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args,
        load_cam_type=cam_type,
        load_cam_names=['fisheye_camera_front', 'fisheye_camera_left',
                        'fisheye_camera_right', 'fisheye_camera_rear',
                        ]),
    dict(
        type='OmniPack3DDetInputs',
        keys=[cam_type, 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'cam2lidar', 'lidar2img', 'lidar2cam',
            'sample_idx', 'token', 'img_path', 'lidar_path', 
            'num_pts_feats', 'box_type_3d',],
        input_img_keys=[cam_type],)
]

import math
model = dict(
    type='OmniPETR',
    data_preprocessor=dict(
        type='OmniDet3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False,
        img_keys=[cam_type],),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=18,
        in_channels=3,
        base_channels=64,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 2, 4),
        out_indices=(2, 3),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet18')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512],
        out_channels=256,
        num_outs=2),
    pts_bbox_head=dict(
        type='OmniPETRHead',
        num_classes=6,
        code_size=8,  # x, y, z, w, l, h, sin, cos, we dont predict velocity
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        depth_start=0.5,
        position_range=[-60, -60, -10.0, 60, 60, 10.0],
        normedlinear=False,
        ocam_fov=cam_fov,
        ocam_path='data/CarlaCollection/calib_results.txt',
        feature_size=(25, 100),
        dbound=(0.5, 48.5, 0.5),
        azimuth_range=(-math.radians(cam_fov/2), math.radians(cam_fov/2)),
        elevation_range=(-math.pi/4, math.pi/4),
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-60, -60, -10.0, 60, 60, 10.0],
            pc_range=detect_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=6),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        input_key=dict(
            img_key=cam_type,
            lidar_key='points',),
        pts=dict(
            grid_size=[480, 480, 1],
            voxel_size=voxel_size,
            point_cloud_range=detect_range,
            out_size_factor=4,
            assigner=dict(
                type='PETRHungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(
                    type='IoUCost', weight=0.0
                ),  # Fake cost. Just to be compatible with DETR head.
                pc_range=detect_range,)))
)

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

# TODO
val_evaluator = dict(
    type='Omni3DMetric',
    ref_range=48,
)
test_evaluator = val_evaluator


learning_rate = 0.0002
max_epochs = 40
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
        eta_min_ratio=5e-4,
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
val_cfg = dict()
test_cfg = dict()


auto_scale_lr = dict(enable=True, base_batch_size=8)


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=10),)
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=10),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='NDS',            
        rule='greater',            
        min_delta=0.0002,             
        strict=True,                
        check_finite=True,          
        patience=3)
    )

custom_hooks = [
    dict(type='SaveDetectionHook', score_thr=0.03, class_names=classes),
]

find_unused_parameters = False

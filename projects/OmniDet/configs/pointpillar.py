_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.OmniDet.utils',
             'projects.OmniDet.dataset',
             'projects.OmniDet.models',
             'projects.OmniDet.models.omni_lss',],
    allow_failed_imports=False)
# custom_imports = dict(imports=['projects.example_project.dummy'])

dataset_type = 'Omni3DDataset'
data_root = 'data/CarlaCollection/'
# classes = ('Pedestrian', 'Cyclist', 'Car')
point_cloud_range=[-50, -50, -5, 50, 50, 3]

backend_args = None

train_pipeline = [
    dict(
        type='LoadOmni3DPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
        load_point_type='lidar'),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,),
    # dict(
    #     type='ObjectNoise',
    #     num_try=100,
    #     translation_std=[1.0, 1.0, 0.5],
    #     global_rot_range=[0.0, 0.0],
    #     rot_range=[-0.78539816, 0.78539816]),
    # dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.78539816, 0.78539816],
    #     scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='OmniPack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['box_type_3d', 'token'],
        input_point_keys=['points', 'lidar_points'],
        input_img_keys=[],)
]

test_pipeline = [
    dict(
        type='LoadOmni3DPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args,
        load_point_type='lidar'),
    dict(
        type='OmniPack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=['box_type_3d', 'token', 'sample_idx'],
        input_point_keys=['points', 'lidar_points'],
        input_img_keys=[],)
]

voxel_size = [0.25, 0.25, 8]
model = dict(
    type='MVXFasterRCNN',
    # extra_config=dict(
    #     img_key='cam_nusc',
    #     lidar_key='lidar_points',
    #     depth_supervision=False,),
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=64,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(30000, 40000))),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[400, 400]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        norm_cfg=dict(type='naiveSyncBN2d', eps=1e-3, momentum=0.01),
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=6,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [-48, -48, -0.6, 48, 48, -0.6],
                [-48, -48, -0.6, 48, 48, -0.6],
                [-48, -48, -0.6, 48, 48, -0.6],
                [-48, -48, -0.6, 48, 48, -0.6],
                [-48, -48, -0.6, 48, 48, -0.6],
                [-48, -48, -0.6, 48, 48, -0.6],
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
            loss_weight=0.2),),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
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
            code_weight=[1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0],
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
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
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets-2hz-all/omni3d_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        # metainfo=dict(classes=classes),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets-2hz-all/omni3d_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        # metainfo=dict(classes=classes),
    )
)
test_dataloader = val_dataloader

# TODO
val_evaluator = dict(
    type='Omni3DMetric',
)
test_evaluator = val_evaluator


# The schedule is usually used by models trained on KITTI dataset
# The learning rate set in the cyclic schedule is the initial learning rate
# rather than the max learning rate. Since the target_ratio is (10, 1e-4),
# the learning rate will change from 0.0018 to 0.018, than go to 0.0018*1e-4
lr = 0.0018
# The optimizer follows the setting in SECOND.Pytorch, but here we use
# the official AdamW optimizer implemented by PyTorch.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 16 epochs, learning rate increases from 0 to lr * 10
    # during the next 24 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=16,
        eta_min=lr * 10,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=24,
        eta_min=lr * 1e-4,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 16 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 24 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=16,
        eta_min=0.85 / 0.95,
        begin=0,
        end=16,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=24,
        eta_min=1,
        begin=16,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Runtime settings，training schedule for 40e
# Although the max_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual max epoch
# number could be Nx40
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (6 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=8)


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1),)

custom_hooks = [
    dict(type='SaveDetectionHook', score_thr=0.01, class_names=classes),
]

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

find_unused_parameters = False

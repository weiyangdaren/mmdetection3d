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
classes = ['Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
detect_range = [-48, -48, -5, 48, 48, 5]
cam_type='cam_nusc'
train_ann_file = 'ImageSets-2hz-0.7-all/omni3d_infos_train.pkl'
val_ann_file = 'ImageSets-2hz-0.7-all/omni3d_infos_val.pkl'


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


model = dict(
    type='OmniLSS',
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
    view_transform=dict(
        type='LSSTransform',
        in_channels=256,
        out_channels=96,
        image_size=[400, 800],
        feature_size=[25, 50],
        xbound=[-48.0, 48.0, 0.3],
        ybound=[-48.0, 48.0, 0.3],
        zbound=[-5.0, 5.0, 10.0],
        dbound=[0.5, 48.5, 0.5],
        downsample=1),
    pts_backbone=dict(
        type='SECOND',
        in_channels=96,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[2, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=6,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransFusionTransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-48.0, -48.0],
            post_center_range=[-50, -50, -5.0, 50, 50, 5.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=8),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)),
    train_cfg=dict(
        input_key=dict(
            img_key=cam_type,
            lidar_key='points',),
        pts=dict(
            dataset='Omni3D',
            point_cloud_range=detect_range,
            grid_size=[1280, 1280, 40],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            assigner=dict(
                type='LSSHungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),),
    test_cfg=dict(
        input_key=dict(
            img_key=cam_type,
            lidar_key='points',),
        pts=dict(
            dataset='Omni3D',
            grid_size=[1280, 1280, 40],
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            pc_range=[-48.0, -48.0],
            nms_type=None),),
)

# CBGSDataset
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
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
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets-2hz-0.7-all/omni3d_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes),
    )
)
test_dataloader = val_dataloader

# TODO implement evaluator by distance and weathers
val_evaluator = dict(
    type='Omni3DMetric',
)
test_evaluator = val_evaluator


learning_rate = 0.0001
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
val_cfg = dict()
test_cfg = dict()


auto_scale_lr = dict(enable=True, base_batch_size=8)


default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5),)

custom_hooks = [
    dict(type='OutputHook', save_dir='output'),
]

find_unused_parameters = False

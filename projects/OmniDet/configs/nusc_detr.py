_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(
    imports=['projects.OmniDet.utils',
             'projects.OmniDet.dataset',
             'projects.OmniDet.models',
             'projects.OmniDet.models.omni_detr3d',],
    allow_failed_imports=False)
# custom_imports = dict(imports=['projects.example_project.dummy'])

dataset_type = 'Omni3DDataset'
data_root = 'data/CarlaCollection/'
classes = ['Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist']
voxel_size = [0.2, 0.2, 10]
detect_range = [-48, -48, -5, 48, 48, 5]
cam_type='cam_nusc'

backend_args = None

ida_aug_conf = {
    'resize_lim': (0.625, 0.625),
    'final_dim': (400, 800),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 720,
    'W': 1280,
    'rand_flip': False,
    'img_key': cam_type,
}

train_pipeline = [
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
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=detect_range),
    dict(
        type='OmniPack3DDetInputs',
        keys=[cam_type, 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'img_shape', 'cam2lidar', 'lidar2cam', 
            'img_aug_matrix', 'box_type_3d'],
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
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=False),
    dict(
        type='OmniPack3DDetInputs',
        keys=[cam_type, 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'img_shape', 'cam2lidar', 'lidar2cam',
            'sample_idx', 'token', 'img_path', 'lidar_path', 
            'num_pts_feats', 'img_aug_matrix', 'box_type_3d',],
        input_img_keys=[cam_type],)
]


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], bgr_to_rgb=False)
model = dict(
    type='OmniDETR3D',
    use_grid_mask=True,
    data_preprocessor=dict(
        type='OmniDet3DDataPreprocessor',
        # mean=[123.675, 116.28, 103.53],
        # std=[58.395, 57.12, 57.375],
        # bgr_to_rgb=False,
        **img_norm_cfg,
        img_keys=[cam_type],),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True),
    pts_bbox_head=dict(
        type='DETR3DHead',
        num_query=900,
        num_classes=6,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=8,
        transformer=dict(
            type='Detr3DTransformer',
            decoder=dict(
                type='Detr3DTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='Detr3DTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',  # mmcv.
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='Detr3DCrossAtten',
                            pc_range=detect_range,
                            num_points=1,
                            embed_dims=256,
                            input_key=cam_type)
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-60, -60, -10.0, 60, 60, 10.0],
            pc_range=detect_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=6),
        positional_encoding=dict(
            type='mmdet.SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
    # model training and testing settings
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
                type='DetrHungarianAssigner3D',
                cls_cost=dict(type='mmdet.FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                # ↓ Fake cost. This is just to get compatible with DETR head
                iou_cost=dict(type='mmdet.IoUCost', weight=0.0),
                pc_range=detect_range))))


train_dataloader = dict(
    batch_size=4,
    num_workers=16,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='ImageSets-2hz-0.7-all/omni3d_infos_train.pkl',
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
        ann_file='ImageSets-2hz-0.7-all/omni3d_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True,
        metainfo=dict(classes=classes),
    )
)
test_dataloader = val_dataloader

# TODO
val_evaluator = dict(
    type='Omni3DMetric',
)
test_evaluator = val_evaluator


learning_rate = 0.00005
max_epochs = 10
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
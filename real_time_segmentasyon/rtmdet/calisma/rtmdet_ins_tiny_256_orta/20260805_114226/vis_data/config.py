ALBU = [
    dict(p=0.5, type='HorizontalFlip'),
    dict(
        p=0.7,
        rotate=(
            -15,
            15,
        ),
        scale=(
            0.85,
            1.15,
        ),
        translate_percent=(
            -0.1,
            0.1,
        ),
        type='Affine'),
    dict(alpha=30, p=0.25, sigma=6, type='ElasticTransform'),
    dict(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
        type='RandomBrightnessContrast'),
    dict(gamma_limit=(
        80,
        120,
    ), p=0.3, type='RandomGamma'),
    dict(
        elementwise=True,
        multiplier=(
            0.85,
            1.15,
        ),
        p=0.3,
        type='MultiplicativeNoise'),
    dict(blur_limit=7, p=0.2, type='MotionBlur'),
]
ANN = 'd:/mamografi/real_time_segmentasyon/veri/coco'
AUG_SEVIYE = 'orta'
BATCH = 8
GIRDI = 256
KOK = 'd:/mamografi/real_time_segmentasyon'
MAX_EPOCH = 100
albu_sarmalayici = dict(
    bbox_params=dict(
        filter_lost_elements=True,
        format='pascal_voc',
        label_fields=[
            'gt_bboxes_labels',
        ],
        min_visibility=0.0,
        type='BboxParams'),
    keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
    transforms=[
        dict(p=0.5, type='HorizontalFlip'),
        dict(
            p=0.7,
            rotate=(
                -15,
                15,
            ),
            scale=(
                0.85,
                1.15,
            ),
            translate_percent=(
                -0.1,
                0.1,
            ),
            type='Affine'),
        dict(alpha=30, p=0.25, sigma=6, type='ElasticTransform'),
        dict(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
            type='RandomBrightnessContrast'),
        dict(gamma_limit=(
            80,
            120,
        ), p=0.3, type='RandomGamma'),
        dict(
            elementwise=True,
            multiplier=(
                0.85,
                1.15,
            ),
            p=0.3,
            type='MultiplicativeNoise'),
        dict(blur_limit=7, p=0.2, type='MotionBlur'),
    ],
    type='AlbuBosGuvenli')
auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_lr = 0.000125
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'ozel_transformlar',
    ])
data_root = 'd:/mamografi/real_time_segmentasyon/Dataset_BUSI_with_GT/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        interval=1,
        max_keep_ckpts=1,
        rule='greater',
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=10, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
interval = 10
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
metainfo = dict(
    classes=('lezyon', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
model = dict(
    _scope_='mmdet',
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.167,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth',
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        type='CSPNeXt',
        widen_factor=0.375),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        feat_channels=96,
        in_channels=96,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_mask=dict(
            eps=5e-06, loss_weight=2.0, reduction='mean', type='DiceLoss'),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=1,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetInsSepBNHead'),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            96,
            192,
            384,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=1,
        out_channels=96,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        mask_thr_binary=0.5,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
optim_wrapper = dict(
    _scope_='mmdet',
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(lr=0.000125, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=200, start_factor=0.001, type='LinearLR'),
    dict(
        T_max=50,
        begin=50,
        by_epoch=True,
        convert_to_iter_based=True,
        end=100,
        eta_min=6.25e-06,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = False
stage2_num_epochs = 20
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='d:/mamografi/real_time_segmentasyon/veri/coco/busi_test.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='d:/mamografi/real_time_segmentasyon/Dataset_BUSI_with_GT/',
        metainfo=dict(classes=('lezyon', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(
                poly2mask=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file='d:/mamografi/real_time_segmentasyon/veri/coco/busi_test.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(
        poly2mask=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    _scope_='mmdet',
    dynamic_intervals=None,
    max_epochs=100,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=8,
    dataset=dict(
        ann_file=
        'd:/mamografi/real_time_segmentasyon/veri/coco/busi_train.json',
        data_prefix=dict(img=''),
        data_root='d:/mamografi/real_time_segmentasyon/Dataset_BUSI_with_GT/',
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        metainfo=dict(classes=('lezyon', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                poly2mask=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                bbox_params=dict(
                    filter_lost_elements=True,
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                    ],
                    min_visibility=0.0,
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
                transforms=[
                    dict(p=0.5, type='HorizontalFlip'),
                    dict(
                        p=0.7,
                        rotate=(
                            -15,
                            15,
                        ),
                        scale=(
                            0.85,
                            1.15,
                        ),
                        translate_percent=(
                            -0.1,
                            0.1,
                        ),
                        type='Affine'),
                    dict(alpha=30, p=0.25, sigma=6, type='ElasticTransform'),
                    dict(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5,
                        type='RandomBrightnessContrast'),
                    dict(gamma_limit=(
                        80,
                        120,
                    ), p=0.3, type='RandomGamma'),
                    dict(
                        elementwise=True,
                        multiplier=(
                            0.85,
                            1.15,
                        ),
                        p=0.3,
                        type='MultiplicativeNoise'),
                    dict(blur_limit=7, p=0.2, type='MotionBlur'),
                ],
                type='AlbuBosGuvenli'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=True, type='LoadAnnotations', with_bbox=True,
        with_mask=True),
    dict(
        bbox_params=dict(
            filter_lost_elements=True,
            format='pascal_voc',
            label_fields=[
                'gt_bboxes_labels',
            ],
            min_visibility=0.0,
            type='BboxParams'),
        keymap=dict(gt_bboxes='bboxes', gt_masks='masks', img='image'),
        transforms=[
            dict(p=0.5, type='HorizontalFlip'),
            dict(
                p=0.7,
                rotate=(
                    -15,
                    15,
                ),
                scale=(
                    0.85,
                    1.15,
                ),
                translate_percent=(
                    -0.1,
                    0.1,
                ),
                type='Affine'),
            dict(alpha=30, p=0.25, sigma=6, type='ElasticTransform'),
            dict(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
                type='RandomBrightnessContrast'),
            dict(gamma_limit=(
                80,
                120,
            ), p=0.3, type='RandomGamma'),
            dict(
                elementwise=True,
                multiplier=(
                    0.85,
                    1.15,
                ),
                p=0.3,
                type='MultiplicativeNoise'),
            dict(blur_limit=7, p=0.2, type='MotionBlur'),
        ],
        type='AlbuBosGuvenli'),
    dict(keep_ratio=False, scale=(
        256,
        256,
    ), type='Resize'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(
        _scope_='mmdet',
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(
        _scope_='mmdet',
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResize'),
    dict(
        _scope_='mmdet',
        allow_negative_crop=True,
        crop_size=(
            640,
            640,
        ),
        recompute_bbox=True,
        type='RandomCrop'),
    dict(_scope_='mmdet', min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(_scope_='mmdet', type='YOLOXHSVRandomAug'),
    dict(_scope_='mmdet', prob=0.5, type='RandomFlip'),
    dict(
        _scope_='mmdet',
        pad_val=dict(img=(
            114,
            114,
            114,
        )),
        size=(
            640,
            640,
        ),
        type='Pad'),
    dict(_scope_='mmdet', type='PackDetInputs'),
]
tta_model = dict(
    _scope_='mmdet',
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.6, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(_scope_='mmdet', backend_args=None, type='LoadImageFromFile'),
    dict(
        _scope_='mmdet',
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    640,
                    640,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    320,
                    320,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    960,
                    960,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(
                    pad_val=dict(img=(
                        114,
                        114,
                        114,
                    )),
                    size=(
                        960,
                        960,
                    ),
                    type='Pad'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='d:/mamografi/real_time_segmentasyon/veri/coco/busi_val.json',
        backend_args=None,
        data_prefix=dict(img=''),
        data_root='d:/mamografi/real_time_segmentasyon/Dataset_BUSI_with_GT/',
        metainfo=dict(classes=('lezyon', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                256,
                256,
            ), type='Resize'),
            dict(
                poly2mask=True,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file='d:/mamografi/real_time_segmentasyon/veri/coco/busi_val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='CocoMetric')
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'd:/mamografi/real_time_segmentasyon/rtmdet/calisma/rtmdet_ins_tiny_256_orta'

_base_ = [
    '../_base_/datasets/base_cd.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
data_root = 'data/whucd'
metainfo = dict(
                classes=('background', 'building'),
                palette=[[0, 0, 0], [255, 255, 255]])

crop_size = (256, 256)
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375])

backbone_norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='EncoderDecoderCDLoss',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        in_channels=3,
        patch_norm=True,
        pretrain_img_size=(512, 256),
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU')),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHeadCDLoss',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                     dict(type='CDLoss', use_sigmoid=True, loss_weight=0.5, class_weight=[0.5, 0.5])]),
    auxiliary_head=dict(
        type='FCNHead',
        num_classes=2,
        out_channels=1,
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00003, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False,
    )
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.txt'))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.txt'))
test_dataloader = val_dataloader

# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

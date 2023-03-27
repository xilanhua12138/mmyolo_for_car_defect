_base_ = './configs/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py'
data_root = '/root/mmyolo/autodl-tmp/defect_yolo' 
work_dir = '/root/mmyolo/autodl-tmp/work_dirs/yolov7_l_swin_t_anchor_free_car_defect'

train_batch_size_per_gpu = 16
train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4
val_batch_size_per_gpu = 8
val_num_workers = 4

max_epochs = 100

img_scale = (1280,1280)
save_epoch_intervals = 5

num_det_layers = 3
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
strides = [8, 16, 32]

loss_cls_weight = 0.5
loss_bbox_weight = 7.5
loss_dfl_weight = 1.5 / 4

tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics


class_name = ('paint_defect','shape_defect')  # 根据 class_with_id.txt 类别信息，设置 class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60),(110, 50, 60)]  # 画图时候的颜色，随便设置即可
)
# checkpoint_file = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
# checkpoint_file = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth' 
# checkpoint_file = './swin_tiny_patch4_window7_224.pth' 
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/van/van-base_8xb128_in1k_20220501-6a4cc31b.pth'  # noqa
model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

model = dict(
    backbone=dict(
        _delete_=True, # 将 _base_ 中关于 backbone 的字段删除
        type='mmcls.VAN', # 使用 mmcls 中的 MobileNetV3
        arch='b',
        out_indices=(1, 2, 3), # 修改 out_indices
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            )), # MMCls 中主干网络的预训练权重含义 prefix='backbone.'，为了正常加载权重，需要把这个 prefix 去掉。
    neck=dict(
        in_channels=[128, 320, 512],
        # The real output channel will be multiplied by 2
        out_channels=[128, 256, 512]),
    bbox_head=dict(
        _delete_=True,
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=1.0,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),
    train_cfg=dict(
        _delete_=True,
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9)),
    test_cfg=model_test_cfg
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        # 数据量太少的话，可以使用 RepeatDataset ，在每个 epoch 内重复当前数据集 n 次，这里设置 5 是重复 5 次
        times=5,
        dataset=dict(
            type=_base_.dataset_type,
            data_root=data_root,
            metainfo=metainfo,
            ann_file='annotation/trainval.json',
            data_prefix=dict(img='image/'),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=_base_.train_pipeline)))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotation/trainval.json',
        data_prefix=dict(img='image/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + '/annotation/trainval.json')
test_evaluator = val_evaluator

default_hooks = dict(
    logger=dict(interval=1),
    visualization=dict(draw=True, interval=1))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(270, 1)])

visualizer = dict(vis_backends = [dict(type='WandbVisBackend')])

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0015,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv7OptimWrapperConstructor')

# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.0001,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg=dict(
#         custom_keys={
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))
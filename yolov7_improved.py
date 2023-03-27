_base_ = './configs/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py'
data_root = './autodl-tmp/defect_yolo' 
work_dir = './autodl-tmp/work_dirs/yolov7_l_car_defect_improved'

train_batch_size_per_gpu = 16
train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4
val_batch_size_per_gpu = 8
val_num_workers = 4

loss_cls_weight = 0.3
loss_bbox_weight = 0.05
loss_obj_weight = 0.7

max_epochs = 200

img_scale = (640,640)
save_epoch_intervals = 5

num_det_layers = 4

# load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco/yolov7_l_syncbn_fast_8x16b-300e_coco_20221123_023601-8113c0eb.pth'
class_name = ('paint_defect','shape_defect')  # 根据 class_with_id.txt 类别信息，设置 class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60),(110, 50, 60)]  # 画图时候的颜色，随便设置即可
)
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
strides = [4, 8, 16, 32]
anchors = [[(12, 12), (21, 17), (41, 14)], [(34, 28), (25, 47), (58, 24)], [(50, 52), (117, 32), (106, 58)], [(69, 122), (310, 43), (175, 101)]]
model = dict(
    backbone=dict(
        type='YOLOv7Backbone',
        arch='L',
        out_indices = (1,2,3,4)
    ),
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(
            type='ELANBlock',
            middle_ratio=0.5,
            block_ratio=0.25,
            num_blocks=4,
            num_convs_in_block=1),
        upsample_feats_cat_first=False,
        in_channels=[256 ,512 ,1024, 1024],
        # The real output channel will be multiplied by 2
        out_channels=[64, 256, 256, 256],
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=2,
            in_channels=[64, 256, 256, 256],
            featmap_strides=strides,
            num_base_priors=3),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight *
            (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight *
            ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        obj_level_weights=[4.0, 1.0, 0.25, 0.06],
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides
            )),


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
    logger=dict(interval=10),
    visualization=dict(draw=True, interval=10))

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
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv7OptimWrapperConstructor')
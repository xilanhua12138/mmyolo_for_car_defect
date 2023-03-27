_base_ = './configs/yolov7/yolov7_l_syncbn_fast_8x16b-300e_coco.py'
data_root = '/root/mmyolo/autodl-tmp/defect_yolo' 
work_dir = '/root/mmyolo/autodl-tmp/work_dirs/yolov7_m_car_defect'

train_batch_size_per_gpu = 16
train_num_workers = 4  # 推荐使用 train_num_workers = nGPU x 4
val_batch_size_per_gpu = 8
val_num_workers = 4

max_epochs = 200

img_scale = (1280,1280)
save_epoch_intervals = 5

num_det_layers = 3

class_name = ('paint_defect','shape_defect')  # 根据 class_with_id.txt 类别信息，设置 class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(220, 20, 60),(110, 50, 60)]  # 画图时候的颜色，随便设置即可
)

anchors = [[[12, 11], [20, 18], [39, 14]], [[34, 29], [69, 23], [37, 66]], [[62, 45], [130, 46], [158, 120]]]
model = dict(
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=2,
            num_base_priors=3),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.3 * (num_classes / 80 * 3 / num_det_layers)),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors)),


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
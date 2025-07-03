_base_ = [
    'faster-rcnn_r101_fpn_2x_coco_10.py' #你没加马赛克增强的配置文件(同一目录下)
]
data_root = '/root/autodl-tmp/VISDRONE/'
dataset_type = 'CocoDataset'
img_scale = (2000, 2000)
img_norm_cfg = dict(
    mean=[0., 0., 0.], std=[255., 255., 255.], to_rgb=True)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2)), # 图像经过马赛克处理后会放大4倍，所以我们使用仿射变换来恢复图像的大小。
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

train_dataset = dict(
    _delete_ = True, # 删除不必要的设置
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2019-DET-train/my_train.json',
        img_prefix=data_root + 'VisDrone2019-DET-train/images/',
        classes=('pedestrian', 'people', 'bicycle',
                'car', 'van', 'truck', 'tricycle',
                'awning-tricycle', 'bus', 'motor'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline
    )

data = dict(
    train=train_dataset
    )

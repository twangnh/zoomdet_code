_base_ = './faster-rcnn_r50_fpn_2x_coco_40.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

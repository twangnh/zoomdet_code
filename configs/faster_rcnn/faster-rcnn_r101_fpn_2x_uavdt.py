_base_ = './faster-rcnn_r50_fpn_2x_uavdt.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

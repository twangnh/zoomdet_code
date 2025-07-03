_base_ = [
    '../_base_/models/retinanet_r50_fpn_argoverse.py',
    '../_base_/datasets/argoverse_detection.py',
    '../_base_/schedules/schedule_argoverse.py', '../_base_/default_runtime.py',
    './retinanet_tta.py'
]
load_from='/root/autodl-tmp/pretrained_coco_retinanet/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
# load_from='/root/autodl-tmp/pretrained_coco_retinanet/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

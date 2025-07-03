_base_ = './faster-rcnn_r50_fpn_2x_argoverse.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
load_from='/root/autodl-tmp/0410_DIOR_deformable_grid_w_mag_loss_1_1_1_clsloss2x_boxwise_loss_sml_x0d2_maglossv2_m6b3a2_mscoco_0-H-1_sml_baseline/epoch_12.pth'
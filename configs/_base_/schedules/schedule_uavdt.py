# training schedule for 2x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=6,
        by_epoch=True,
        milestones=[4, 5],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    # clip_grad=dict(max_norm=10.0, norm_type=2)
    paramwise_cfg=dict(
        custom_keys={'saliency_network': dict(lr_mult=0.1, decay_mult=1.0)})
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=8)

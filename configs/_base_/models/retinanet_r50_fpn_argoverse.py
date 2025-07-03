# model settings
# class_name = ('pedestrian', 'people', 'bicycle',
#               'car', 'van', 'truck', 'tricycle',
#               'awning-tricycle', 'bus', 'motor')
# num_classes = len(class_name)


model = dict(
    # type='RetinaNet',
    type='RetinaNetZoom',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # mean=[0, 0, 0],
        # std=[255., 255., 255.],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=8,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
        saliency_loss_weight=dict(
            weight=0.5,
            # small_threshold=0.693,
            # medium_threshold=0.693,
            # large_threshold=0.693,
            # small_threshold=0.598,
            # medium_threshold=0.644,
            # large_threshold=0.693,
            # small_threshold=0.693, #1.0
            # medium_threshold=0.644, #1.1
            # large_threshold=0.598, #1.2
            # small_threshold=0.644,  #
            # medium_threshold=0.598,  #
            # large_threshold=0.554,  # 1.3
            small_threshold=0.598,  #
            medium_threshold=0.554,  #
            large_threshold=0.513,  # 1.4
            # small_threshold = 0.644,
            # medium_threshold = 0.644,
            # large_threshold = 0.644,
            # small_threshold=0.598,
            # medium_threshold=0.598,
            # large_threshold=0.598

        ),
        fa_loss_weight=dict(
            weight=0.1
        ),
        gaussian_probability=dict(
            p=0
        ),
        detach_detector_loss=dict(
            switch=1
        ),
        sep_fwhm=dict(
            fwhm=350
        ),
        nonsep_fwhm=dict(
            fwhm=10
        )
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100,
        multiple_sum = dict(
            sum=0,
            small_sum=0,
            medium_sum=0,
            large_sum=0,
            max_multiple=0
        ),
        box_num_sum = dict(
            sum=0,
            small_sum=0,
            medium_sum=0,
            large_sum=0,
            multiple_more_than_2=0,
            image_sum=0
        ),
        iou_sum = dict(
            sum=0,
            small_sum=0,
            medium_sum=0,
            large_sum=0,
        ),
        iou_box_num_sum = dict(
            image_sum=0,
            sum=0,
            small_sum=0,
            medium_sum=0,
            large_sum=0,
        ),
        valdata_total = 0,
        ap_excel = dict(
            img_id=[],
            ap=[]
        )
))

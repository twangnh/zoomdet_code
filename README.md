

<div align="center">   

# Adaptive Image Zoom-in with Bounding Box Transformation for Aerial Object Detection
</div>

This codebase implements the paper *"Adaptive Image Zoom-in with Bounding Box Transformation for Aerial Object Detection"* submitted 
to **ISPRS Journal of Photogrammetry and Remote Sensing**

Faster R-CNN is used in this repository, for the YOLO-based model reported in the paper, please refer to
[zoomdet_yolo](https://github.com/twangnh/zoomdet_yolo) repository.


## Installation

Please refer to [mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions. 


## Data Preparation
Download the datasets from official released sources:
[SeaDroneSee](https://seadronessee.cs.uni-tuebingen.de/dataset)
[VisDrone](https://aiskyeye.com/home/)
[UAVDT](https://datasetninja.com/uavdt)
,and convert the annotations to COCO json format.

prepare the data folder as:

```
data
    VisDrone
        train
            images
            instances_train.json
        val
            images
            instances_val.json
    UAVDT
        ...
    SeaDroneSee
        ...
```

## Usage

### Train with VisDrone

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2201 tools/train.py ./configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py --work-dir work_dir --num_gpu 4
```
> replace the work_dir with your customized one
### Test with VisDrone

```

python -m torch.distributed.launch --nproc_per_node=4 tools/test.py ./configs/faster_rcnn/faster-rcnn_r101_fpn_2x_coco.py /root/autodl-tmp/mmdetection_deformable_grid/mmdetection/visdrone_zoomdet/epoch_20.pth --num_gpu 4
```
> replace the checkpoint with the corresponding trained one on the dataset
### Train with UAVDT

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2201 tools/train.py ./configs/faster_rcnn/faster-rcnn_r101_fpn_2x_uavdt.py --work-dir work_dir --num_gpu 4
```
> replace the work_dir with your customized one


### Test with UAVDT

```

python -m torch.distributed.launch --nproc_per_node=4 tools/test.py ./configs/faster_rcnn/faster-rcnn_r101_fpn_2x_uavdt.py /root/autodl-tmp/mmdetection_deformable_grid/mmdetection/visdrone_zoomdet/epoch_20.pth --num_gpu 4
```
> replace the checkpoint with the corresponding trained one on the dataset

### Train with SeaDroneSee

```
python -m torch.distributed.launch --nproc_per_node=4 --master_port=2201 tools/train.py ./configs/faster_rcnn/faster-rcnn_r101_fpn_2x_seadronesee.py --work-dir work_dir --num_gpu 4
```
> replace the work_dir with your customized one
### Test with SeaDroneSee

```

python -m torch.distributed.launch --nproc_per_node=4 tools/test.py ./configs/faster_rcnn/faster-rcnn_r101_fpn_2x_seadronesee.py /root/autodl-tmp/mmdetection_deformable_grid/mmdetection/visdrone_zoomdet/epoch_20.pth --num_gpu 4
```
> replace the checkpoint with the corresponding trained one on the dataset



## Acknowledgement

Zoomdet is an open source project, and is based on [mmdetection](https://github.com/open-mmlab/mmdetection)

## License

This project is released under the [Apache 2.0 license](LICENSE).

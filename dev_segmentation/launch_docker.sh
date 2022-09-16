#!/bin/bash

# Running retraining!!!
devdir=/mnt/NewHDD/jie/Workarea/analysis/detectron2/dev_segmentation
pretrained=/mnt/NewHDD/jie/Workarea/analysis/detectron2/dev/pretrained
docker run --gpus 1 --rm \
	-v ${devdir}:/home/appuser/detectron2_repo/dev_segmentation \
	-v ${pretrained}:/home/appuser/detectron2_repo/dev_segmentation/pretrained \
	-it \
	detectron2:v0

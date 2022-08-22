#!/bin/bash

# Running retraining!!!
devdir=/mnt/NewHDD/jie/Workarea/analysis/detectron2/dev_detection
pretrained=/mnt/NewHDD/jie/Workarea/analysis/detectron2/dev/pretrained
docker run --gpus 1 --rm \
	-v ${devdir}:/home/appuser/detectron2_repo/dev_detect \
	-v ${pretrained}:/home/appuser/detectron2_repo/dev_detect/pretrained \
	-it \
	detectron2:v0

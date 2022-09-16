#!/bin/bash

cocodataset=${HOME}/3dselfie/3DSelfie/data/coco-dataset-2014
cocosubdata=${HOME}/3dselfie/3DSelfie/data/coco-2014-person-sportsBall
devdir=/mnt/NewHDD/jie/Workarea/analysis/detectron2/dev_retrain
# configfile="../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# weights="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
#
# use -d to run docker in background
#
docker run --gpus 1 --rm \
	-d \
	-v ${devdir}:/home/appuser/detectron2_repo/dev_retrain \
	-v ${demodir}:/home/appuser/detectron2_repo/demo \
	-v ${imagedir}:/data \
	-v ${outdir}:/outputs \
	-v ${cocodataset}:/data/coco \
	-v ${cocosubdata}:/data/coco_person_sportsball \
	detectron2:v0 \
	/bin/sh -c \
	"""
	cd dev_retrain;
	echo step 1 > output/log.txt;
	source run_retrain_person_sportsball.sh; 
	echo step 2 >> output/log.txt;
	cp output/model_final.pth output/model_final_segmentation_person_sportsball_X101_more-person-dataset.pth
	echo step 3 >> output/log.txt;
	"""
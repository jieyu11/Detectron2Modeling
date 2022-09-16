#!/bin/bash

yamlfile="../configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
# weightfile="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"
weightfile="detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"
python retrain_person_sportsball.py \
        --traindir /data/coco_person_sportsball/train/images \
        --trainjson coco-manager/person_sportsball_train2014.json \
		--yamlfile $yamlfile \
		--weightfile $weightfile \
		--retrain-even-exists


# balanced is smaller
#        --trainjson coco-manager/person_sportsball_train2014_balanced.json \
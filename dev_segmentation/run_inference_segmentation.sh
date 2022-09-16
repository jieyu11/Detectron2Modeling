#!/bin/bash
python inference_segmentation.py \
	-v videos/shoot_short.mp4 \
	-o output/shoot_short \
 	--classes "person" "sports ball" \
	-w retrained/model_final_segmentation.pth 
# -n 10 
# # images
# python inference_segmentation.py \
# 	-i images/coco_mini/0ff86d84bed3933e.jpg \
# 	-o output/coco_mini \
#  	--classes "person", "sports ball" \
# 	-w retrained/model_final_segmentation.pth
#!/bin/bash

python inference_detection.py \
	-v videos/basketball_recording.mov \
	-o output/inferences/demo_basketball \
 	--classes "person", "sports ball" \
	-n 100000 \
	--confidence-threshold 0.5 \
	-w retrained/model_final_human_ball.pth

# python retrain_detection.py \
# 	-v videos/demo_prod_002.mov \
# 	-o output/inferences/demo_003_model_comb_v01 \
# 	--classes 'boxed product' "container" \
# 	-tj coco/demo_003.json 

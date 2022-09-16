#!/bin/bash

# https://github.com/immersive-limit/coco-manager

# python filter.py --input_json /data/coco/annotations/instances_val2014.json \
# 	--output_json person_sportsball_val2014.json \
# 	--categories "person" "sports ball"

python filter.py --input_json /data/coco/annotations/instances_train2014.json \
	--output_json person_sportsball_train2014.json \
	--categories "person" "sports ball"
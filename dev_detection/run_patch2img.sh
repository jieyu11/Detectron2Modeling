#!/bin/bash


# python3 patch_to_image.py \
# 	-p images/raw/products/*.png \
# 	-b images/raw/backgrounds/*.png \
# 	-o images/synthetic_demo_003/products \
# 	-c coco/demo_003_prod.json \
# 	-n 500 \
# 	-cid 1 \
# 	--name "boxed product" \
# 	-imgidx 0
# 

python3 patch_to_image.py \
	-p images/raw/containers/*.png \
	-b images/raw/backgrounds/*.png \
	-o images/synthetic_demo_003/containers \
	-c coco/demo_003_contain.json \
	-n 250 \
	-cid 2 \
	--name "container" \
	-imgidx 6000


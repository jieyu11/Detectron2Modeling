#!/bin/bash
python tracking.py \
 	-v videos/demo_prod_002.mov \
 	-o output/tracking/demo_004_model_comb_v01 \
 	--classes "product" "container" \
	-n 100000 \
	--confidence-threshold 0.5 \
	-w retrained/model_final_box_prod_data-5-6.pth
	
 	# --classes "boxed product" "container" \
	# retrained/model_final_box_prod.pth 

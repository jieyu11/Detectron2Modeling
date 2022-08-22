#!/bin/bash

ffmpeg -framerate 30 -pattern_type glob \
	-i "output/inferences/demo_003_model_comb_v01/*.png" \
	-c:v libx264 -pix_fmt yuv420p -vf \
	pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" \
	output/inference_demo_003_v1.mp4


# ffmpeg -framerate 30 -pattern_type glob \
# 	-i "output/inferences/demo_002_model_v2/*.png" \
# 	-c:v libx264 -pix_fmt yuv420p -vf \
# 	pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" out.mp4
# 

# inputfolder=output/inferences/demo_002_model_v2
# # outputname=output/inferences/demo_002_model_v2/demo_prod_002.mp4
# outputname=demo_prod_002.mp4
# framerate=60
# ffmpeg -framerate $framerate \
# 	-i ${inputfolder}/*.png \
# 	-c:v libx264 -pix_fmt yuv420p \
# 	-vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" \
# 	$outputname
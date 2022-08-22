#!/bin/bash

input=${HOME}/3dselfie/3DSelfie/data/clip_videos/production_line/demo_prod_002.mov
mkdir -p images/demo_prod_002
ffmpeg -i ${input} -qscale:v 1 -vf fps=10 images/demo_prod_002/%06d.png

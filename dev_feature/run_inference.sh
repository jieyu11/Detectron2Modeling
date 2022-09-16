#!/bin/bash

python main.py \
	-v videos/shoot_short.mp4 \
	--classes "person", "sports ball" \
	-y "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml" \
	-w retrained/model_segmentation_human_ball.pth


# -i images/celeb/leonardo_dicaprio/0006.jpg images/celeb/leonardo_dicaprio/0007.jpg \
# -i images/celeb/leonardo_dicaprio/0006.jpg images/celeb/leonardo_dicaprio/0007.jpg \

# 0006 - 0007: 0.6023
# 0006 - 0008: 0.4285
# 0006 - 0009: 0.1617
# 0006 - 0010: 0.5463
# 0007 - 0008: 0.7810
# 0007 - 0009: 0.4588
# 0007 - 0010: 0.4304
# 0008 - 0009: 0.7741
# 0008 - 0010: 0.3994
# 0009 - 0010: 0.2212

# size 6
# 0006 - 0007: 0.6134

# size 5
# 0006 - 0007: 0.6841
# 0006 - 0009: 0.1861
# 0009 - 0010: 0.2448

# size 3
# 0006 - 0007: 0.6639
# 0006 - 0009: 0.1294
#	-w retrained/model_final_human_ball.pth

# -i images/celeb/brad_pitt/0003.jpg images/celeb/brad_pitt/0002.jpg \
# -i images/celeb/brad_pitt/0001.jpg images/celeb/emma_watson/0005.jpg \
# -i images/celeb/leonardo_dicaprio/0001.jpg images/celeb/leonardo_dicaprio/0001b.jpg \
# 0006 - 0006b: 0.9846
# 0006 - 0007: 0.6306
# 0006 - 0008: 0.3101
# 0006 - 0009: 0.0540
# 0006 - 0010: 0.4937
# 0007 - 0008: 0.7721
# 0007 - 0009: 0.4132
# 0007 - 0010: 0.3787
# 0008 - 0009: 0.6878
# 0008 - 0010: 0.2433
# 0009 - 0010: 0.1319





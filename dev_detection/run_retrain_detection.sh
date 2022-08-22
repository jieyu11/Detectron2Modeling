#!/bin/bash

# python retrain_detection.py \
# 		--classes 'boxed production' \
#         --traindir images/synthetic_demo_002 \
#         --trainjson coco/demo_002.json \
# 		--retrain-even-exists

python retrain_detection.py \
		--classes 'boxed product' "container" \
        --traindir images/synthetic_demo_003/combine_linked \
        --trainjson coco/demo_003.json \
		--retrain-even-exists \
		-o retrained
# https://github.com/facebookresearch/detectron2/issues/1840
import torch
import numpy as np
import cv2

from detectron2.modeling import build_backbone
from detectron2.modeling import build_model
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import ImageList
import detectron2.data.transforms as T

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
aug = T.ResizeShortestEdge(
    [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
)
cfg.MODEL.WEIGHTS = "model_final_68b088.pkl"
device = torch.device("cuda")
backbone = build_backbone(cfg).to(device)
# Note the second parameter below (1.0 / 8 , ...) must match the size factor from the initial
#   image to the corresponding layer.
#   For example, initial image is size: (h=1080, w=1920) and
#     the feature['p3'] has size [1, 256, h=136, w=240], then the factor is 1080 / 136 = 8, 1920 / 240 = 8
#     the feature['p4'] has size [1, 256, 68, 120]
#     the feature['p5'] has size [1, 256, 34, 60]
#     the feature['p6'] has size [1, 256, 17, 30]

pooler = ROIPooler(7, (1.0 / 8, 1.0 / 16, 1.0 / 32, 1.0 / 64), 0, "ROIAlignV2").to(device)
images = []
boxess = []
path = "images/shoot_short_example.png"
image = cv2.imread(path)
print("image size", image.shape)

image = (image - cfg.MODEL.PIXEL_MEAN) / cfg.MODEL.PIXEL_STD
image = torch.from_numpy(image).float().permute(2, 0, 1)
images.append(image)
# note 32 below is found by following method:
# 	model = build_model(cfg)
# 	print(model.backbone.size_divisibility)
# otherwise using image_list = ImageList.from_tensors(images) --> error:
# RuntimeError: The size of tensor a (43) must match the size of tensor b (44)
# at non-singleton dimension 2
image_list = ImageList.from_tensors(images, 32)
final_image = image_list.tensor.to(device)

# bounding box is defined as [x1, y1, x2, y2] corresponding to the original
# image
boxes = np.array([972.8062,  297.1974, 1264.2311,  888.5700])
boxes = torch.from_numpy(boxes).float().unsqueeze(0)
boxes = Boxes(boxes).to(device)
boxess.append(boxes)
feature = backbone(final_image)

for key in feature:
	print("feature key", key, type(feature[key]), feature[key].size())

features = []
features.append(feature['p3'])
features.append(feature['p4'])
features.append(feature['p5'])
features.append(feature['p6'])
# roi feature map
rois = pooler(features, boxess)
print("rois", rois.size(), torch.nonzero(rois).size())
print("roi max value", torch.max(rois))

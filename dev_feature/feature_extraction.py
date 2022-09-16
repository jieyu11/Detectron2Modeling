import os
import cv2
import torch
import logging
import argparse
import numpy as np
from time import time
from datetime import timedelta
from video_loader import VideoLoader
from typing import List, Set, Tuple, Dict
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures.image_list import ImageList
from detectron2.structures import Boxes
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
	"""Feature extraction class. Using ROIPooling to extract features for
		objects in the detected bounding boxes within framework of detectron2.
	"""
	feature_scales = {
		"p2": 1.0 /  4,
		"p3": 1.0 /  8, 
		"p4": 1.0 / 16, 
		"p5": 1.0 / 32, 
		"p6": 1.0 / 64, 
	}

	def __init__(self, parameters = {}):
		"""Initialize the model .

		Args:
			parameters (dict, optional): include the parameters
				- yamlfile: e.g. COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml
				- weightfile: e.g. model_final.pth
				- thing_classes: the multiple classes for detection
				- confidence_threshold: only keep objects above the threshold
				- nframes: run the number of frames from video.
		"""
		weightfile = "pretrained/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
		weightfile = parameters["weightfile"] if "weightfile" in parameters else weightfile
		yamlfile = "../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
		yamlfile = parameters["yamlfile"] if "yamlfile" in parameters else yamlfile
		self.cfg = get_cfg()
		self.cfg.merge_from_file(yamlfile)
		threshold = parameters.get("confidence_threshold", 0.7)
		self.thing_classes = parameters["thing_classes"]
		self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
		self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.thing_classes)
		self.cfg.MODEL.WEIGHTS = weightfile
		self.cfg.freeze()

		# pooler_types: ROIPool, ROIAlignV2, ROIAlign
		# self.pooler = ROIPooler(7, , 0, "ROIPool").to("cuda")
		# if using features: ["p3", "p4", "p5", "p6"], then
		# scales = (1.0 / 8, 1.0 / 16, 1.0 / 32, 1.0 / 64)
		# if using features: ["p2", "p3", "p4", "p5"], then
		# scales = (1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32)
		pooler_type = parameters.get("pooler_type", "ROIAlignV2")
		# the length of the names should match the pooler, names should match the model
		self.feature_names = parameters.get("feature_names", ["p3", "p4", "p5", "p6"])
		scales = tuple([self.feature_scales[name] for name in self.feature_names])
		feature_output_size = parameters.get("feature_output_size", 7)
		self.pooler = ROIPooler(feature_output_size, scales, 0, pooler_type).to("cuda")

		self.model = build_model(self.cfg)
		DetectionCheckpointer(self.model).load(weightfile)
		self.model.eval()
		self.predictor = DefaultPredictor(self.cfg)

	def roi_full(self, image: np.ndarray):
		"""Given image, generate full information about each ROI, which includes the 
		bounding boxes, the confidence scores, the class / labels, the masks and the 
		feature vectors.

		Args:
			image (np.ndarray): input image

		Returns:
			dict: roi full output. including keys:
				- boxes: the list of detected bounding boxes
				- scores: the list of confidence scores of each bounding box
				- classes: the list of the classes each bounding boxes is 
					associated with.
				- masks: the list of the np.array as masks of the detected object.
				- features: the list of the feature vectors
		"""

		detected = self._get_detection(image)
		features = self._generate_features(image, detected)
		detected["features"] = features
		for key, itm in detected.items():
			logger.debug("detected %s type %s" % (key, str(type(itm))))
		return detected

	def _generate_features(self, image: np.ndarray, detected: dict):
		""" Given image and detected bounding boxes, generate the list of
		ROI for features with / without mask information.

		Args:
			image (np.ndarray): initial image.
			detected (dict): with keys of 'boxes', 'classes', 'masks'

		Returns:
			list[torch.tensor]: for each bounding box, a feature vector represented
				with a torch.tensor is calculated.
		"""
		boxes = detected["boxes"]
		image = image.copy()
		image = (image - self.cfg.MODEL.PIXEL_MEAN) / self.cfg.MODEL.PIXEL_STD
		labels = detected["classes"]
		masks = detected["masks"]
		if masks is None:
			roi_list = self._feature_no_mask(image, boxes, labels)
		else:
			masks = masks.cpu().numpy()
			roi_list = self._feature_with_mask(image, boxes, labels, masks)
		return roi_list
		
	def _feature_with_mask(self, image, boxes, labels: torch.Tensor, masks: np.ndarray):
		"""Generating ROI features with the masks from segmentation.

		Args:
			image (np.ndarray): initial image
			boxes (Boxes): the bounding box to extract roi features.
			labels (torch.Tensor): labels of the corresponding boxes.
			masks: (np.ndarray): mask array of the bounding boxes

		Returns:
			list[torch.Tensor]: ROI feature for each bounding box, which
				form a list of feature vectors.
		"""
		roi_list = []
		raw_image = image.copy()
		for lb, box, mask in zip(labels, boxes, masks):
			lb = lb.item()
			# keep only person
			# if lb != 0: continue

			# start from the raw image and apply corresponding mask
			#   of the corresponding bounding box!
			image = raw_image.copy()
			image[mask==False] = 0.0
			features = self._get_features(image)
			features = [features[name] for name in self.feature_names]
			# convert box from tensor([a,b,c,d]) to tensor([[a,b,c,d]])
			box_t = box.view(1, -1)
			box = Boxes(box_t)
			roi = self.pooler(features, [box])
			roi = torch.flatten(roi)
			roi_list.append(roi)
		return roi_list
	
	def _feature_no_mask(self, image, boxes, labels: torch.Tensor):
		"""Generating ROI features without using the masks from segmentation.

		Args:
			image (np.ndarray): initial image
			boxes (Boxes): the bounding box to extract roi features.
			labels (torch.Tensor): labels of the corresponding boxes.

		Returns:
			list[torch.Tensor]: ROI feature for each bounding box, which
				form a list of feature vectors.
		"""
		boxes = [boxes]
		features = self._get_features(image)
		features = [features[name] for name in self.feature_names]
		# the size of rois torch.Size([n-boxes, 256, 7, 7])
		rois = self.pooler(features, boxes)
		roi_list = []
		for idx, roi in enumerate(rois):
			# split roi per bounding box
			label = labels[idx].item()
			# if label != 0: continue
			roi = torch.flatten(roi)
			roi_list.append(roi)
		return roi_list

	def _get_features(self, image: np.ndarray):
		"""Run object detection inference model backbone on the given image.

		Args:
			image (np.ndarray): input image for feature extraction.

		Returns:
			dict: the extracted features in key: torch.Tensor format
		"""
		image = np.transpose(image, (2, 0, 1))
		image = torch.tensor(image, device='cuda', dtype=torch.float)
		images = [image]
		images = ImageList.from_tensors(images, 32)
		features = self.model.backbone(images.tensor)
		for key in features:
			logger.debug("Features %s in %s " % (key, str(features[key].size())))
		return features

	def _get_detection(self, im):
		"""Make inference on an image

		Args:
			im (np.array): input image array.

		Returns:
			dict: including keys and values of
				- boxes: the list of detected bounding boxes
				- scores: the list of confidence scores of each bounding box
				- classes: the list of the classes each bounding boxes is 
					associated with.
				- masks: the list of the np.array as masks of the detected object.
		"""
		outputs = self.predictor(im)
		# use predictions.to("cpu") to detach from GPU graph
		predictions = outputs["instances"] 
		# convert Boxes to torch.Tensor with predictions.pred_boxes.tensor
		boxes = predictions.pred_boxes
		scores = predictions.scores
		classes = predictions.pred_classes
		try:
			masks = predictions.pred_masks
		except Exception:
			masks = None
		result = {"boxes": boxes, "scores": scores, "classes": classes, "masks": masks}
		return result

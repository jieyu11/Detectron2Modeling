from detectron2.structures import Boxes
import os
import logging
from time import time
from datetime import timedelta
import argparse
import random
import cv2
import numpy as np
import torch

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectedObject:
	"""
	Detected object class.
	"""
	def __init__(self,
			object_id: int,
			label: int,
			bounding_box: Boxes,
			feature_vector: torch.Tensor,
			name: str, 
			detect_score: float,
			track_score: float = 0.0):
		"""Initialize an object.
		"""
		self.id = object_id
		self.label = label
		self.bounding_box = bounding_box
		self.feature_vector = feature_vector
		self.name = name
		self.detect_score = detect_score
		self.track_score = track_score
		self.track_similarity = track_score
		self.track_iou = track_score

	def update(self,
			bounding_box: Boxes,
			feature_vector: torch.Tensor,
			detect_score: float,
			track_score: float,
			track_similarity: float,
			track_iou: float
		):
		self.bounding_box = bounding_box
		self.feature_vector = feature_vector
		self.detect_score = detect_score
		self.track_score = track_score
		self.track_similarity = track_similarity
		self.track_iou = track_iou

def create_objects(detected: dict):
	"""Using detected information to generate objects.
	The detected dict is obtained by calling: FeatureExtractor.roi_full(image)

	Args:
		detected (dict): dict containing keys, "boxes", "classes",
			"scores", "features", "names"
	"""
	try:
		N = len(detected["features"])
		objects = []
		for idx in range(N):
			obj = DetectedObject(idx, detected["classes"][idx],
					detected["boxes"][idx],
					detected["features"][idx],
					detected["names"][idx],
					detected["scores"][idx]
					)
			objects.append(obj)
		return objects
	except:
		raise Exception("Input is incorrect.")
	
def record_objects(objects):
	result = []
	for i, obj in enumerate(objects):
		r = {
			"object_id": obj.id,
			"object_label": obj.label.item(),
			"object_name": obj.name,
			"bounding_box": torch.flatten(obj.bounding_box.tensor).cpu().numpy().tolist(),
			"detect_score": obj.detect_score.item(),
			"track_score": obj.track_score,
			"track_similarity": obj.track_similarity,
			"track_iou": obj.track_iou,
		}
		result.append(r)
	return result

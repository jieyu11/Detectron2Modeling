import os
import logging
from time import time
from datetime import timedelta
import argparse
import random
import cv2
import numpy as np
import torch
from object import DetectedObject
from typing import List, Set, Tuple, Dict
from detectron2.structures import Boxes, pairwise_iou

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Tracker:
    """Tracker class. Make object tracking based on the ROIPooling features and bounding
            boxes from objects detection.
    """

    def __init__(self):
        """Initialize the instance and setup the CosineSimilarity function.
        """
        self.cosinesimilarity = torch.nn.CosineSimilarity(dim=0)

    def tracking(self, objects: List[DetectedObject],
                 detected: dict,
                 frame_id: int):
        """ For each object detected in the current frame (including feature vectors,
                bounding boxes and labels), find the best match from a list of existing objects.

        Args:
                objects (list[DetectedObject]): list of existing objects.
                detected (dict) with keys and types:
                        features (list[torch.tensor]): feature vectors in the current frame.
                        boxes (list[torch.tensor]): bounding boxes in the current frame.
                        labels (list[int]): corresponding labels/classes in the current frame.
                        scores: List[float]: corresponding detected confidence scores.
                frame_id (int): frame index.

        Returns:
                list[DetectedObject], list[float], list[int]:
                        updated objects, corresponding combined scores for matching,
                        matched index of objects
        """
        for idx, obj in enumerate(objects):
            logger.debug("frame %d, box %s" % (idx, str(obj.bounding_box)))
            logger.debug(" - feature %s" % str(obj.feature_vector[0:10]))
            logger.debug(" - id %d label %s name %s" % (obj.id, str(obj.label), obj.name))
        try:
            matched, new_objects = [], []
            new_objects_idx = len(objects)
            N = len(detected["classes"])
            for idx in range(N):
                feature = detected["features"][idx]
                box = detected["boxes"][idx]
                label = detected["classes"][idx]
                detect_score = detected["scores"][idx]

                result = self.find_match(objects, feature, box, label, matched)
                matched_idx = result["matched_idx"]

                logger.debug("current object id %d box %s" % (idx, str(box)))
                logger.debug(" -   feature %s size %s" % (str(feature[0:10]), str(feature.size())))

                if matched_idx < 0:
                    logger.info("Frame: %d, object not matched, keep as new object." % frame_id)
                    new_objects.append(
                        DetectedObject(new_objects_idx,
                                       label, box, feature, "new object", detect_score)
                    )
                    matched.append(new_objects_idx)
                    new_objects_idx += 1
                else:
                    logger.debug("new object %d matched to %d with" % (idx, matched_idx))
                    logger.debug("    similarity %.3f iou %.3f" %
                                 (result["matched_similarity"], result["matched_iou"]))
                    objects[matched_idx].update(box, feature, detect_score, result["maxscore"],
                                                result["matched_similarity"], result["matched_iou"])
                    matched.append(matched_idx)

            if len(new_objects) > 0:
                objects.extend(new_objects)

            return objects, matched
        except:
            raise Exception("Input is incorrect.")

    def find_match(self, objects: List[DetectedObject],
                   feature: torch.Tensor,
                   box: Boxes,
                   label: int,
                   matched: List[int]):
        """ Find the matched index of the objects in the objects list based
        on current object's feature, box and label.

        Args:
                objects (List[DetectedObject]): list of existing objects, whose index to be matched.
                feature (torch.Tensor): feature vector of the new object
                box (torch.Tensor): bounding box of the new object
                label (int): label of the new object

        Returns:
                float, int: maximum score of the matched object, index of the matched object
        """
        maxscore, matched_sim, matched_iou, matched_idx = 0.0, 0.0, 0.0, -1
        for idx, obj in enumerate(objects):
            if idx in matched:
                continue
            if label != obj.label:
                continue

            # similarity score
            sim = self.cosinesimilarity(obj.feature_vector, feature)
            sim = float(sim.cpu().detach().numpy())
            # iou compared to the last box of that object
            # iou = pairwise_iou(Boxes(obj.bounding_box.view(1, -1)), Boxes(box.view(1, -1)))
            iou = pairwise_iou(obj.bounding_box, box)
            iou = float(iou.cpu().detach().numpy())
            comb = (sim + iou) / 2.0
            if comb > maxscore:
                maxscore = comb
                matched_sim = sim
                matched_iou = iou
                matched_idx = idx
        result = {
            "maxscore": maxscore,
            "matched_similarity": matched_sim,
            "matched_iou": matched_iou,
            "matched_idx": matched_idx
        }
        return result

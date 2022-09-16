import os
import cv2
import json
import torch
import logging
import argparse
import numpy as np
from time import time
from datetime import timedelta
from video_loader import VideoLoader
from feature_extraction import FeatureExtractor
from visualizer import Visualizer
from tracking import Tracker
from object import DetectedObject, create_objects, record_objects

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weight-file", "-w", type=str, default=None, required=False,
        help="Retrained weight file"
    )
    parser.add_argument(
        "--yaml-file", "-y", type=str, default=None, required=False,
        help="Yaml file"
    )
    parser.add_argument(
        "--images", "-i", type=str, required=False, nargs="+", default=None,
        help="Retrained weight file"
    )
    parser.add_argument("--video", "-v", default=None,
                        type=str, required=False,
                        help="video for inference.",
                        )
    parser.add_argument(
        "--nframes", "-n", default=1e6, type=int, required=False,
        help="Your number of frames to be read from input video.",
    )
    parser.add_argument("--classes", "-c", default=["person", "sports ball"], nargs="+",
                        type=str, required=True,
                        help="names of the classes / categories.",
                        )
    parser.add_argument("--outdir", "-o", default="output/viz",
                        type=str, required=False,
                        help="Your output folder name.",
                        )

    args = parser.parse_args()
    config = {
        "thing_classes": args.classes,
    }
    if args.weight_file:
        config["weightfile"] = args.weight_file
    if args.yaml_file:
        config["yamlfile"] = args.yaml_file

    vz = Visualizer(args.classes, args.outdir)
    rt = FeatureExtractor(config)
    tracker = Tracker()
    if args.video:
        basename = os.path.basename(args.video)
        basename = os.path.splitext(basename)[0]

        vl = VideoLoader(args.video, args.nframes)
        detected = rt.roi_full(vl.frames[0])
        # pre defined names
        detected["names"] = ["igor", "bob", "basketball"]
        objects = create_objects(detected)
        vz.draw(vl.frames[0], detected, "%s_%04d" % (basename, 0))

        # out put json
        frame_objects = record_objects(objects)
        out_dict = {"result": [{"frame_id": 0, "objects": frame_objects}]}
        logger.info("output json first frame %s" % str(out_dict))

        for idx, frame in enumerate(vl.frames[1:], start=1):
            if idx % 10 == 0:
                logger.info("running frame: %d" % idx)
            detected = rt.roi_full(frame)
            objects, matched_idx = tracker.tracking(
                objects, detected, idx
            )

            detected["names"] = [objects[j].name for j in matched_idx]
            detected["track_scores"] = [objects[j].track_score for j in matched_idx]

            # detected["masks"] = None
            vz.draw(frame, detected, "%s_%04d" % (basename, idx))

            # record the outputs
            fr_obj = record_objects([obj for i, obj in enumerate(objects) if i in matched_idx])
            out_dict["result"].append({"frame_id": idx, "objects": fr_obj})

            torch.cuda.empty_cache()

        with open("output/out.json", "w") as outfile:
            json.dump(out_dict, outfile, indent=4)

    if args.images:
        for imagename in args.images:
            basename = os.path.basename(imagename)
            basename = os.path.splitext(basename)[0]

            image = cv2.imread(imagename)
            detected = rt.get_detection(image)
            features = rt.generate_features(image, detected)
            vz.draw(image, detected, basename)
    tdif = time() - t_start
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    main()

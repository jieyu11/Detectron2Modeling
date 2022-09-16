# need to run this in docker 
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import os
import logging
from time import time
from datetime import timedelta
import argparse
import random
import cv2
import numpy as np

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class Segmentation:
	"""Detector class. Make object detection given any model in the framework
	of detectron2.
	"""
	colors = [(0, 255, 255), (255, 0, 128),
           (0, 128, 128), (128, 128, 0), (128, 0, 128), (255, 0, 0),
           (0, 0, 255), (0, 0, 128), (0, 128, 0), ]

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
		yamlfile = "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
		yamlfile = parameters["yamlfile"] if "yamlfile" in parameters else yamlfile
		weightfile = "pretrained/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
		weightfile = parameters["weightfile"] if "weightfile" in parameters else weightfile

		model_name = parameters.get("model_name", "detector")
		self.metadata = MetadataCatalog.get(model_name)
		self.metadata.thing_classes = parameters["thing_classes"]

	
		self.cfg = get_cfg()
		self.cfg.merge_from_file(yamlfile)
		threshold = parameters.get("confidence_threshold", 0.7)
		self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
		self.cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = threshold
		self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.metadata.thing_classes)
		self.cfg.MODEL.WEIGHTS = weightfile
		self.cfg.freeze()
		self.predictor = DefaultPredictor(self.cfg)
		self.nframes = parameters.get("nframes", 1e6)
	
	@staticmethod
	def read_video_file(videofile, nframes=1e6):
		"""Read a video file .

		Args:
			videofile ([type]): full path to the input video.
			nframes ([type], optional): [description]. Defaults to 1e6.

		Returns:
			[list]: the list of frames read from the input video.
		"""
		cap = cv2.VideoCapture(videofile)
		frames = []
		idx = 0
		while(cap.isOpened()):
			if idx >= nframes:
				break

			ret, frame = cap.read()
			if not ret:
				logger.error("Can't receive frame (stream end?). Exiting ...")
				break
			# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			# frame = frame[:, :, ::-1]
			frames.append(frame)
			idx += 1
		cap.release()
		cv2.destroyAllWindows()
		logger.info("Read %d frames from %s." % (len(frames), videofile))
		return frames

	def inference_video(self, videofile, outdir, nframes=1e6):
		"""Run object detection inference on the given video.

		Args:
			videofile (str): full path to the video file.
			outdir (str): full path to the output file location.
			nframes (int, optional): the number of frames to run. Defaults to 1e6.
		"""
		os.makedirs(outdir, exist_ok=True)
		frames = self.read_video_file(videofile, nframes)
		assert len(frames) > 0, "No frames in the video: %s" % videofile

		fbase = os.path.basename(videofile)
		fbase = os.path.splitext(fbase)[0]
		out_file = os.path.join(outdir, fbase + '.mp4')
		outvideo = cv2.VideoWriter(
			out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frames[0].shape[1],
			frames[0].shape[0]), True
		)
		logger.info("Saving video with shape: %s" % str(frames[0].shape))

		for idx, frame in enumerate(frames):
			result = self._make_image_inference(frame)
			# frame_detected = self._make_detected_frame(frame, result)
			frame_detected = result["frame"]
			outvideo.write(frame_detected)
			if idx % 50 == 0:
				logger.info("finished with %dth frame." % idx)
			cv2.imwrite(os.path.join(outdir, fbase+"_frame_%04d.png" % idx), frame)

			# save the masks into images
			obj_id = 0
			for mask, lb in zip(result["masks"], result["classes"]):
				mask = mask.numpy().astype(np.uint8)
				print(np.sum(mask))
				mask[mask==True] = 255
				mask[mask==False] = 0
				print(mask.shape, np.sum(mask==255))
				img_shape = tuple([mask.shape[0], mask.shape[1], 3])
				img = np.zeros(img_shape)
				img[:, :, 0] = mask
				img[:, :, 1] = mask
				img[:, :, 2] = mask
				slb = self.metadata.thing_classes[lb].replace(" ", "-")
				fname = fbase+"_frame_%04d_mask-%d_%s.png" % (idx, obj_id, slb)
				cv2.imwrite(os.path.join(outdir, fname), img)
				obj_id += 1

		logger.info("video saved to: %s with %d frames" % (out_file, len(frames)))
		outvideo.release() 

	def _make_detected_frame(self, frame, result: dict):
		"""Use original frame and prediction result to make a frame overlayed with
		prediction bounding boxes.

		Args:
			frame (np.ndarray): original image
			result (dict): prediction results, including keys of boxes, classes,
				scores and masks
		Returns:
			np.ndarray: image overlayed with prediction results.
		"""
		frame_detected = frame.copy()
		boxes = result["boxes"].tensor.numpy()
		for ilabel, box, score in zip(result["classes"], boxes, result["scores"]):
			x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
			color = self.colors[ilabel]
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
			txt = self.metadata.thing_classes[ilabel]
			txt += "(%.0f%%)" % (100 * score)
			cv2.putText(frame, txt,
						(x1, y1-10), 0, 0.5, color)
		return frame_detected

	def inference_images(self, imagefiles, outdir):
		for fname in imagefiles:
			fbase = os.path.basename(fname)
			outname = os.path.join(outdir, fbase)
			im = cv2.imread(fname)
			# frame = self._make_image_inference(im)
			result = self._make_image_inference(im)
			frame = result["frame"]
			logger.info("Write inference to %s" % outname)
			cv2.imwrite(outname, frame)
		
	def _make_image_inference(self, im):
		"""Make inference on an image

		Args:
			im (np.array): input image array.

		Returns:
			dict: including keys and values of
				- boxes: the list of detected bounding boxes
				- scores: the list of confidence scores of each bounding box
				- classes: the list of the classes each bounding boxes is 
					associated with.
		"""
		outputs = self.predictor(im)
		predictions = outputs["instances"].to("cpu")
		boxes = predictions.pred_boxes
		scores = predictions.scores
		classes = predictions.pred_classes
		masks = predictions.pred_masks
		visualizer = Visualizer(im[:, :, ::-1],
                          metadata=self.metadata,
                          scale=1.0,
                          # _BW   # remove the colors of unsegmented pixels
                          instance_mode=ColorMode.IMAGE
                          )
		v = visualizer.draw_instance_predictions(predictions)
		frame = v.get_image()[:, :, ::-1]
		result = {
			"boxes": boxes,
			"scores": scores,
			"classes": classes,
			"masks": masks,
			"frame": frame
		}
		return result

def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--yamlconfig", "-y", default=None,
		type=str, required=False,
		help="Yaml config file",
	)
	parser.add_argument(
		"--weight-file", "-w",
		type=str, required=True,
		help="Retrained weight file",
	)
	parser.add_argument("--classes", "-c", default=["person", "sports ball"], nargs="+",
		type=str, required=True,
		help="names of the classes / categories.",
	)
	parser.add_argument("--inference-video", "-v", default=None,
		type=str, required=False,
		help="video for inference.",
	)
	parser.add_argument(
		"--outdir", "-o", default="output/inference", type=str, required=True,
		help="Your output folder.",
	)
	parser.add_argument(
		"--nframes", "-n", default=1e6, type=int, required=False,
		help="Your number of frames to be read from input video.",
	)
	parser.add_argument(
		"--confidence-threshold", "-th", default=0.7, type=float, required=False,
		help="Your confidence threshold for detected objects.",
	)
	parser.add_argument("--inference-images", "-i", default=None, nargs="+",
		type=str, required=False,
		help="list of images for inference.",
	)

	args = parser.parse_args()
	logger.info("classes: %s" % str(args.classes))
	config = {
		"thing_classes": args.classes,
		"weightfile": args.weight_file,
		"nframes": args.nframes,
		"confidence_threshold": args.confidence_threshold,
	}
	if args.yamlconfig:
		config["yamlfile"] = args.yamlconfig

	os.makedirs(args.outdir, exist_ok=True)
	rt = Segmentation(config)
	if args.inference_video is not None:
		rt.inference_video(
			videofile=args.inference_video,
			outdir=args.outdir,
			nframes=args.nframes
		)
	if args.inference_images is not None:
		rt.inference_images(
			imagefiles=args.inference_images,
			outdir=args.outdir
		)
	
	tdif = time() - t_start
	logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
	main()

import os
import logging
import cv2
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class Visualizer:
	"""Detector class. Make object detection given any model in the framework
	of detectron2.
	"""
	colors = [(255,   0,   0), (  0, 255,   0), (255,   0, 255), (255, 255,   0),
           	  (  0, 255, 255), (255,   0, 128), (  0, 128, 128), (128, 128,   0),
			  (128,   0, 128), (255,   0,   0), (  0,   0, 255), (  0,   0, 128)]

	def __init__(self, thing_classes, outdir="output/viz"):
		self.thing_classes = thing_classes
		self.outdir = outdir
		os.makedirs(self.outdir, exist_ok=True)

	def draw(self, frame, detected, outname):
		frame = frame.copy()
		# fontScale = min(frame.shape[0], frame.shape[1]) / 1000
		fontScale = 0.75 * frame.shape[0] / 1000
		# fontThick = int(3 * frame.shape[0] / 1000)
		fontThick = int(2 * frame.shape[0] / 1000)
		fontThick = fontThick if fontThick > 0 else 1
		boxes = detected["boxes"].tensor.cpu().numpy()
		# boxes = detected["boxes"].cpu().numpy()
		classes = detected["classes"].cpu().numpy()
		# detection confidence score
		scores = detected["scores"].cpu().numpy()
		masks = detected["masks"]
		if masks is not None:
			masks = masks.cpu().numpy()
			# non mask area to white
			mask = masks[0] == False
			for m in masks[1:]:
				mask &= m == False
			
			frame[mask] = [255, 255, 255]

		track_scores = detected.get("track_scores", None)
		names = detected.get("names", None)
		object_id = 0
		for ilabel, box, score in zip(classes, boxes, scores):
			# if object_id > 2: break
			# if ilabel != 0: continue

			x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
			color = self.colors[ilabel]
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
			# txt = "#%d: " % object_id
			txt = self.thing_classes[ilabel] + " (detect=%.1f%%)" % (100 * score)
			# y1 + 15: inside box, y1 - 15 on top of box
			# fontScale = 0.5
			cv2.putText(frame, txt, (x1+5, y1+40), 0, fontScale, color, fontThick)
			if track_scores is not None:
				# sim, matched_id = similarity[object_id]
				# txt = "s = %.5f (matched #%d) " % (sim, matched_id)
				scr = track_scores[object_id]
				name = names[object_id]
				txt = "%s (track=%.1f%%) " % (name, 100*scr)
				cv2.putText(frame, txt, (x1+5, y1+85), 0, fontScale, color, fontThick)
				# cv2.putText(frame, txt, (x1+5, y1+85), 0, 0.75 * fontScale, color,
				# 	max(int(fontThick*0.5), 1))
			object_id += 1
		cv2.imwrite(os.path.join(self.outdir, outname+".png"), frame)

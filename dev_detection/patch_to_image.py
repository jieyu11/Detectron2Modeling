import argparse
from time import time
from datetime import timedelta
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import random
import json

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
					level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchToImage:
	def __init__(self, config):
		"""
		Adding patch to a giving image to create synthetic dataset.	
		
		Parameters:
			config, dict, parameters
				"patches":, str, list of patch image files
				"backgrounds:, str, list of background image files
		"""
		filenames = config.get("patches", None)
		assert filenames, "Define patches in config."
		self.patches = self._load_images(filenames)

		filenames = config.get("backgrounds", None)
		assert filenames, "Define backgrounds in config."
		self.backgrounds = self._load_images(filenames)

		self.category_id = config.get("category_id")
		self.cocodict = {"images": [], "annotations": [],
				   "categories": [
					{'supercategory': 'box',
					'id': self.category_id,
					'name': config.get("category_name")},
				]
				}
		self.image_idx_base = config.get("image_idx_base", 0)
		#In coco , a bounding box is defined by four values in pixels [x_min,
		#y_min, width, height]
		# annotations: 
		# {'image_id': 79966, 
		# 'bbox': [183.53, 228.67, 34.95, 23.02],
		# 'category_id': 2, 'id': 303000
		# }
		# images:
		# 'file_name': 'COCO_val2014_000000327701.jpg', 
		# 'height': 428, 'width': 640,
		# 'id': 327701

	@staticmethod
	def _load_images(imagefiles):
		imagesdict = {}
		for imgname in imagefiles:
			basename = os.path.basename(imgname)
			# basename = os.path.splitext(imgname)[0]
			assert os.path.exists(imgname), "File: %s not found" % imgname
			imagesdict[basename] = cv2.imread(imgname)
		
		logger.info("N frames: %d" % len(imagesdict))
		return imagesdict

	@staticmethod
	def _resize(img, scale_min=0.25, scale_max=1.5):
		scale = random.uniform(scale_min, scale_max) 
		width = int(img.shape[1] * scale)
		height = int(img.shape[0] * scale)
		dim = (width, height)
		# resize image
		resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
		return resized
	
	@staticmethod
	def _flip(img):
		"""
		axis = 0 (vertical), 1 (horizental), -1 (both)
		"""
		axis = random.choice([0, 1, -1])
		flipped = cv2.flip(img, axis)
		return flipped

	def _create_new_image(self, patname, bkgname):
		"""
		Create new image given patch image name and background image name
		"""
		pat = self.patches[patname].copy()
		# randomly resized:
		pat = self._resize(pat)
		# randomly flipped:
		pat = self._flip(pat)

		w1, h1 = pat.shape[0], pat.shape[1]
		img = self.backgrounds[bkgname].copy()
		w2, h2 = img.shape[0], img.shape[1]
		resize_scale = 1.0
		while w2//2 <= w1 or h2//2 <= h1:
			resize_scale *= 0.75
			minsize, maxsize = resize_scale / 3., resize_scale
			pat = self._resize(pat)
			w1, h1 = pat.shape[0], pat.shape[1]

		rnd1 = random.randint(0, w2 - w1)
		rnd2 = random.randint(0, h2 - h1)
		img[rnd1:rnd1+w1, rnd2:rnd2+h1, :] = pat
		# x_min, y_min, width, height
		# box = [rnd1, rnd2, w1, h1]
		box = [rnd2, rnd1, h1, w1]
		return img, box

	def _update_metadata(self, idx, img, box, name):
		cat_id=self.category_id
		self.cocodict["images"].append(
			{"file_name": name,
			"height": img.shape[0],
			"width": img.shape[1],
			"id": idx,
			}
		)
		self.cocodict["annotations"].append(
			{
				"image_id": idx,
				"bbox": box,
				"category_id": cat_id,
				"id": idx,
			}
		)

	def generate_images(self, outfolder, repeat=100):
		idx = self.image_idx_base
		os.makedirs(outfolder, exist_ok=True)
		cat_id = 1
		for patname in self.patches:
			for bkgname in self.backgrounds:
				for _ in range(repeat):
					imgname = "%06d.png" % idx
					outname = os.path.join(outfolder, imgname)
					img, box = self._create_new_image(patname, bkgname)
					self._update_metadata(idx, img, box, imgname)
					logger.info("converted image: %s" % outname)
					cv2.imwrite(outname, img)
					idx += 1
	def save_coco(self, outname):
		with open(outname, 'w') as f:
			json.dump(self.cocodict, f)
		logger.info("coco format file: %s" % outname)

def main():
	t_start = time()
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--patches", "-p", default=None, type=str, required=True,
		nargs="+",
		help="Your input patch image files.",
	)
	parser.add_argument(
		"--backgrounds", "-b", default=None, type=str, required=True,
		nargs="+",
		help="Your input background image files.",
	)
	parser.add_argument(
		"--category", "-cid", default=None, type=int, required=True,
		help="Your input image category id.",
	)
	parser.add_argument(
		"--name", default=None, type=str, required=True,
		help="Your input image category name.",
	)
	parser.add_argument(
		"--image-index-base", "-imgidx", default=0, type=int, required=True,
		help="Your input image index starting number.",
	)
	parser.add_argument(
		"--output", "-o", default="images/synthetic", type=str, required=True,
		help="Your output folder.",
	)
	parser.add_argument(
		"--coco", "-c", default="coco.json", type=str, required=False,
		help="Your output coco annotation file.",
	)
	parser.add_argument(
		"--nrepeat", "-n", default=100, type=int, required=False,
		help="Your number of random patch locations.",
	)



	args = parser.parse_args()
	pa = PatchToImage({
		"patches": args.patches,
		"backgrounds": args.backgrounds,
		"category_id": args.category,
		"category_name": args.name,
		"image_idx_base": args.image_index_base,
	})
	pa.generate_images(args.output, args.nrepeat)
	pa.save_coco(args.coco)
	tdif = time() - t_start
	logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
	"""
	Execute example:
		python3 patch_to_image.py \
			-p patch_01.png patch_02.png ... \
			-b background_01.png background_02.png ... \
			-o /path/to/output
	"""
	main()

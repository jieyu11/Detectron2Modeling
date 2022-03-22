from polimorfo.datasets.coco import CocoDataset as Coco
import json
import matplotlib.pyplot as plt

cat_selected = ["person", "sports ball"]
for key in ["val", "train"]:
	print("working on: %s" % key)
	# dump a slected list
	imgfoldname = "/data/coco/%s2014" % key
	coco = Coco('/data/coco/annotations/instances_%s2014.json' % key, image_path=imgfoldname)
	coco.keep_categories(cat_selected)
	res = coco.dumps()
	jsfilename = '/outputs/person_sports-ball_%s.json' % key
	coco.dump(jsfilename)

	# dump the filtered images and the segmented masks
	coco = Coco(jsfilename, image_path=imgfoldname)
	out_path = "/outputs/output_person_sports-ball_%s" % key
	coco.save_images_and_masks(out_path)
	print("out:", out_path)

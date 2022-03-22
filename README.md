# Detectron2 Model Retraining

Detectron2 ([github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2))
is Facebook AI Research's next generation library that provides state-of-the-art
detection and segmentation algorithms. 


## Build Docker Image
Assuming in the current directory (where the `Dockerfile` is located), building the docker
image can be done simply with:
```
docker build --build-arg USER_ID=$UID -t detectron2:v0 .
```
which builds the docker image with tag: `detectron2:v0`. The following processes rely on the
docker image built here.

### Launch Docker Container
To launch docker container, it assumes the CoCo Dataset is downloaded on the local server as
described in the next section.

Put the following code into `launch_docker_container.sh` and run `source launch_docker_container.sh`
```
#!/bin/bash

cocodataset=${HOME}/3dselfie/3DSelfie/data/coco-dataset-2014
imagedir=/path/to/images/for/inference
videodir=/path/to/videos/for/inference
outdir=/path/to/where/to/put/outputs
mkdir -p ${outdir} 

docker run --gpus 1 --rm \
 	-it \
	-v ${outdir}:/outputs \
	-v ${cocodataset}:/data/coco \
	-v ${imagedir}:/data/images \
	-v ${videodir}:/data/videos \
	detectron2:v0
```

## CoCo Dataset Image Filtering
CoCo dataset ([here](https://cocodataset.org/#home)) is used for model retraining in this project.

To start a docker container, we launch it by including the initial CoCo Dataset which is already
downloaded to our local server.

Coco Dataset 2014 is downloaded to:
```
192.168.1.74:/groups1/3DSelfie/data/coco-dataset-2014
```

If one has mounted it to the local server with:
```
mkdir $HOME/3dselfie
sudo mount -t nfs 192.168.1.74:/groups1 $HOME/3dselfie
```

Then, it can be listed through
```
ls ${HOME}/3dselfie/3DSelfie/data/coco-dataset-2014
> annotations test2014 train2014 val2014 densepose
```
The Coco images are under `train/val/test2014`, which the `annotations` and `densepose`
has the segmentation information.


### CoCo Annotation Structure
For this project, we take the segmentation as example. The structure of the Coco annotation is discussed
with example file: `${HOME}/3dselfie/3DSelfie/data/coco-dataset-2014/annotations/instances_val2014.json`.

It is a json file, which essentially is a python dict. It has five top level keys:
`"info"`, `"licenses"`, `"images"`, `"annotations"`, `"categories"`.

It can be read through python:
```
with open(input_coco_json_path) as json_file:
    coco_dict = json.load(json_file)
```

**CoCo Info**

One can retrieve the coco info through:
```
print(coco_dict["info"])

{'description': 'COCO 2014 Dataset', 
 'url': 'http://cocodataset.org', 
 'version': '1.0', 
 'year': 2014,
 'contributor': 'COCO Consortium',
 'date_created': '2017/09/01'
}
```

**CoCo Licenses**

Licenses are list of python dict's.
```
print(coco_dict["licenses"])

[
{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'},
{'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'},
{'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'},
{'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'},
{'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'},
{'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'},
{'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'},
{'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}
]
```

**CoCo Images**

CoCo images are list of dict's, where each dict represent one image in CoCo Dataset.

```
print(coco_dict["images"][0])

{
'license': 3, 
'file_name': 'COCO_val2014_000000391895.jpg',
'coco_url': 'http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg',
'height': 360, 'width': 640,
'date_captured': '2013-11-14 11:18:45',
'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
'id': 391895
}
```

Note that `instances_val2014.json` has **40504** images, which `instances_train2014.json`
has **82783**.


**CoCo Annotations**

CoCo annotations are list of dict's, where each annotation dict is an object segmentation.
Note that there can be multiple segments on one single image. 

```
print(coco_dict["annotations"][0])

{
'segmentation': [[239.97, 260.24, 222.04, 270.49, 199.84, 253.41, 213.5, 227.79, 259.62, 200.46, 274.13, 202.17, 277.55, 210.71, 249.37, 253.41, 237.41, 264.51, 242.54, 261.95, 228.87, 271.34]],
'area': 2765.1486500000005,
'iscrowd': 0,
'image_id': 558840,
'bbox': [199.84, 200.46, 77.71, 70.88],
'category_id': 58,
'id': 156
}
```

Note that `instances_val2014.json` has **291875** annotations, while `instances_train2014.json` has **604907**.
Meanwhile, the key of `image_id` can help to retrieve the corresponding image of the segment.

**CoCo Categories**

CoCo has 90 categories, which can be found:

```
print(coco_dict["categories"])

[
  {'supercategory': 'person', 'id': 1, 'name': 'person'},
  {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'},
  {'supercategory': 'vehicle', 'id': 3, 'name': 'car'},
  ...
  ...
  {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'},
  {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}
]
```

the whole list of the 90 categories and names can be found in
*coco_categories.json* in this repository.


### Annotation JSON Filtering
To selectively use part of the CoCo Dataset for model training, a filter with the
initial annotation file is needed. For example, only categories of
`"person"` and `"sports ball"` are needed for model retraining / finetuning, then
the rest of the categories are filtered out. To do so, we use `coco-manager` 
found [github.com/immersive-limit/coco-manager](https://github.com/immersive-limit/coco-manager).

```
cd coco-manager
python filter.py --input_json /data/coco/annotations/instances_train2014.json \
	--output_json person_sportsball_train2014.json \
	--categories "person" "sports ball"
```

After the filtering, the number of images and annotations dropped to **21694** and **90371**
for `preson_sportsball_val2014.json` and **45289** and **189708** in `preson_sportsball_train2014.json`.

Note after the filtering, the output `json` file has only two categories and their `id`s are 
`1: person, 2: sports ball`.

#### Further Filtering Towards Balance Categories
From the filtering above, the category of `"person"` has
far more instances compared to the category of `"sports ball"`,
where `"person": 88153, "sports ball": 2218` are found for
`preson_sportsball_val2014.json` and `"person": 185316, "sports ball": 4392` for `preson_sportsball_train2014.json`.

The amount of instances in the two categories have huge difference, which can cause issue in model training, where the model may not be able to detect `"sports ball"` correctly.

To avoid the instances imbalance, we further filtered the annotation `json` to only keep images having both categories.

```
cd work/coco_filtering
python filter_balance_instance.py -i ../coco-manager/person_sportsball_val2014.json \
    -o person_sportsball_val2014_balanced.json
```
which reduces `"person"` instances to **7535** in `val*json` and **14477** in
`train*json` while keeping the same amount of `"sports ball"` instances.

These filtered and balanced json file is used later for model retraining.

### Image Filtering And Segmentation Creation
**(Note: this step is optional.)**
One can also filter the images and save them to a different location, meanwhile, the segmentation 
masks can also be generated. This tool can be useful when disk space is limited or when segmentation
masks are needed for model traing, e.g. in DeepLabV3 retraining ([here](https://gitlab.g6labs.com/3dselfies/gepetto/-/tree/add-detectron2-model-retraining/third_party/BgSegmentation/BgSegmentation_DeepLabRetrain/ModelRetraining)).

To filter out the images for `"person"` and `"sports ball"`, a simple example code using
[polimorfo](https://github.com/fabiofumarola/polimorfo/blob/master/docs/installation.rst)
can be executed in the docker container:
```
cd work/coco_filtering
python filter_images_save_segments.py
```

For this project, we assume this step is not proceeded and only use the original CoCo Dataset
with the filtered annotation `json` file obtained in the section above.

## Segmentation Model Retraning with Detectron2

### Segmentation Model Retraining
When the images are filtered and the amount of instances are balanced as mentioned above, then we can continue to retrain a segmentation model.

The code we developed is located in: `work/retrain_person_sportsball.py` in a docker container.

To understand the usage, run the `-h` option:
```
$ python retrain_person_sportsball.py -h 
usage: retrain_person_sportsball.py [-h] [--traindir TRAINDIR]
                                    [--trainjson TRAINJSON]
                                    [--classes CLASSES [CLASSES ...]]
                                    [--inference-images INFERENCE_IMAGES [INFERENCE_IMAGES ...]]
                                    [--inference-video INFERENCE_VIDEO]
                                    [--retrain-even-exists]

optional arguments:
  -h, --help            show this help message and exit
  --traindir TRAINDIR, -t TRAINDIR
                        train images dir
  --trainjson TRAINJSON, -tj TRAINJSON
                        train json file
  --classes CLASSES [CLASSES ...], -c CLASSES [CLASSES ...]
                        names of the classes / categories.
  --inference-images INFERENCE_IMAGES [INFERENCE_IMAGES ...], -i INFERENCE_IMAGES [INFERENCE_IMAGES ...]
                        list of images for inference.
  --inference-video INFERENCE_VIDEO, -v INFERENCE_VIDEO
                        video for inference.
  --retrain-even-exists
                        Use argument to do retraining even output model exists.
```

Notice the `configfile` and `weights` parameters in the file `retrain_person_sportsball.py`, which is the baseline model for
retraining.

For example:
```
configfile="../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
weights="detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
```

To run model retraining, one can execute:
```
python retrain_person_sportsball.py \
    	--classes "person" "sports ball" \
        --traindir /data/coco/train/images \
        --trainjson coco_filtering/person_sportsball_train2014_balanced.json \
		--retrain-even-exists
```
when it finishes, it produces a model file: `output/model_final.pth`.
Note, the `classes` arguments here should match exactly with the ones
in the previous sections (e.g. `"person"`, `"sports ball"`).

### Images Inference with Retrained Model
The same code is used for images inference, one can use the same input config `json` file. Simply run:
```
python retrain_person_sportsball.py -i /data/images/*rig*.png
```
by default, the outputs will be in `output/inferences/` with the same image names. 

### Video Inference with Retrained Model
The same code is used for video inference as well. Simply run:
```
python retrain_person_sportsball.py -i /data/videos/my-selected-video.mp4
```
by default, the outputs are images of all frames in `output/inferences/${VIDEO_NAME}_frame_%04d.png` with `${VIDEO_NAME}` being the video image names. The output video `output/inferences/${VIDEO_NAME}.mp4` sometimes has issues and cannot be opened. One can use `ffmpeg` to overwrite it:

```
ffmpeg -framerate 30 -i output/inferences/${VIDEO_NAME}_frame_%04d.png \
	-c:v libx264 -vf "fps=25,format=yuv420p"
	output/inferences/${VIDEO_NAME}.mp4
```

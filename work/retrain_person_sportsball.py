# need to run this code in a detectron docker container
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
import os
import logging
from time import time
from datetime import timedelta
import argparse
import random
import cv2

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# check example: https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/
class retraining:
    def __init__(self, name, imagedir, jsonfile, thing_classes):
        self.name = name
        register_coco_instances(name, {}, jsonfile, imagedir)
        self.metadata = MetadataCatalog.get(name)
        self.metadata.thing_classes = thing_classes
        # list[dict] for each dict is a image + metadata
        # self.data[0]["file_name"] is the file name of the 0th image
        self.data  = DatasetCatalog.get(name)
        self.model_outname = "model_final.pth"

    def set_config(self, parameters = {}):
        # assuming running in ./detectron2_repo/work/
        yamlfile = "../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        yamlfile = parameters["yamlfile"] if "yamlfile" in parameters else yamlfile
        weightfile = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
        weightfile = parameters["weightfile"] if "weightfile" in parameters else weightfile

        self.cfg = get_cfg()
        self.cfg.merge_from_file( yamlfile )
        # dataset name is initialized 
        self.cfg.DATASETS.TRAIN = (self.name)
        self.cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
        self.cfg.DATALOADER.NUM_WORKERS = parameters["num_workers"] if "num_workers" in parameters else 4
        # initialize from model zoo
        self.cfg.MODEL.WEIGHTS = weightfile
        self.cfg.SOLVER.IMS_PER_BATCH = 2
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.SOLVER.MAX_ITER = 1200 # 300
        # 300 iterations seems good enough, but you can certainly train longer
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 # 128 # 512 # (  )
        # faster, and good enough for this toy dataset
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(self.metadata.thing_classes)
        if "output" in parameters:
            self.cfg.OUTPUT_DIR = parameters["output"]
        logger.info("output: %s" % self.cfg.OUTPUT_DIR)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
    
    def train(self, force_retrain=False):
        outmodel = os.path.join(self.cfg.OUTPUT_DIR, self.model_outname)
        if os.path.exists(outmodel) and not force_retrain:
            logger.info("Model %s exists. Not retrained." % outmodel)
            return

        logger.info("Start of training.")
        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        logger.info("End of training... output in the folder:")
        os.listdir(self.cfg.OUTPUT_DIR)

    def inference(self, imagefiles=None, videofile=None, outdir="output/inferences"):
        """
        imagefiles: list of str of image files.
        """
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, self.model_outname)
        # set the testing threshold for this model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8 # 0.5
        self.cfg.DATASETS.TEST = (self.name)
        self.predictor = DefaultPredictor(self.cfg)
        os.makedirs(outdir, exist_ok=True)
        if imagefiles:
            self._inference_images(imagefiles, outdir)
        if videofile:
            self._inference_video(videofile, outdir)

    def _inference_video(self, videofile, outdir):
        fbase = os.path.basename(videofile)
        fbase = os.path.splitext(fbase)[0]
        out_file = os.path.join(
            outdir, fbase + '.mp4')
        outvideo = None
        cap = cv2.VideoCapture(videofile)
        nframes = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                logger.error("Can't receive frame (stream end?). Exiting ...")
                break
            if not outvideo:
                outvideo = cv2.VideoWriter(
                    out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                    (frame.shape[1], frame.shape[0]), True)
                logger.info("Saving video with shape: %s" % str(frame.shape))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame[:] = frame[:, :, ::-1]
            frame = self._make_image_inference(frame)
            outvideo.write(frame)
            if nframes % 50 == 0:
                logger.info("finished with %dth frame." % nframes)
            # save images for debugging
            cv2.imwrite(os.path.join(outdir, fbase+"_frame_%04d.png" % nframes), frame)
            nframes += 1
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

        logger.info("video output saved to: %s with %d frames" % (out_file, nframes))
    
    def _inference_images(self, imagefiles, outdir, example_from_dataset=False):
        if imagefiles is None and example_from_dataset:
            imagefiles = random.sample(self.data, 10)
            imagefiles = [d["file_name"] for d in imagefiles]

        if not imagefiles:
            logger.info("No images in the list. Retrun.")
            return

        for fname in imagefiles:
            fbase = os.path.basename(fname)
            outname = os.path.join(outdir, fbase)
            im = cv2.imread(fname)
            frame = self._make_image_inference(im)
            logger.info("Write inference to %s" % outname)
            cv2.imwrite(outname, frame)
    
    def _make_image_inference(self, im):
            outputs = self.predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=self.metadata,
                           scale=0.6,
                           # remove the colors of unsegmented pixels
                           instance_mode=ColorMode.IMAGE_BW   
                           )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            frame = v.get_image()[:, :, ::-1]
            return frame

def main():
    t_start = time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", "-n", default="person_ball_train",
        type=str, required=False,
        help="name of the dataset instance",
    )
    parser.add_argument(
        "--traindir", "-t", default="/data/coco/train/images",
        type=str, required=False,
        help="train images dir",
    )
    parser.add_argument(
        "--trainjson", "-tj", default="coco_filtering/person_sportsball_train2014_balanced.json",
        type=str, required=False,
        help="train json file",
    )
    parser.add_argument("--classes", "-c", default=["person", "sports ball"], nargs="+",
        type=str, required=False,
        help="names of the classes / categories.",
    )
    parser.add_argument("--inference-images", "-i", default=None, nargs="+",
        type=str, required=False,
        help="list of images for inference.",
    )
    parser.add_argument("--inference-video", "-v", default=None,
        type=str, required=False,
        help="video for inference.",
    )
    parser.add_argument("--retrain-even-exists", dest="force_retrain",
        action="store_true", 
        help="Use argument to do retraining even output model exists.",
    )

    args = parser.parse_args()
    logger.info("classes: %s" % str(args.classes))

    rt = retraining(args.name, args.traindir, args.trainjson, args.classes)
    rt.set_config()
    rt.train(args.force_retrain)
    rt.inference(imagefiles=args.inference_images, videofile=args.inference_video)
    tdif = time() - t_start
    logger.info("Time used: %s" % str(timedelta(seconds=tdif)))


if __name__ == "__main__":
    # use:
    #         python retrain_person_sportsball.py -h
    # for how to run the code
    main()

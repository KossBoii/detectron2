import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
import json
from detectron2.structures import BoxMode

# Custom Dataset Mapper
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import copy

from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from datetime import datetime

def get_roadstress_dicts(img_dir):
    # Load and read json file stores information about annotations
    json_file = os.path.join(img_dir, "via_export_json.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []          # list of annotations info for every images in the dataset
    for idx, v in enumerate(imgs_anns.values()):
        if(v["regions"]):
            record = {}         # a dictionary to store all necessary info of each image in the dataset
            
            # open the image to get the height and width
            filename = os.path.join(img_dir, v["filename"])
            height, width = cv2.imread(filename).shape[:2]
            
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
          
            # getting annotation for every instances of object in the image
            annos = v["regions"]
            objs = []
            for anno in annos:
                anno = anno["shape_attributes"]
                px = anno["all_points_x"]
                py = anno["all_points_y"]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts


class Trainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg: CfgNode, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=customMapper(cfg))

    @classmethod
    def build_train_loader(cls, cfg: CfgNode):
        return build_detection_train_loader(cfg, mapper=customMapper)
 

# Custom DatasetMapper
def customMapper(dataset_dict):
  dataset_dict = copy.deepcopy(dataset_dict)
  image = utils.read_image(dataset_dict["file_name"], format="BGR")

  transform_list = [
                    T.Resize((800, 1333)),
                    T.RandomFlip(prob=0.6, horizontal=True, vertical=False),
                    T.RandomFlip(prob=0.6, horizontal=False, vertical=True),
                    T.RandomBrightness(0.5, 1.8),
                    ]
  image, transforms = T.apply_transform_gens(transform_list, image)
  dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
  annos = [
		utils.transform_instance_annotations(obj, transforms, image.shape[:2])
		for obj in dataset_dict.pop("annotations")
		if obj.get("iscrowd", 0) == 0
	]
  instances = utils.annotations_to_instances(annos, image.shape[:2])
  dataset_dict["instances"] = utils.filter_empty_instances(instances)
  return dataset_dict

if __name__ == "__main__":
    # Register the dataset:
    for d in ["train", "val"]:
    	DatasetCatalog.register("roadstress_" + d, lambda d=d: get_roadstress_dicts("roadstress_new/" + d))
    	MetadataCatalog.get("roadstress_" + d).set(thing_classes=["roadstress"])
    	roadstress_metadata = MetadataCatalog.get("roadstress_train")
    
    # Configure detectron2's configs
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("roadstress_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Transfer learning with weights from model_zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.005  # Learning rate
    cfg.SOLVER.MAX_ITER = 20000   
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classification classes excluding the background - only has one class (roadstress)
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-120, -90, -30 , -45, -60, 0, 30, 45, 60, 90, 120]]

    print(cfg.dump())               # print out all the info in the model configuration
    print("Done config")

    curTime = datetime.now()
    cfg.OUTPUT_DIR = "./output/" + curTime.strftime("%m%d%Y%H%M%S")

    # Start training model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()




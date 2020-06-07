import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Logging
import logging
logger = logging.getLogger("detectron2")

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
import json
from detectron2.structures import BoxMode

# Custom Dataset Mapper
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import copy

# Custom Training Loop
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import CommonMetricPrinter, EventStorage, JSONWriter, TensorboardXWriter
import detectron2.utils.comm as comm
from datetime import datetime
from detectron2.modeling import build_model

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

def config():
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
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256, 512]]
    cfg.TEST.DETECTIONS_PER_IMAGE = 256

    return cfg


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg, mapper=customMapper)
    # data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()                  # call this function at the beginning of each iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 50 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Detectron2 to detect roadstress")
    parser.add_argument('--dataset', required=False, metavar="/path/to/roadstress/dataset/", help='Directory of the roadstress dataset')
    parser.add_argument("--weights", required=True, metavar="/path/to/weights.pth", help="Path to weights .pth file or 'coco'")
    parser.add_argument("command", metavar="<command>", help="'train' or 'eval'")
    
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "eval":
        assert args.weights, "Argument --weights is required for evaluation"

    print("Mode: ", args.command)
    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)

    # Register the dataset:
    for d in ["train", "val"]:
        DatasetCatalog.register("roadstress_" + d, lambda d=d: get_roadstress_dicts("roadstress_new/" + d))
        MetadataCatalog.get("roadstress_" + d).set(thing_classes=["roadstress"])
        roadstress_metadata = MetadataCatalog.get("roadstress_train")
    print("Done Registering the dataset")

    if args.command == "train":
        

        # Configure detectron2's configs
        cfg = config()

        # Load in the weights:
        if args.weights.lower() == "coco":
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        else:
            assert os.path.exists(args.weights), "Path to weights %s not exists" % args.weights
            cfg.MODEL.WEIGHTS = args.weights

        # Setup Logging folder
        curTime = datetime.now()
        cfg.OUTPUT_DIR = "./output/" + curTime.strftime("%m%d%Y%H%M%S")
        if not os.path.exists(os.getcwd() + "/output/"):
            os.mkdir(os.getcwd() + "/output/")
        os.mkdir(os.getcwd() + "/output/" + curTime.strftime("%m%d%Y%H%M%S"))        

        # print out all the info in the model configuration
        print(cfg.dump())               

        # Train the model
        model = build_model(cfg)
        do_train(cfg, model)
        print("Done Training!")
    elif args.command == "eval":
        cfg = config()

        assert os.path.exists(args.weights), "Path to weights %s not exists" % args.weights
        cfg.MODEL.WEIGHTS = args.weights

        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader

        evaluator = COCOEvaluator("roadstress_val", cfg, False, output_dir="./output/")
        
        val_loader = build_detection_test_loader(cfg, "roadstress_val")
        inference_on_dataset(build_model(cfg), val_loader, evaluator)

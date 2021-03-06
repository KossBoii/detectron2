# import some common libraries
import numpy as np
import cv2
import random
import json
from detectron2.structures import BoxMode
#from google.colab.patches import cv2_imshow

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from skimage.io import imshow, imsave

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


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
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

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
                #and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
            
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

# def setup(args):
#     """
#     Create configs and perform basic setups.
#     """

#     for d in ["train", "val"]:
#         DatasetCatalog.register("roadstress_" + d, lambda d=d: get_roadstress_dicts("roadstress_new/" + d))
#         MetadataCatalog.get("roadstress_" + d).set(thing_classes=["roadstress"])

#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.DATASETS.TRAIN = ("roadstress_train",)
#     cfg.DATASETS.TEST = ("roadstress_val",)
#     cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Transfer learning with weights from model_zoo
#     cfg.SOLVER.IMS_PER_BATCH = 1
#     cfg.SOLVER.BASE_LR = 0.005  # Learning rate
#     cfg.SOLVER.MAX_ITER = 40000 
#     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classification classes excluding the background - only has one class (roadstress)
#     cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-120, -90, -30 , -45, -60, 0, 30, 45, 60, 90, 120]]
#     cfg.SOLVER.CHECKPOINT_PERIOD = 2000
#     cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256, 512]]
#     cfg.TEST.DETECTIONS_PER_IMAGE = 256

#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
#     MetadataCatalog.get("roadstress_val").evaluator_type = "coco"

#     cfg.MODEL.OUTPUT_DIR = "./output/06062020232004/model_0005999.pth"

#     os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     default_setup(
#         cfg, args
#     )  # if you don't like any of the default setup, write your own setup code
#     return cfg


# def main(args):
#     cfg = setup(args)

#     model = build_model(cfg)
#     logger.info("Model:\n{}".format(model))
#     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
#         cfg.MODEL.WEIGHTS, resume=args.resume
#     )
#     return do_test(cfg, model)
#     print("Done")

def visualize(args):
    if not os.path.exists(os.getcwd() + "/predict/"):
        os.mkdir(os.getcwd() + "/predict/")
    targetFolder = args.weights[10:23]
    if not os.path.exists(os.getcwd() + "/predict/%s" % targetFolder):
        os.mkdir(os.getcwd() + "/predict/%s" % targetFolder)
    print("Done making output folder")

    # Register the dataset
    for d in ["train", "val"]:
        DatasetCatalog.register("roadstress_" + d, lambda d=d: get_roadstress_dicts(args.dataset + "/" + d))
        MetadataCatalog.get("roadstress_" + d).set(thing_classes=["roadstress"])
    
    roadstress_metadata = MetadataCatalog.get("roadstress_train")

    # Configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("roadstress_train",)
    cfg.DATASETS.TEST = ("roadstress_val", )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = args.weights
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.005  # Learning rate
    cfg.SOLVER.MAX_ITER = 20000 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classification classes excluding the background - only has one class (roadstress)
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-120, -90, -30 , -45, -60, 0, 30, 45, 60, 90, 120]]
    cfg.SOLVER.CHECKPOINT_PERIOD = 2000
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128, 256, 512]]
    cfg.TEST.DETECTIONS_PER_IMAGE = 1024

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(args.threshold)   # Recommend 0.7
    predictor = DefaultPredictor(cfg)

    
    dataset_dicts = get_roadstress_dicts("%s/val" % args.dataset)
    for d in dataset_dicts:    
        fileName = d["file_name"][-12:-4]
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
    
        v = Visualizer(im[:, :, ::-1],
                    metadata=roadstress_metadata, 
                    scale=1.0, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
        img = v.get_image()[:, :, ::-1]	
        imsave(os.getcwd() + "/predict/%s/%s_result.jpg"%(targetFolder, fileName), img)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict roadstress using Detectron2")
    parser.add_argument('--dataset', required=True, metavar="/path/to/roadstress/dataset/", help='Directory of the roadstress dataset')
    parser.add_argument("--weights", required=True, metavar="/path/to/weights.pth", help="Path to weights .pth file")
    parser.add_argument("--threshold", required=True, help="Threshold for prediction certainty")
    
    args = parser.parse_args()

    print("dataset path: %s" % args.dataset)
    print("dataset weights: %s" % args.weights)
    print("dataset threshold: %s" % args.threshold)

    visualize(args)

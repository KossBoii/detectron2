# from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os

def eval(cfg, trainer):
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    trainer.resume_or_load(resume=True)

    if not os.path.exists(os.getcwd() + "/prediction/"):
        os.mkdir(os.getcwd() + "/prediction/")
    evaluator = COCOEvaluator("roadstress_val", cfg, False, output_dir="./prediction/")
    val_loader = build_detection_test_loader(cfg, "roadstress_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)

if __name__ == "__main__":
    print("this is prediction script")
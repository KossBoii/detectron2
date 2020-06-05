from detectron2.utils.visualizer import ColorMode

if __name__ == "__main__":
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("roadstress_val", )
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_roadstress_dicts("roadstress_new/val")
    for d in random.sample(dataset_dicts, 1):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        print(im.shape)

        v = Visualizer(im[:, :, ::-1],
                    metadata=roadstress_metadata, 
                    scale=1.0, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(v.get_image()[:, :, ::-1])~
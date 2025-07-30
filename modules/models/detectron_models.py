from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

class DetectronModel:
    def init(self):
        self.detectron = None

    def load_detectron(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        cfg.MODEL.DEVICE = "cpu"
        self.detector = DefaultPredictor(cfg)
        # Get dataset name from config, not from DefaultPredictor
        dataset_name = cfg.DATASETS.TRAIN[0]
        self.coco_classes = MetadataCatalog.get(dataset_name).thing_classes
        print(self.coco_classes)
        return self.detector, self.coco_classes


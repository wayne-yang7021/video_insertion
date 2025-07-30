import numpy as np
from PIL import Image
from typing import List, Dict

class ObjectDetector:
    def __init__(self, model):
        """初始化：傳入已載入的模型物件"""
        self.predictor = model.detector
        self.class_names = model.coco_classes

    def detect_objects_in_image(self, image: Image.Image) -> List[Dict]:
        """使用 Detectron2 對圖像執行物體偵測"""
        img_array = np.array(image.convert("RGB"))
        outputs = self.predictor(img_array)

        instances = outputs["instances"].to("cpu")
        results = []

        for i in range(len(instances)):
            result = {
                "label": self.class_names[instances.pred_classes[i]],
                "score": float(instances.scores[i]),
                "box": instances.pred_boxes[i].tensor.numpy()[0].tolist(),
                "mask": instances.pred_masks[i].numpy()
            }
            results.append(result)
        return results

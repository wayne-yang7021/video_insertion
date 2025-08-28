import numpy as np
from PIL import Image
from typing import List, Dict
import cv2

class ObjectDetector:
    def __init__(self, model):
        """初始化：傳入已載入的模型物件"""
        self.predictor = model.detector
        self.class_names = model.coco_classes

    def detect_objects_in_image(self, image: Image.Image) -> List[Dict]:
        """使用 Detectron2 對圖像執行物體偵測"""
        if isinstance(image, np.ndarray):
            img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            img_array = np.array(image.convert("RGB"))
            
        outputs = self.predictor(img_array)

        instances = outputs["instances"].to("cpu")
        results = []

        for i in range(len(instances)):
            score = float(instances.scores[i])
            if score > 0.5:  # 篩選條件
                result = {
                    "label": self.class_names[instances.pred_classes[i]],
                    "score": score,
                    "box": instances.pred_boxes[i].tensor.numpy()[0].tolist(),
                    "mask": instances.pred_masks[i].numpy()
                }
                results.append(result)

        return results

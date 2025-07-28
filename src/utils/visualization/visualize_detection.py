import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict

def visualize_detections(
    image: Image.Image,
    detections: List[Dict],
    output_path: str = "./output/detection_result.jpg"
) -> None:
    """
    將 Detectron2 偵測結果（bounding boxes）畫在圖片上並輸出。

    Args:
        image (PIL.Image.Image): 原圖
        detections (List[Dict]): 偵測結果，每個 dict 包含 'label', 'box', 'score'
        output_path (str): 輸出檔案路徑，預設為 ./output/detection_result.jpg
    """
    # 確保輸出資料夾存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 將 PIL Image 轉成可用於 OpenCV 的格式 (RGB → BGR)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for obj in detections:
        label = obj["label"]
        score = obj["score"]
        x1, y1, x2, y2 = map(int, obj["box"])

        # 畫框
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(img, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 儲存結果圖
    cv2.imwrite(output_path, img)
    print(f"✅ 偵測結果已儲存至：{output_path}")

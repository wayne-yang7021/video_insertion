import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import supervision as sv
from ultralytics import YOLO
from utils.get_video_first_frame import extract_first_frame

# 載入模型
model = YOLO("checkpoints/yolov8m.pt")

# 取得圖片
# image = extract_first_frame("src/data/videos/house_tour.mp4")
image = cv2.imread('src/data/living-room.jpg')

# 執行推論
results = model(image)[0]  # 取得第一張圖片的結果

# 建立 Detections 物件
detections = sv.Detections.from_ultralytics(results)

# 篩選信心值 > 0.5 的偵測結果
detections = detections[detections.confidence > 0.1]

# 用 COCO 類別名稱對應（如果你用的是 YOLOv8 預設模型）=
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()  # 加上這個會顯示對應文字

annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# 顯示
# 如果 annotated_image 是 PIL Image，轉成 numpy
if isinstance(annotated_image, Image.Image):
    annotated_image = np.array(annotated_image)

# OpenCV 是 BGR，但通常圖片是 RGB，要轉色彩
if annotated_image.shape[-1] == 3:
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

cv2.imshow("Detection", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


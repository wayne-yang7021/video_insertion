import sys
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
from modules.models.detectron_models import DetectronModel
from modules.detection.detection import ObjectDetector
from utils.visualization.visualize_detection import visualize_detections
from PIL import Image
import cv2

# 初始化模型
model = DetectronModel()
model.load_detectron()

detector = ObjectDetector(model)

# 使用 OpenCV 讀取圖片並轉為 PIL 格式
img_bgr = cv2.imread("data/living-room.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
image = Image.fromarray(img_rgb)

# 偵測
detections = detector.detect_objects_in_image(image)
visualize_detections(image, detections, output_path="./output/sample_detected.jpg")

for r in detections:
    print(f"{r['label']} ({r['score']:.2f}) → box: {r['box']}")

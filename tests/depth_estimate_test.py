import sys
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
from modules.models.matching_model import DepthEstimator
from modules.matching.depth_matching import DepthHandler
from utils.visualization.visualize_depth import overlay_depth_map, visualize_depth_grayscale, visualize_depth_segmented, draw_bbox_depth_labels

from modules.models.detectron_models import DetectronModel
from modules.detection.detection import ObjectDetector
from utils.visualization.visualize_detection import visualize_detections

from PIL import Image
import cv2
import os

# 初始化模型
detectron_model = DetectronModel()
detectron_model.load_detectron()
detector = ObjectDetector(detectron_model)

depth_model = DepthEstimator()
depth_handler = DepthHandler(depth_model)

# 讀取圖像
img_bgr = cv2.imread("data/living-room.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
image = Image.fromarray(img_rgb)

# 1. 偵測物件
detections = detector.detect_objects_in_image(image)
visualize_detections(image, detections, output_path="./output/sample_detected.jpg")

# 2. 擷取 bbox
bboxes = [det["box"] for det in detections]  # 假設每個 detection 有個 bbox 欄位

# 3. 估計深度圖
depth_map = depth_handler.estimate_depth_map(image)

# 4. 每個 bbox 的平均深度
bbox_depths = depth_handler.get_bbox_depths(depth_map, bboxes)
print("bbox depths:", bbox_depths)

# 5. 視覺化 overlay 深度圖
# overlay 熱圖（你原本的）
vis_overlay = overlay_depth_map(img_rgb, depth_map)
cv2.imwrite("output/vis_overlay_magma.jpg", cv2.cvtColor(vis_overlay, cv2.COLOR_RGB2BGR))

# 灰階深度圖
vis_gray = visualize_depth_grayscale(depth_map)
cv2.imwrite("output/vis_gray.jpg", cv2.cvtColor(vis_gray, cv2.COLOR_RGB2BGR))

# 區段色彩圖
vis_segmented = visualize_depth_segmented(depth_map)
cv2.imwrite("output/vis_segmented.jpg", cv2.cvtColor(vis_segmented, cv2.COLOR_RGB2BGR))

# 原圖 + bbox 標註深度
vis_labeled = draw_bbox_depth_labels(img_rgb, bboxes, bbox_depths)
cv2.imwrite("output/vis_labeled.jpg", cv2.cvtColor(vis_labeled, cv2.COLOR_RGB2BGR))

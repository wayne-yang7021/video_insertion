import sys
import os
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
import cv2
import numpy as np
from PIL import Image
from modules.models.detectron_models import DetectronModel
from modules.detection.detection import ObjectDetector
from modules.models.depth_match_model import DepthEstimator
from modules.matching.depth_matching import DepthHandler
from modules.matching.best_coordinate import OptimalPlacementDetector
from utils.visualization.visualize_detection import visualize_detections

# 初始化模型
detectron_model = DetectronModel()
detectron_model.load_detectron()
detector = ObjectDetector(detectron_model)

depth_model = DepthEstimator()
depth_handler = DepthHandler(depth_model)

placement_detector = OptimalPlacementDetector()

# 讀取圖片
img_bgr = cv2.imread("data/living-room.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
image = Image.fromarray(img_rgb)

# 偵測物件
detections = detector.detect_objects_in_image(image)
visualize_detections(image, detections, output_path="./output/placement_detections.jpg")

# 顯示物件清單給使用者選
print("\n偵測到的物件列表：")
for idx, det in enumerate(detections):
    print(f"[{idx}] label: {det['label']}, score: {det['score']:.2f}")

target_idx = int(input("\n請輸入你想選擇作為放置區域的物件 index："))
target = detections[target_idx]

# 擷取 mask 和 label
mask = target["mask"]  # shape: (H, W)，bool或0/1
label = target["label"]

# 估計深度
depth_map = depth_handler.estimate_depth_map(image)

# 使用最佳置入點檢測器
optimal_points = placement_detector.find_optimal_placement_points(mask.astype(np.uint8), depth_map, img_rgb, object_label=label)

print(f"找到 {len(optimal_points)} 個最佳置入點")
for p in optimal_points:
    print(f"位置: {p['position']}, 分數: {p['score']:.3f}, 置信度: {p.get('confidence', 'N/A')}")

# 在原圖上畫出置入點（使用亮綠色，並放大標記）
output_img = img_rgb.copy()
for p in optimal_points:
    x, y = p['position']
    cv2.circle(output_img, (x, y), radius=10, color=(0, 255, 0), thickness=-1)  # 亮綠色 BGR
    cv2.putText(output_img, f"{p['score']:.2f}", (x + 10, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 儲存輸出
os.makedirs("output", exist_ok=True)
cv2.imwrite("output/optimal_placement.jpg", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
print("✅ 最佳置入點結果已儲存至：output/optimal_placement.jpg")

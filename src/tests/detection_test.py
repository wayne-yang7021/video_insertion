from modules.models.models import Model
from modules.detection.detection import ObjectDetector
from utils.get_video_first_frame import extract_first_frame
from utils.visualization.visualize_detection import visualize_detections

# 初始化模型
model = Model()
model.load_detectron()

# 建立偵測器
detector = ObjectDetector(model)

# 偵測圖像
image = extract_first_frame("src/data/videos/house_tour.mp4")
detections = detector.detect_objects_in_image(image)
visualize_detections(image, detections, output_path="./output/sample_detected.jpg")

for r in detections:
    print(f"{r['label']} ({r['score']:.2f}) → box: {r['box']}")

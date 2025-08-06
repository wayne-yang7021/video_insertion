# utils/visualization/depth_vis.py
import cv2
import numpy as np

def overlay_depth_map(image: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """將深度圖 overlay 在原始圖片上（彩色熱圖）"""
    heatmap = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return overlay

def visualize_depth_grayscale(depth_map: np.ndarray) -> np.ndarray:
    """以灰階方式顯示深度圖（0=黑=近，1=白=遠）"""
    norm_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    gray_depth = norm_depth.astype(np.uint8)
    return cv2.cvtColor(gray_depth, cv2.COLOR_GRAY2BGR)  # 為了可以跟原圖一起顯示

def draw_bbox_depth_labels(image: np.ndarray, bboxes: list, depths: dict) -> np.ndarray:
    """在圖片上標出每個 bbox 的深度值"""
    img = image.copy()
    for idx, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{depths.get(idx, '?'):.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return img

def visualize_depth_segmented(depth_map: np.ndarray, segments: int = 5) -> np.ndarray:
    """把深度圖分段顯示成不同顏色"""
    norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    step = 256 // segments
    segmented = (norm // step) * step
    color = cv2.applyColorMap(segmented, cv2.COLORMAP_JET)
    return color


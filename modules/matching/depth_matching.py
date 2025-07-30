# modules/matching/depth_matching.py
import numpy as np
from modules.models.matching_model import DepthEstimator

class DepthHandler:
    def __init__(self, model: DepthEstimator):
        self.model = model

    def estimate_depth_map(self, image_pil):
        """輸入 PIL 圖片，輸出 normalized depth map"""
        image_np = np.array(image_pil)  # PIL → np RGB
        depth_map = self.model.predict(image_np)
        return self._normalize_depth(depth_map)

    def get_bbox_depths(self, depth_map: np.ndarray, bboxes: list) -> dict:
        bbox_depths = {}
        for idx, (x1, y1, x2, y2) in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            region = depth_map[y1:y2, x1:x2]
            avg_depth = float(np.mean(region)) if region.size > 0 else 0.0
            bbox_depths[idx] = avg_depth
        return bbox_depths

    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        min_val, max_val = depth_map.min(), depth_map.max()
        if max_val - min_val < 1e-5:
            return np.zeros_like(depth_map)
        return (depth_map - min_val) / (max_val - min_val)

# models/matching_model.py
import sys
import os
midas_path = os.path.abspath("external/midas")  # 相對於主程式的路徑
if midas_path not in sys.path:
    sys.path.append(midas_path)
import cv2
import numpy as np
import torch
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

class DepthEstimator:
    def __init__(self, model_path="checkpoints/dpt_large_384.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True
        ).to(self.device).eval()

        self.transform = Compose([
            Resize(
                width=384, height=384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32, resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC
            ),
            NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            PrepareForNet()
        ])

    def predict(self, image: np.ndarray) -> np.ndarray:
        """輸入 image 為 numpy 格式 (HWC RGB)，回傳 normalized depth map"""
        input_tensor = self.transform({"image": image})["image"]
        sample = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model.forward(sample)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()
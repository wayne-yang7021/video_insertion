"""
Depth estimator using Hugging Face DPT model (hybrid MiDaS).
"""

import numpy as np
from PIL import Image
import cv2
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation


class DepthEstimator:
    """Depth estimator using Hugging Face DPT hybrid MiDaS model."""

    def __init__(self):
        self.model_loaded = False
        self.device = None
        self.model = None
        self.processor = None
        self._initialize()

    def _initialize(self):
        """Initialize the depth estimation model."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🔧 Using device: {self.device}")

            print("📥 Loading Hugging Face DPT model...")
            self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
            self.model = DPTForDepthEstimation.from_pretrained(
                "Intel/dpt-hybrid-midas",
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()

            self.model_loaded = True
            print(f"✅ DPT hybrid MiDaS model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load DPT model: {e}")
            raise e

    def estimate_depth(self, image):
        """Estimate depth map from an image."""
        if not self.model_loaded:
            raise RuntimeError("Depth estimator not initialized")

        try:
            print("🧠 Running Hugging Face DPT depth estimation...")

            # Ensure image is PIL
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # Prepare input
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original image size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],  # (height, width)
                mode="bicubic",
                align_corners=False,
            )

            # Convert to numpy
            depth_map = prediction.squeeze().cpu().numpy()

            # Normalize 0-1 and invert (closer = higher value)
            depth_min, depth_max = depth_map.min(), depth_map.max()
            if depth_max > depth_min:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
                depth_map = 1 - depth_map
            else:
                print("⚠️ Uniform depth values detected")
                depth_map[:] = 0.5

            print("✅ Depth estimation completed successfully")
            return depth_map

        except Exception as e:
            print(f"❌ Depth estimation failed: {e}")
            return None

    def visualize_depth(self, depth_map, save_path=None):
        """Create a colored visualization of the depth map."""
        if depth_map is None:
            return None

        try:
            # Convert to 8-bit
            depth_8bit = (depth_map * 255).astype(np.uint8)

            # Apply colormap
            depth_colored = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_PLASMA)
            depth_colored_rgb = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

            # Convert to PIL
            depth_image = Image.fromarray(depth_colored_rgb)

            if save_path:
                depth_image.save(save_path)
                print(f"💾 Depth visualization saved: {save_path}")

            return depth_image

        except Exception as e:
            print(f"❌ Error visualizing depth: {e}")
            return None

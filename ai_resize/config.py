"""
Configuration management for the object placement pipeline.
"""

import sys
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class Config:
    """Configuration manager for the pipeline."""
    
    def __init__(self, data_folder="data"):
        self.data_folder = Path(data_folder)
        self.output_folder = self.data_folder / "output"
        
        # Create directories if they don't exist
        self.data_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        
        # Model configurations
        self.yolo_model_size = "yolov8m"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
        self.sam_model_type = "vit_l"     # vit_b, vit_l, vit_h
        self.midas_model_type = "DPT_Large"  # MiDaS_small, MiDaS, DPT_Large, DPT_Hybrid
        
        # Detection thresholds
        self.yolo_confidence_threshold = 0.5
        self.sam_mask_threshold = 0.5
        
        # Image processing settings
        self.max_display_width = 1024
        self.max_display_height = 768
        
        # Gemini API settings
        self.gemini_model = "gemini-2.5-pro"
        self.gemini_max_retries = 3
        
    def load_api_key(self):
        """Load Gemini API key from file."""
        # api_key_path = self.data_folder / "api_key.txt"
        
        # if not api_key_path.exists():
        #     raise FileNotFoundError(
        #         f"API key file not found at {api_key_path}.\n"
        #         f"Please create this file with your Gemini API key."
        #     )
        
        # with open(api_key_path, 'r') as f:
        #     api_key = f.read().strip()
        
        # if not api_key:
        #     raise ValueError("API key file is empty")
        
        # print(f"✅ API key loaded from {api_key_path}")
        api_key = os.getenv("GEMINI_API_KEY")
        return api_key
    
    def get_image_paths(self):
        """Get expected image file paths."""
        # Environment image options
        env_patterns = ['env.jpg', 'env.png', 'env.jpeg', 
                       'environment.jpg', 'environment.png', 'environment.jpeg']
        
        # Object image options
        obj_patterns = ['obj.jpg', 'obj.png', 'obj.jpeg',
                       'object.jpg', 'object.png', 'object.jpeg']
        
        env_path = None
        obj_path = None
        
        # Find environment image
        for pattern in env_patterns:
            path = self.data_folder / pattern
            if path.exists():
                env_path = path
                break
        
        # Find object image
        for pattern in obj_patterns:
            path = self.data_folder / pattern
            if path.exists():
                obj_path = path
                break
        
        return env_path, obj_path
    
    def get_output_paths(self):
        """Get output file paths."""
        return {
            'composite': self.output_folder / "composite_result.png",
            'analysis': self.output_folder / "analysis_result.json",
            'detected_objects': self.output_folder / "detected_objects.json",
            'depth_map': self.output_folder / "depth_map.png",
            'depth_data': self.output_folder / "depth_data.npy",
            'resized_object': self.output_folder / "resized_object.png",
            'marked_environment': self.output_folder / "marked_environment.png",
            'coordinates': self.output_folder / "selected_coordinates.txt",
            'detection_visualization': self.output_folder / "detection_visualization.png"
        }
    
    @property
    def models_cache_dir(self):
        """Directory for caching downloaded models."""
        cache_dir = self.data_folder / "models_cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir
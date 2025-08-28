"""
Image loading and preprocessing module.
"""

import os
from PIL import Image
from pathlib import Path

from utils.get_video_first_frame import extract_first_frame
from .config import Config


class ImageLoader:
    """Handles loading and preprocessing of images."""
    
    def __init__(self, data_folder="data"):
        self.config = Config(data_folder)
    
    def load_images(self):
        """Load environment and object images from data folder."""
        env_path, obj_path = self.config.get_image_paths()
        
        # Check if files exist
        # if not env_path:
        #     env_patterns = ['env.jpg', 'env.png', 'environment.jpg', 'environment.png']
        #     raise FileNotFoundError(
        #         f"Environment image not found in {self.config.data_folder}.\n"
        #         f"Expected files: {env_patterns}"
        #     )
        
        # if not obj_path:
        #     obj_patterns = ['obj.jpg', 'obj.png', 'object.jpg', 'object.png']
        #     raise FileNotFoundError(
        #         f"Object image not found in {self.config.data_folder}.\n"
        #         f"Expected files: {obj_patterns}"
        #     )
        
        try:
            # Load images
            # environment_image = Image.open(env_path).convert('RGB')
            environment_image = Image.open("data/pictures/classroom.jpg").convert('RGB')
            # environment_image = extract_first_frame("data/videos/living_room.mp4").convert('RGB')
            # object_image = Image.open(obj_path)  # Keep original format for transparency
            object_image = Image.open("data/pictures/starbucks.png")
            
            # Validate images
            if environment_image.size[0] < 100 or environment_image.size[1] < 100:
                raise ValueError("Environment image too small (minimum 100x100)")
            
            if object_image.size[0] < 10 or object_image.size[1] < 10:
                raise ValueError("Object image too small (minimum 10x10)")
            
            print(f"✅ Environment image loaded: {env_path} ({environment_image.size})")
            print(f"✅ Object image loaded: {obj_path} ({object_image.size})")
            
            return environment_image, object_image
            
        except Exception as e:
            print(f"❌ Error loading images: {e}")
            return None, None
    
    def scale_image_for_display(self, image, max_width=None, max_height=None):
        """Scale image for display while maintaining aspect ratio."""
        if max_width is None:
            max_width = self.config.max_display_width
        if max_height is None:
            max_height = self.config.max_display_height
            
        original_width, original_height = image.size
        
        # Calculate scaling factor
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_ratio = min(width_ratio, height_ratio, 1.0)  # Don't upscale
        
        if scale_ratio < 1.0:
            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS), scale_ratio
        
        return image.copy(), 1.0
    
    def validate_image(self, image_path):
        """Validate if image file is valid and readable."""
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False
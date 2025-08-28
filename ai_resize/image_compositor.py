"""
Image composition module for creating final composite images.
"""
from PIL import Image
from .config import Config

class ImageCompositor:
    """Handles simple image composition."""
    
    def __init__(self):
        self.config = Config()
    
    def create_composite(self, environment_image, object_image, placement_point, scale_factor):
        """Create composite image with object placed in environment."""
        try:
            # Resize object according to scale factor
            resized_object = self._resize_object(object_image, scale_factor)
            if not resized_object:
                print("❌ Failed to resize object")
                return None
            
            # Create composite
            composite = environment_image.copy()
            
            # Calculate placement position (center object on point)
            obj_width, obj_height = resized_object.size
            paste_x = placement_point[0] - obj_width // 2
            paste_y = placement_point[1] - obj_height // 2
            
            # Ensure object stays within environment bounds
            env_width, env_height = environment_image.size
            paste_x = max(0, min(paste_x, env_width - obj_width))
            paste_y = max(0, min(paste_y, env_height - obj_height))
            
            # Paste object onto environment
            if resized_object.mode == 'RGBA':
                # Use alpha channel for transparency
                composite.paste(resized_object, (paste_x, paste_y), resized_object)
            else:
                # Simple paste without transparency
                composite.paste(resized_object, (paste_x, paste_y))
            
            print(f"✅ Composite created, object placed at ({paste_x}, {paste_y})")
            return composite
            
        except Exception as e:
            print(f"❌ Composition error: {e}")
            return None
    
    def _resize_object(self, object_image, scale_factor):
        """Resize object image with high quality resampling."""
        try:
            original_size = object_image.size
            new_size = (
                max(1, int(original_size[0] * scale_factor)),
                max(1, int(original_size[1] * scale_factor))
            )
            
            # Use high-quality resampling
            resized_object = object_image.resize(new_size, Image.Resampling.LANCZOS)
            
            print(f"🔄 Object resized from {original_size} to {new_size} (scale: {scale_factor:.3f})")
            return resized_object
            
        except Exception as e:
            print(f"❌ Resize error: {e}")
            return None
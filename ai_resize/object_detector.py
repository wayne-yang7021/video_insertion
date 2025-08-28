"""
Object detection module using YOLO and SAM.
"""

import numpy as np
from PIL import Image, ImageDraw
import cv2
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Install with: pip install ultralytics")

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ SAM not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

from .config import Config


class ObjectDetector:
    """Object detection using YOLO and SAM."""
    
    def __init__(self):
        self.config = Config()
        self.yolo_model = None
        self.sam_predictor = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize YOLO and SAM models."""
        print("🔄 Initializing object detection models...")
        
        # Initialize YOLO
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO(f'{self.config.yolo_model_size}.pt')
                print(f"✅ YOLO model ({self.config.yolo_model_size}) loaded")
            except Exception as e:
                print(f"⚠️ Failed to load YOLO: {e}")
                self.yolo_model = None
        
        # Initialize SAM
        if SAM_AVAILABLE:
            try:
                # Download SAM model if needed
                sam_checkpoint = self._download_sam_model()
                if sam_checkpoint:
                    sam = sam_model_registry[self.config.sam_model_type](checkpoint=sam_checkpoint)
                    self.sam_predictor = SamPredictor(sam)
                    print(f"✅ SAM model ({self.config.sam_model_type}) loaded")
            except Exception as e:
                print(f"⚠️ Failed to load SAM: {e}")
                self.sam_predictor = None
        
        if not self.yolo_model and not self.sam_predictor:
            print("⚠️ No object detection models available")
    
    def _download_sam_model(self):
        """Download SAM model checkpoint if needed."""
        import urllib.request
        from pathlib import Path
        
        model_urls = {
            'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
        }
        
        model_type = self.config.sam_model_type
        if model_type not in model_urls:
            print(f"⚠️ Unknown SAM model type: {model_type}")
            return None
        
        # Define checkpoint path
        checkpoint_name = model_urls[model_type].split('/')[-1]
        checkpoint_path = self.config.models_cache_dir / checkpoint_name
        
        # Download if not exists
        if not checkpoint_path.exists():
            print(f"⬇️ Downloading SAM checkpoint: {checkpoint_name}")
            try:
                urllib.request.urlretrieve(model_urls[model_type], checkpoint_path)
                print(f"✅ Downloaded: {checkpoint_path}")
            except Exception as e:
                print(f"❌ Download failed: {e}")
                return None
        
        return str(checkpoint_path)
    
    def detect_objects(self, image):
        """Detect objects in the image using YOLO and optionally SAM."""
        if isinstance(image, Image.Image):
            # Convert PIL to OpenCV format for YOLO
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = image
        
        detected_objects = []
        
        # YOLO detection
        if self.yolo_model:
            yolo_objects = self._yolo_detection(cv_image)
            detected_objects.extend(yolo_objects)
        
        # SAM segmentation (if available and YOLO found objects)
        if self.sam_predictor and detected_objects:
            detected_objects = self._enhance_with_sam(image, detected_objects)
        
        return detected_objects
    
    def _yolo_detection(self, cv_image):
        """Run YOLO object detection."""
        try:
            results = self.yolo_model(cv_image, conf=self.config.yolo_confidence_threshold)
            detected_objects = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0].item())
                        class_name = self.yolo_model.names[class_id]
                        
                        detected_objects.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'detection_method': 'YOLO',
                            'mask': None  # Will be filled by SAM if available
                        })
            
            print(f"🎯 YOLO detected {len(detected_objects)} objects")
            return detected_objects
            
        except Exception as e:
            print(f"❌ YOLO detection failed: {e}")
            return []
    
    def _enhance_with_sam(self, image, yolo_objects):
        """Enhance YOLO detections with SAM masks."""
        if not self.sam_predictor:
            return yolo_objects
        
        try:
            # Set image for SAM
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            self.sam_predictor.set_image(image_array)
            
            enhanced_objects = []
            for obj in yolo_objects:
                bbox = obj['bbox']
                
                # Use bounding box as prompt for SAM
                input_box = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
                
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[0],
                    multimask_output=False,
                )
                
                if len(masks) > 0 and len(scores) > 0:
                    # Take the best mask
                    best_mask = masks[0]
                    obj['mask'] = best_mask
                    obj['mask_score'] = scores[0]
                    obj['detection_method'] = 'YOLO+SAM'
                
                enhanced_objects.append(obj)
            
            print(f"🎭 SAM enhanced {len(enhanced_objects)} objects with masks")
            return enhanced_objects
            
        except Exception as e:
            print(f"❌ SAM enhancement failed: {e}")
            return yolo_objects
    
    def visualize_detections(self, image, detected_objects):
        """Create visualization of detected objects."""
        if isinstance(image, np.ndarray):
            vis_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            vis_image = image.copy()
        
        draw = ImageDraw.Draw(vis_image)
        
        for i, obj in enumerate(detected_objects):
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            # Draw bounding box
            draw.rectangle(bbox, outline='red', width=3)
            
            # Draw label
            label = f"{class_name} ({confidence:.2f})"
            
            # Simple text drawing (for better text rendering, use PIL with custom fonts)
            draw.text((bbox[0], bbox[1] - 20), label, fill='red')
        
        return vis_image
    
    def get_detection_summary(self, detected_objects):
        """Get summary of detected objects."""
        if not detected_objects:
            return "No objects detected"
        
        class_counts = {}
        for obj in detected_objects:
            class_name = obj['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary = f"Detected {len(detected_objects)} objects: "
        summary += ", ".join([f"{count}x {name}" for name, count in class_counts.items()])
        
        return summary
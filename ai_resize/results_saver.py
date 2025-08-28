"""
Results saving module for organizing and storing all pipeline outputs.
"""

import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from .config import Config


class ResultsSaver:
    """Handles saving all pipeline results to organized output folder."""
    
    def __init__(self, data_folder="data"):
        self.config = Config(data_folder)
        self.output_paths = self.config.get_output_paths()
        
        # Ensure output directory exists
        self.config.output_folder.mkdir(exist_ok=True)
    
    def save_all_results(self, analysis, composite_image, environment_image, 
                        object_image, detected_objects, depth_map, 
                        placement_point, scale_factor):
        """Save all pipeline results in organized manner."""
        print("💾 Saving all results...")
        
        saved_files = []
        
        # 1. Save analysis results
        analysis_file = self._save_analysis(analysis, placement_point, scale_factor)
        if analysis_file:
            saved_files.append(f"Analysis: {analysis_file}")
        
        # 2. Save composite image
        composite_file = self._save_composite_image(composite_image)
        if composite_file:
            saved_files.append(f"Composite: {composite_file}")
        
        # 3. Save detected objects info
        objects_file = self._save_detected_objects(detected_objects)
        if objects_file:
            saved_files.append(f"Objects: {objects_file}")
        
        # 4. Save depth map
        depth_files = self._save_depth_map(depth_map)
        saved_files.extend(depth_files)
        
        # 5. Save resized object
        object_file = self._save_resized_object(object_image, scale_factor)
        if object_file:
            saved_files.append(f"Resized object: {object_file}")
        
        # 6. Save marked environment
        marked_file = self._save_marked_environment(
            environment_image, detected_objects, placement_point
        )
        if marked_file:
            saved_files.append(f"Marked environment: {marked_file}")
        
        # 7. Save coordinates
        coords_file = self._save_coordinates(placement_point)
        if coords_file:
            saved_files.append(f"Coordinates: {coords_file}")
        
        # 8. Save detection visualization
        detection_file = self._save_detection_visualization(
            environment_image, detected_objects
        )
        if detection_file:
            saved_files.append(f"Detection viz: {detection_file}")
        
        # 9. Create summary report
        summary_file = self._create_summary_report(
            analysis, detected_objects, placement_point, scale_factor
        )
        if summary_file:
            saved_files.append(f"Summary: {summary_file}")
        
        print(f"✅ Saved {len(saved_files)} result files:")
        for file_info in saved_files:
            print(f"  • {file_info}")
        
        return saved_files
    
    def _save_analysis(self, analysis, placement_point, scale_factor):
        """Save complete analysis results."""
        try:
            analysis_data = {
                'pipeline_info': {
                    'version': '1.0',
                    'selected_coordinates': placement_point,
                    'final_scale_factor': scale_factor,
                    'timestamp': self._get_timestamp()
                },
                'analysis_results': analysis
            }
            
            with open(self.output_paths['analysis'], 'w') as f:
                json.dump(analysis_data, f, indent=2, default=self._json_serializer)
            
            return self.output_paths['analysis']
            
        except Exception as e:
            print(f"❌ Error saving analysis: {e}")
            return None
    
    def _save_composite_image(self, composite_image):
        """Save the final composite image."""
        try:
            if composite_image:
                composite_image.save(self.output_paths['composite'])
                return self.output_paths['composite']
            return None
            
        except Exception as e:
            print(f"❌ Error saving composite: {e}")
            return None
    
    def _save_detected_objects(self, detected_objects):
        """Save detected objects information."""
        try:
            objects_data = {
                'detection_summary': {
                    'total_objects': len(detected_objects) if detected_objects else 0,
                    'timestamp': self._get_timestamp()
                },
                'detected_objects': detected_objects or []
            }
            
            with open(self.output_paths['detected_objects'], 'w') as f:
                json.dump(objects_data, f, indent=2, default=self._json_serializer)
            
            return self.output_paths['detected_objects']
            
        except Exception as e:
            print(f"❌ Error saving detected objects: {e}")
            return None
    
    def _save_depth_map(self, depth_map):
        """Save depth map data and visualization."""
        saved_files = []
        
        if depth_map is None:
            return saved_files
        
        try:
            # Save depth visualization (PNG)
            if 'normalized_depth' in depth_map:
                depth_vis = self._create_depth_visualization(depth_map)
                if depth_vis:
                    depth_vis.save(self.output_paths['depth_map'])
                    saved_files.append(f"Depth map: {self.output_paths['depth_map']}")
            
            # Save raw depth data (NPY)
            if 'raw_depth' in depth_map:
                np.save(self.output_paths['depth_data'], depth_map['raw_depth'])
                saved_files.append(f"Depth data: {self.output_paths['depth_data']}")
                
        except Exception as e:
            print(f"❌ Error saving depth map: {e}")
        
        return saved_files
    
    def _save_resized_object(self, object_image, scale_factor):
        """Save the resized object image."""
        try:
            if object_image and scale_factor:
                from .image_compositor import ImageCompositor
                compositor = ImageCompositor()
                resized_object = compositor._resize_object(object_image, scale_factor)
                
                if resized_object:
                    resized_object.save(self.output_paths['resized_object'])
                    return self.output_paths['resized_object']
            
            return None
            
        except Exception as e:
            print(f"❌ Error saving resized object: {e}")
            return None
    
    def _save_marked_environment(self, environment_image, detected_objects, placement_point):
        """Save environment image with markings."""
        try:
            marked_image = self._create_marked_environment(
                environment_image, detected_objects, placement_point
            )
            
            if marked_image:
                marked_image.save(self.output_paths['marked_environment'])
                return self.output_paths['marked_environment']
            
            return None
            
        except Exception as e:
            print(f"❌ Error saving marked environment: {e}")
            return None
    
    def _save_coordinates(self, placement_point):
        """Save selected coordinates."""
        try:
            coords_text = f"{placement_point[0]},{placement_point[1]}\n"
            coords_text += f"Selected placement coordinates: ({placement_point[0]}, {placement_point[1]})"
            
            with open(self.output_paths['coordinates'], 'w') as f:
                f.write(coords_text)
            
            return self.output_paths['coordinates']
            
        except Exception as e:
            print(f"❌ Error saving coordinates: {e}")
            return None
    
    def _save_detection_visualization(self, environment_image, detected_objects):
        """Save object detection visualization."""
        try:
            if not detected_objects:
                return None
            
            from .object_detector import ObjectDetector
            detector = ObjectDetector()
            detection_vis = detector.visualize_detections(environment_image, detected_objects)
            
            if detection_vis:
                detection_vis.save(self.output_paths['detection_visualization'])
                return self.output_paths['detection_visualization']
            
            return None
            
        except Exception as e:
            print(f"❌ Error saving detection visualization: {e}")
            return None
    
    def _create_marked_environment(self, environment_image, detected_objects, placement_point):
        """Create environment image with all markings."""
        try:
            marked_image = environment_image.copy()
            draw = ImageDraw.Draw(marked_image)
            
            # Draw detected objects
            colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
            
            for i, obj in enumerate(detected_objects or []):
                bbox = obj['bbox']
                class_name = obj['class_name']
                confidence = obj['confidence']
                
                color = colors[i % len(colors)]
                
                # Draw bounding box
                draw.rectangle(bbox, outline=color, width=2)
                
                # Draw label
                label = f"{class_name} ({confidence:.2f})"
                
                # Draw label background
                text_width = len(label) * 7
                text_height = 15
                label_bg = [bbox[0], bbox[1] - text_height - 4, 
                           bbox[0] + text_width, bbox[1]]
                draw.rectangle(label_bg, fill=color)
                
                # Draw text
                draw.text((bbox[0] + 2, bbox[1] - text_height - 2), label, fill='white')
            
            # Draw placement point
            radius = 12
            draw.ellipse([
                placement_point[0] - radius, placement_point[1] - radius,
                placement_point[0] + radius, placement_point[1] + radius
            ], fill='red', outline='darkred', width=3)
            
            # Add crosshair
            cross_size = 20
            draw.line([
                placement_point[0] - cross_size, placement_point[1],
                placement_point[0] + cross_size, placement_point[1]
            ], fill='red', width=3)
            draw.line([
                placement_point[0], placement_point[1] - cross_size,
                placement_point[0], placement_point[1] + cross_size
            ], fill='red', width=3)
            
            # Add placement label
            draw.text(
                (placement_point[0] + 15, placement_point[1] - 15),
                f"Placement: ({placement_point[0]}, {placement_point[1]})",
                fill='red'
            )
            
            return marked_image
            
        except Exception as e:
            print(f"❌ Error creating marked environment: {e}")
            return None
    
    def _create_depth_visualization(self, depth_map):
        """Create colorized depth map visualization."""
        try:
            from .depth_estimator import DepthEstimator
            estimator = DepthEstimator()
            return estimator.visualize_depth(depth_map)
            
        except Exception as e:
            print(f"❌ Error creating depth visualization: {e}")
            return None
    
    def _create_summary_report(self, analysis, detected_objects, placement_point, scale_factor):
        """Create a human-readable summary report."""
        try:
            report_path = self.config.output_folder / "summary_report.txt"
            
            with open(report_path, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("OBJECT PLACEMENT PIPELINE SUMMARY REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Generated: {self._get_timestamp()}\n\n")
                
                # Placement info
                f.write("PLACEMENT INFORMATION:\n")
                f.write(f"  • Selected coordinates: ({placement_point[0]}, {placement_point[1]})\n")
                f.write(f"  • Final scale factor: {scale_factor:.3f}\n\n")
                
                # Object detection summary
                if detected_objects:
                    f.write("DETECTED OBJECTS:\n")
                    f.write(f"  • Total objects detected: {len(detected_objects)}\n")
                    
                    object_counts = {}
                    for obj in detected_objects:
                        class_name = obj['class_name']
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1
                    
                    for obj_class, count in object_counts.items():
                        f.write(f"  • {obj_class}: {count}\n")
                    f.write("\n")
                
                # Analysis summary
                if analysis:
                    f.write("ANALYSIS SUMMARY:\n")
                    
                    # Environment
                    env = analysis.get('environment', {})
                    if env:
                        f.write("  Environment Analysis:\n")
                        f.write(f"    - Type: {env.get('environment_type', 'Unknown')}\n")
                        f.write(f"    - Surface: {env.get('surface_type', 'Unknown')}\n")
                        f.write(f"    - Perspective: {env.get('perspective', 'Unknown')}\n")
                    
                    # Object
                    obj = analysis.get('object', {})
                    if obj:
                        f.write("  Object Analysis:\n")
                        f.write(f"    - Name: {obj.get('object_name', 'Unknown')}\n")
                        f.write(f"    - Category: {obj.get('object_category', 'Unknown')}\n")
                        f.write(f"    - Size category: {obj.get('size_category', 'Unknown')}\n")
                    
                    # Placement evaluation
                    place = analysis.get('placement', {})
                    if place:
                        f.write("  Placement Evaluation:\n")
                        f.write(f"    - Overall suitable: {'Yes' if place.get('overall_suitable') else 'No'}\n")
                        f.write(f"    - Logical fit: {'Yes' if place.get('logical_fit') else 'No'}\n")
                        f.write(f"    - Surface appropriate: {'Yes' if place.get('surface_appropriate') else 'No'}\n")
                        
                        if place.get('concerns'):
                            f.write(f"    - Concerns: {place.get('concerns')}\n")
                    
                    # Sizing
                    sizing = analysis.get('sizing', {})
                    if sizing:
                        f.write("  Size Calculation:\n")
                        f.write(f"    - Recommended scale: {sizing.get('recommended_scale', 'Unknown')}\n")
                        f.write(f"    - Confidence: {sizing.get('confidence', 0):.1%}\n")
                        f.write(f"    - Reasoning: {sizing.get('size_reasoning', 'Not provided')}\n")
                    
                    # Overall confidence
                    confidence = analysis.get('overall_confidence', 0)
                    f.write(f"\n  Overall Confidence: {confidence:.1%}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("OUTPUT FILES:\n")
                f.write("  • composite_result.png - Final composite image\n")
                f.write("  • analysis_result.json - Complete analysis data\n")
                f.write("  • detected_objects.json - Object detection results\n")
                f.write("  • depth_map.png - Depth visualization\n")
                f.write("  • resized_object.png - Scaled object image\n")
                f.write("  • marked_environment.png - Environment with annotations\n")
                f.write("  • And more...\n")
                f.write("=" * 60 + "\n")
            
            return report_path
            
        except Exception as e:
            print(f"❌ Error creating summary report: {e}")
            return None
    
    def _get_timestamp(self):
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return str(obj)
    
    def create_backup(self):
        """Create backup of current results."""
        try:
            from datetime import datetime
            backup_folder = self.config.output_folder / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_folder.mkdir(exist_ok=True)
            
            # Copy all files from output folder to backup
            import shutil
            for file_path in self.config.output_folder.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, backup_folder / file_path.name)
            
            print(f"✅ Backup created: {backup_folder}")
            return backup_folder
            
        except Exception as e:
            print(f"❌ Backup creation failed: {e}")
            return None
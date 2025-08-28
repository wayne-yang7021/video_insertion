"""
Gemini AI analysis module for intelligent object placement and sizing.
"""

import base64
import json
from io import BytesIO
from PIL import Image, ImageDraw
import google.generativeai as genai
from .config import Config
import numpy as np


class GeminiAnalyzer:
    """Gemini AI analyzer for object placement and sizing."""
    
    def __init__(self, api_key):
        self.config = Config()
        self.api_key = api_key
        self.model = None
        self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini API."""
        if not self.api_key:
            raise ValueError("API key is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.config.gemini_model)
        print(f"✅ Gemini API initialized with model: {self.config.gemini_model}")
    
    def analyze_placement(self, environment_image, object_image, detected_objects, 
                         depth_map, placement_point):
        """Comprehensive placement analysis using detected objects and depth info."""
        print("🧠 Starting comprehensive Gemini analysis...")
        
        # Enhance detected objects with depth information
        enhanced_objects = self._enhance_detected_objects_with_depth(detected_objects, depth_map)
        
        # Single comprehensive analysis
        comprehensive_analysis = self._analyze_placement_and_sizing(
            environment_image, object_image, enhanced_objects, 
            depth_map, placement_point
        )
        
        if not comprehensive_analysis:
            return None
        
        # Extract components and add summary information
        combined_result = {
            'environment': comprehensive_analysis.get('environment_analysis', {}),
            'object': comprehensive_analysis.get('object_analysis', {}),
            'placement': comprehensive_analysis.get('placement_evaluation', {}),
            'sizing': comprehensive_analysis.get('sizing_analysis', {}),
            'detected_objects_info': self._summarize_detected_objects(enhanced_objects),
            'depth_info': self._get_placement_depth_info(depth_map, placement_point),
            'recommended_scale_factor': comprehensive_analysis.get('sizing_analysis', {}).get('recommended_scale', 1.0),
            'overall_confidence': comprehensive_analysis.get('overall_confidence', 0.5)
        }
        
        print("✅ Comprehensive Gemini analysis completed!")
        return combined_result
    
    def _enhance_detected_objects_with_depth(self, detected_objects, depth_map):
        """Add depth information to each detected object."""
        if not detected_objects or depth_map is None:
            return detected_objects
        
        enhanced_objects = []
        
        for obj in detected_objects:
            enhanced_obj = obj.copy()
            bbox = obj['bbox']
            
            # Calculate center of bounding box
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # Get depth at object center
            try:
                depth_info = self._get_depth_at_point(depth_map, center_x, center_y)
                enhanced_obj['depth_info'] = depth_info
                
            except Exception as e:
                print(f"⚠️ Could not get depth for {obj['class_name']}: {e}")
                enhanced_obj['depth_info'] = None
            
            enhanced_objects.append(enhanced_obj)
        
        return enhanced_objects
    
    def _get_depth_at_point(self, depth_map, x, y):
        """Get normalized depth information at a specific point."""
        try:
            h, w = depth_map.shape
            x = max(0, min(int(x), w-1))
            y = max(0, min(int(y), h-1))
            
            raw_depth = float(depth_map[y, x])
            
            # Normalize depth relative to the entire image
            valid_depths = depth_map[depth_map > 0]
            if len(valid_depths) > 0:
                min_depth = float(np.min(valid_depths))
                max_depth = float(np.max(valid_depths))
                
                if max_depth > min_depth:
                    relative_depth = (raw_depth - min_depth) / (max_depth - min_depth)
                else:
                    relative_depth = 0.5
            else:
                relative_depth = 0.5
            
            # Categorize depth
            if relative_depth < 0.33:
                depth_category = "foreground"
            elif relative_depth < 0.67:
                depth_category = "midground"  
            else:
                depth_category = "background"
            
            return {
                'raw_depth': raw_depth,
                'relative_depth': relative_depth,
                'depth_category': depth_category
            }
            
        except Exception as e:
            print(f"Error getting depth at point ({x}, {y}): {e}")
            return None
    
    def _analyze_placement_and_sizing(self, environment_image, object_image, detected_objects, 
                                    depth_map, placement_point):
        """Single comprehensive analysis for placement and sizing."""
        
        # Create marked environment image
        marked_env = self._create_marked_environment_image(
            environment_image, detected_objects, placement_point
        )
        
        env_b64 = self._encode_image_to_base64(marked_env)
        obj_b64 = self._encode_image_to_base64(object_image)
        
        # Get placement point depth
        placement_depth = self._get_depth_at_point(depth_map, placement_point[0], placement_point[1])
        
        # Create enhanced context
        depth_context = self._create_enhanced_depth_context(detected_objects, placement_point, placement_depth)
        reference_context = self._create_size_reference_context(detected_objects, placement_point, placement_depth)
        
        prompt = f"""
        Perform comprehensive analysis for intelligent object placement and sizing.
        
        IMAGES PROVIDED:
        1. Environment image with detected objects (colored boxes) and placement point (red circle)
        2. Object image to be placed
        
        DEPTH AND REFERENCE CONTEXT:
        {depth_context}
        
        {reference_context}
        
        ANALYSIS TASKS:
        
        1. ENVIRONMENT ANALYSIS:
           - What type of environment/scene is this?
           - What surface is at the placement point?
           - Camera perspective and viewing angle?
           - Spatial relationships between placement and nearby objects?
        
        2. OBJECT ANALYSIS:
           - What specific object is this?
           - Typical real-world dimensions?
           - Object category and size classification?
           - Viewing angle and orientation?
        
        3. PLACEMENT EVALUATION:
           - Does this object logically belong in this environment?
           - Is the placement point appropriate?
           - Any physics or stability concerns?
           - Contextual consistency?
        
        4. SIZING ANALYSIS (MOST IMPORTANT):
           - Use nearby detected objects as size references
           - Account for depth relationships (objects farther appear smaller)
           - Consider perspective effects
           - Calculate realistic proportions
           - Determine optimal scale factor (1.0 = original size, <1.0 = smaller, >1.0 = larger)
        
        Respond in JSON format:
        {{
            "environment_analysis": {{
                "environment_type": "specific description",
                "surface_type": "what surface the placement point is on",
                "perspective": "camera angle description",
                "depth_zone": "foreground/midground/background",
                "nearby_objects": ["list", "of", "nearby", "objects"],
                "spatial_relationships": "how placement relates to nearby objects"
            }},
            "object_analysis": {{
                "object_name": "specific object identification",
                "object_category": "general category",
                "typical_dimensions_cm": {{"length": 0, "width": 0, "height": 0}},
                "size_category": "tiny/small/medium/large/huge",
                "photo_angle": "viewing angle description"
            }},
            "placement_evaluation": {{
                "logical_fit": true,
                "surface_appropriate": true,
                "context_match": true,
                "physics_stability": true,
                "overall_suitable": true,
                "concerns": "any issues identified",
                "confidence_score": 0.85
            }},
            "sizing_analysis": {{
                "depth_analysis": "how depth information guides sizing",
                "reference_objects_used": ["objects", "used", "for", "size", "reference"],
                "depth_perspective_factor": "how depth affects apparent size",
                "recommended_scale": 0.75,
                "reasoning": "detailed explanation for scale decision",
                "confidence": 0.8,
                "size_category_in_scene": "relative size description"
            }},
            "overall_confidence": 0.8
        }}
        """
        
        return self._query_gemini(prompt, env_b64, obj_b64)
    
    def _create_enhanced_depth_context(self, detected_objects, placement_point, placement_depth):
        """Create comprehensive depth context for analysis."""
        context = []
        
        # Placement point depth analysis
        if placement_depth:
            context.append("PLACEMENT POINT DEPTH:")
            context.append(f"- Depth category: {placement_depth['depth_category']}")
            context.append(f"- Relative depth: {placement_depth['relative_depth']:.3f} (0.0=closest, 1.0=farthest)")
            context.append(f"- Raw depth value: {placement_depth['raw_depth']:.3f}")
        else:
            context.append("PLACEMENT POINT DEPTH: Not available")
        
        # Nearby objects depth comparison
        nearby_objects = self._get_nearby_objects_info(detected_objects, placement_point, distance_threshold=200)
        
        if nearby_objects:
            context.append("\nNEARBY OBJECTS DEPTH COMPARISON:")
            for obj in nearby_objects:
                if 'depth_info' in obj and obj['depth_info']:
                    depth_info = obj['depth_info']
                    distance = obj['distance_to_placement']
                    
                    # Calculate depth difference
                    if placement_depth:
                        depth_diff = depth_info['relative_depth'] - placement_depth['relative_depth']
                        if depth_diff > 0.1:
                            depth_relationship = "significantly behind placement"
                        elif depth_diff < -0.1:
                            depth_relationship = "significantly in front of placement"
                        else:
                            depth_relationship = "at similar depth to placement"
                    else:
                        depth_relationship = "unknown relationship"
                    
                    context.append(f"- {obj['class_name']} (~{distance:.0f}px away):")
                    context.append(f"  * Depth category: {depth_info['depth_category']}")
                    context.append(f"  * Relative depth: {depth_info['relative_depth']:.3f}")
                    context.append(f"  * Relationship: {depth_relationship}")
        else:
            context.append("\nNo nearby objects with depth information available.")
        
        return "\n".join(context)
    
    def _create_size_reference_context(self, detected_objects, placement_point, placement_depth):
        """Create context specifically for size estimation."""
        context = []
        
        # Find objects that can serve as size references
        nearby_objects = self._get_nearby_objects_info(detected_objects, placement_point, distance_threshold=300)
        reference_objects = []
        
        for obj in nearby_objects:
            if 'depth_info' in obj and obj['depth_info'] and placement_depth:
                obj_depth = obj['depth_info']['relative_depth']
                placement_rel_depth = placement_depth['relative_depth']
                
                # Calculate depth ratio
                depth_ratio = obj_depth / placement_rel_depth if placement_rel_depth > 0 else 1.0
                
                # Get typical size for this object type
                typical_size = self._get_typical_object_size(obj['class_name'])
                
                if typical_size:
                    reference_objects.append({
                        'object': obj,
                        'depth_ratio': depth_ratio,
                        'typical_size': typical_size
                    })
        
        if reference_objects:
            context.append("SIZE REFERENCE ANALYSIS:")
            context.append("Objects that can help determine appropriate scale:")
            
            for ref in reference_objects[:5]:  # Limit to top 5 references
                obj = ref['object']
                context.append(f"\n- {obj['class_name']} (confidence: {obj['confidence']:.2f}):")
                context.append(f"  * Typical size: {ref['typical_size']}")
                context.append(f"  * Distance from placement: {obj['distance_to_placement']:.0f}px")
                context.append(f"  * Depth ratio vs placement: {ref['depth_ratio']:.2f}")
                
                if ref['depth_ratio'] > 1.2:
                    context.append(f"  * Much farther than placement → appears smaller than real size")
                elif ref['depth_ratio'] < 0.8:
                    context.append(f"  * Much closer than placement → appears larger than real size") 
                else:
                    context.append(f"  * Similar depth to placement → good size reference")
        else:
            context.append("SIZE REFERENCE ANALYSIS:")
            context.append("No suitable size reference objects found nearby.")
        
        return "\n".join(context)
    
    def _create_marked_environment_image(self, environment_image, detected_objects, placement_point):
        """Create environment image with detected objects and placement point marked."""
        marked_image = environment_image.copy()
        draw = ImageDraw.Draw(marked_image)
        
        # Draw detected objects with different colors
        colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
        
        for i, obj in enumerate(detected_objects or []):
            bbox = obj['bbox']
            class_name = obj['class_name']
            confidence = obj['confidence']
            
            color = colors[i % len(colors)]
            
            # Draw bounding box
            draw.rectangle(bbox, outline=color, width=3)
            
            # Create enhanced label with depth info
            label = f"{class_name} ({confidence:.2f})"
            if 'depth_info' in obj and obj['depth_info']:
                depth_cat = obj['depth_info']['depth_category']
                label += f" [{depth_cat}]"
            
            # Calculate text position
            text_x = bbox[0]
            text_y = bbox[1] - 30
            
            # Draw text background
            text_width = len(label) * 8
            text_height = 25
            draw.rectangle(
                [text_x, text_y, text_x + text_width, text_y + text_height],
                fill=color
            )
            
            # Draw text
            draw.text((text_x + 2, text_y + 2), label, fill='white')
        
        # Draw placement point as red circle
        radius = 15
        draw.ellipse([
            placement_point[0] - radius, placement_point[1] - radius,
            placement_point[0] + radius, placement_point[1] + radius
        ], fill='red', outline='darkred', width=3)
        
        # Add crosshair at placement point
        cross_size = 25
        draw.line([
            placement_point[0] - cross_size, placement_point[1],
            placement_point[0] + cross_size, placement_point[1]
        ], fill='red', width=3)
        draw.line([
            placement_point[0], placement_point[1] - cross_size,
            placement_point[0], placement_point[1] + cross_size
        ], fill='red', width=3)
        
        return marked_image
    
    def _get_typical_object_size(self, class_name):
        """Get typical real-world size for common objects."""
        size_database = {
            'person': '160-180cm tall',
            'chair': '45cm seat height, 80cm total height',
            'table': '75cm height, varies in width',
            'sofa': '80cm height, 180-220cm width',
            'bed': '50cm height, 200cm length',
            'tv': '50-80cm diagonal',
            'laptop': '30-35cm width',
            'book': '20-25cm height',
            'cup': '8-10cm height',
            'bottle': '20-25cm height',
            'phone': '15cm height',
            'car': '4-5m length, 1.8m height',
            'bicycle': '180cm length, 110cm height',
            'door': '200cm height, 80cm width',
            'window': 'varies, typically 120-150cm height'
        }
        
        return size_database.get(class_name.lower())
    
    def _get_nearby_objects_info(self, detected_objects, placement_point, distance_threshold=150):
        """Get information about objects near the placement point."""
        nearby_objects = []
        
        for obj in detected_objects:
            bbox = obj['bbox']
            
            # Ensure bbox is a list of floats
            if isinstance(bbox, np.ndarray):
                bbox = bbox.tolist()
            else:
                bbox = [float(v) for v in bbox]
            
            # Calculate center of bounding box
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Calculate distance to placement point
            distance = ((center_x - placement_point[0])**2 + (center_y - placement_point[1])**2)**0.5
            
            if distance <= distance_threshold:
                obj_info = {}
                for k, v in obj.items():
                    # Convert any numpy array/scalar to Python types
                    if isinstance(v, np.ndarray):
                        obj_info[k] = v.tolist()
                    elif isinstance(v, (np.generic,)):
                        obj_info[k] = v.item()
                    else:
                        obj_info[k] = v
                obj_info['distance_to_placement'] = float(distance)
                obj_info['center_point'] = (float(center_x), float(center_y))
                nearby_objects.append(obj_info)
        
        # Sort by distance
        nearby_objects.sort(key=lambda x: x['distance_to_placement'])
        
        return nearby_objects
    
    def _summarize_detected_objects(self, detected_objects):
        """Create summary of all detected objects."""
        if not detected_objects:
            return {"count": 0, "objects": []}
        
        summary = {
            "count": len(detected_objects),
            "objects": [],
            "categories": {}
        }
        
        for obj in detected_objects:
            obj_summary = {
                "class_name": obj['class_name'],
                "confidence": obj['confidence'],
                "bbox": obj['bbox'],
                "detection_method": obj.get('detection_method', 'unknown')
            }
            
            if 'depth_info' in obj and obj['depth_info']:
                obj_summary['depth_category'] = obj['depth_info'].get('depth_category')
                obj_summary['relative_depth'] = obj['depth_info'].get('relative_depth')
            
            summary["objects"].append(obj_summary)
            
            # Count categories
            category = obj['class_name']
            summary["categories"][category] = summary["categories"].get(category, 0) + 1
        
        return summary
    
    def _get_placement_depth_info(self, depth_map, placement_point):
        """Get depth information for the placement point."""
        if depth_map is None:
            return {"available": False}
        
        try:
            depth_info = self._get_depth_at_point(depth_map, placement_point[0], placement_point[1])
            
            if depth_info:
                return {
                    "available": True,
                    "depth_category": depth_info['depth_category'],
                    "relative_depth": depth_info['relative_depth'],
                    "raw_depth": depth_info['raw_depth']
                }
        except Exception as e:
            print(f"⚠️ Error getting placement depth info: {e}")
        
        return {"available": False}
    
    def _encode_image_to_base64(self, image):
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _query_gemini(self, prompt, *images):
        """Query Gemini with prompt and optional images."""
        for attempt in range(self.config.gemini_max_retries):
            try:
                content = [prompt]
                
                # Add images if provided
                for image_b64 in images:
                    content.append({
                        "mime_type": "image/png",
                        "data": image_b64
                    })
                
                response = self.model.generate_content(content)
                response_text = response.text
                
                # Extract JSON from response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    print(f"⚠️ No JSON found in response: {response_text}")
                    return None
                
            except Exception as e:
                print(f"❌ Gemini query attempt {attempt + 1} failed: {e}")
                if attempt == self.config.gemini_max_retries - 1:
                    print(f"❌ All {self.config.gemini_max_retries} attempts failed")
                    return None
        
        return None
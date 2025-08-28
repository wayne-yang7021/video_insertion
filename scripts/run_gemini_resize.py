#!/usr/bin/env python3
"""
Intelligent Object Placement and Resizing Pipeline
Main entry point that orchestrates the entire pipeline.
"""

from pathlib import Path
import sys
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from ai_resize.config import Config
from ai_resize.image_loader import ImageLoader
from ai_resize.object_detector import ObjectDetector
from ai_resize.depth_estimator import DepthEstimator
from ai_resize.point_selector import PointSelector
from ai_resize.gemini_analyzer import GeminiAnalyzer
from ai_resize.image_compositor import ImageCompositor
from ai_resize.results_saver import ResultsSaver
from modules.validation.vlm.vlm import get_best_placement_point, denormalize_point
from utils.get_video_first_frame import extract_first_frame


class ObjectPlacementPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, data_folder="data"):
        self.config = Config(data_folder)
        self.image_loader = ImageLoader(data_folder)
        self.object_detector = ObjectDetector()
        self.depth_estimator = DepthEstimator()
        self.point_selector = None
        self.gemini_analyzer = None
        self.compositor = ImageCompositor()
        self.results_saver = ResultsSaver(data_folder)
        
        # Pipeline state
        self.environment_image = None
        self.object_image = None
        self.detected_objects = None
        self.depth_map = None
        self.selected_point = None
        self.analysis_result = None
    
    def initialize(self):
        """Initialize all components."""
        print("🔧 Initializing pipeline components...")
         
        # Load API key and initialize Gemini
        api_key = self.config.load_api_key()
        self.gemini_analyzer = GeminiAnalyzer(api_key)
        
        print("✅ Pipeline initialized successfully!")
        return True
    
    def load_images(self):
        """Load environment and object images."""
        print("📁 Loading images...")
        
        self.environment_image, self.object_image = self.image_loader.load_images()
        if not self.environment_image or not self.object_image:
            print("❌ Failed to load required images")
            return False
            
        print("✅ Images loaded successfully!")
        return True
    
    def detect_objects(self):
        """Detect objects in the environment image."""
        print("🔍 Detecting objects in environment...")
        
        self.detected_objects = self.object_detector.detect_objects(self.environment_image)
        
        if not self.detected_objects:
            print("⚠️ No objects detected in environment")
            return False
        
        print(f"✅ Detected {len(self.detected_objects)} objects!")
        
        # Print detected objects summary
        for i, obj in enumerate(self.detected_objects):
            print(f"   {i+1}. {obj['class_name']} (confidence: {obj['confidence']:.2f})")
        
        return True
    
    def estimate_depth(self):
        """Estimate depth map for the environment."""
        print("📏 Estimating depth map...")
        
        self.depth_map = self.depth_estimator.estimate_depth(self.environment_image)
        
        if self.depth_map is None:
            print("❌ Failed to generate depth map")
            return False
            
        print("✅ Depth map generated successfully!")
        return True
    
    def select_placement_point(self):
        """Interactive point selection."""
        print("📍 Opening point selection interface...")
        
        # self.point_selector = PointSelector(
        #     self.environment_image,
        #     self.detected_objects,
        #     self.depth_map
        # )
        
        # self.selected_point = self.point_selector.select_point()
        
        product_path = "data/pictures/starbucks.png"         # 要置入的產品照
        background_path = "data/pictures/classroom.jpg"           # 背景照（你原本程式就讀了這張）
        # background_video = "data/videos/living_room.mp4"
        cx, cy = get_best_placement_point(
            product_image_path = product_path,
            background_image_path = background_path,
            user_prompt="",  # 可填你的客製需求
            out_dir="./output",
        )
        import cv2
        img = cv2.imread(background_path)
        if img is None:
            raise FileNotFoundError(background_path)
        H, W = img.shape[:2]
        
        # img = extract_first_frame("data/videos/living_room.mp4")
        # W, H = img.size

        self.selected_point = denormalize_point(cx, cy, W, H)
        
        if not self.selected_point:
            print("❌ No point selected")
            return False
            
        print(f"✅ Point selected: {self.selected_point}")
        return True
    
    def analyze_with_gemini(self):
        """Use Gemini to analyze placement and determine optimal size."""
        print("🧠 Analyzing with Gemini AI...")
        
        self.analysis_result = self.gemini_analyzer.analyze_placement(
            environment_image=self.environment_image,
            object_image=self.object_image,
            detected_objects=self.detected_objects,
            depth_map=self.depth_map,
            placement_point=self.selected_point
        )
        
        if not self.analysis_result:
            print("❌ Gemini analysis failed")
            return False
            
        print("✅ Gemini analysis completed!")
        self._print_analysis_summary()
        return True
    
    def create_composite(self):
        """Create final composite image."""
        print("🎨 Creating composite image...")
        
        scale_factor = self.analysis_result.get('recommended_scale_factor', 1.0)
        
        composite_image = self.compositor.create_composite(
            environment_image=self.environment_image,
            object_image=self.object_image,
            placement_point=self.selected_point,
            scale_factor=scale_factor
        )
        
        if not composite_image:
            print("❌ Failed to create composite")
            return False
            
        print("✅ Composite image created!")
        
        # Save all results
        self.results_saver.save_all_results(
            analysis=self.analysis_result,
            composite_image=composite_image,
            environment_image=self.environment_image,
            object_image=self.object_image,
            detected_objects=self.detected_objects,
            depth_map=self.depth_map,
            placement_point=self.selected_point,
            scale_factor=scale_factor
        )
        
        return True
    
    def _print_analysis_summary(self):
        """Print a summary of the Gemini analysis."""
        if not self.analysis_result:
            return
            
        print("\n" + "="*50)
        print("📋 ANALYSIS SUMMARY")
        print("="*50)
        
        # Environment analysis
        env = self.analysis_result.get('environment', {})
        print("🏞️  ENVIRONMENT:")
        print(f"   Type: {env.get('environment_type', 'Unknown')}")
        print(f"   Surface: {env.get('surface_type', 'Unknown')}")
        print(f"   Context: {env.get('scene_context', 'Unknown')}")
        
        # Object analysis
        obj = self.analysis_result.get('object', {})
        print(f"\n🎯 OBJECT:")
        print(f"   Name: {obj.get('object_name', 'Unknown')}")
        print(f"   Category: {obj.get('size_category', 'Unknown')}")
        
        # Placement evaluation
        place = self.analysis_result.get('placement', {})
        print(f"\n📍 PLACEMENT:")
        print(f"   Suitable: {'✅' if place.get('overall_suitable') else '❌'}")
        print(f"   Logic fit: {'✅' if place.get('logical_fit') else '❌'}")
        
        # Size recommendation
        scale = self.analysis_result.get('recommended_scale_factor', 1.0)
        confidence = self.analysis_result.get('overall_confidence', 0.0)
        print(f"\n📏 SIZING:")
        print(f"   Scale factor: {scale:.3f}")
        print(f"   Confidence: {confidence:.1%}")
        
        print("="*50)
    
    def run_pipeline(self):
        """Run the complete pipeline."""  
        print("🚀 Starting Object Placement Pipeline")
        print("="*50)
        
        # pre-select points
        
        steps = [
            ("Initialize", self.initialize),
            ("Load Images", self.load_images),
            ("Detect Objects", self.detect_objects),
            ("Estimate Depth", self.estimate_depth),
            ("Select Point", self.select_placement_point),
            ("Analyze with Gemini", self.analyze_with_gemini),
            ("Create Composite", self.create_composite)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 Step: {step_name}")
            if not step_func():
                print(f"❌ Pipeline failed at: {step_name}")
                return False
        
        print("\n" + "="*50)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("📁 Check the 'data/output' folder for all results:")
        print("   • composite_result.png - Final placed object")
        print("   • analysis_result.json - Complete analysis")
        print("   • detected_objects.json - Object detection results")
        print("   • depth_map.png - Depth visualization")
        print("   • And more...")
        
        return True


def main():
    """Main function."""
    try:
        pipeline = ObjectPlacementPipeline(data_folder="data")
        success = pipeline.run_pipeline()
        
        if not success:
            print("\n💡 Expected folder structure:")
            print("data/")
            print("  ├── api_key.txt          # Your Gemini API key")
            print("  ├── env.jpg/png          # Environment image")
            print("  └── obj.jpg/png          # Object to place")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
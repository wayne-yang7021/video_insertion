"""
Intelligent Object Placement and Resizing Pipeline

A comprehensive pipeline that combines:
- YOLO + SAM for object detection and segmentation
- MiDaS for depth estimation
- Gemini AI for intelligent placement analysis and sizing
- Advanced image composition techniques

Version: 1.0
"""

__version__ = "1.0.0"
__author__ = "Object Placement Pipeline"

from .config import Config
from .image_loader import ImageLoader
from .object_detector import ObjectDetector
from .depth_estimator import DepthEstimator
from .point_selector import PointSelector
from .gemini_analyzer import GeminiAnalyzer
from .image_compositor import ImageCompositor
from .results_saver import ResultsSaver

__all__ = [
    'Config',
    'ImageLoader', 
    'ObjectDetector',
    'DepthEstimator',
    'PointSelector',
    'GeminiAnalyzer',
    'ImageCompositor',
    'ResultsSaver'
]
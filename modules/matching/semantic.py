
"""
語意匹配處理模組
負責協調語意匹配的邏輯流程，使用models中的SemanticMatcher進行實際計算
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

# Import the model class
from modules.models.matching_model import SemanticMatcher


class SemanticMatchingProcessor:
    """簡化的語意匹配處理器"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.semantic_matcher = SemanticMatcher(device=device)
        
    def get_best_matches(self, matching_results: List[Dict], top_k: int = 3) -> List[Dict]:
        """獲取最佳的K個匹配結果"""
        return sorted(matching_results, key=lambda x: x['compatibility_score'], reverse=True)[:top_k]
    
    def find_best_surfaces(self, object_label: str, object_embedding: np.ndarray, 
                          background_surfaces: List[Dict], top_k: int = 3,
                          scene_embedding: Optional[np.ndarray] = None) -> List[Dict]:
        """
        為單一物件找到最佳的前k個表面匹配
        
        Args:
            object_label: 物件標籤 (如 "cup", "book")
            object_embedding: 物件的CLIP嵌入向量 [512,]
            background_surfaces: 背景表面列表 [{'label': str, 'bbox': [x1,y1,x2,y2], ...}, ...]
            top_k: 返回前k個最佳匹配，預設3
            scene_embedding: 可選的場景嵌入向量
            
        Returns:
            List[Dict]: 前k個最佳匹配，格式為:
            [
                {
                    'surface_label': str,
                    'bbox': [x1, y1, x2, y2],
                    'compatibility_score': float,
                    'rank': int,
                    'detailed_scores': Dict
                },
                ...
            ]
        """
        # 準備物件資訊
        object_info = {
            'primary_label': object_label,
            'embedding': object_embedding,
            'id': f'input_{object_label}'
        }
        
        # 計算與每個表面的匹配分數
        matches = []
        for i, surface in enumerate(background_surfaces):
            try:
                compatibility_score, detailed_scores = self.semantic_matcher.calculate_semantic_compatibility(
                    object_info=object_info,
                    surface_obj=surface,
                    scene_embedding=scene_embedding
                )
                
                match = {
                    'surface_label': surface['label'],
                    'bbox': surface['bbox'],
                    'compatibility_score': compatibility_score,
                    'detailed_scores': detailed_scores,
                    'surface_id': surface.get('id', f'surface_{i}'),
                    'surface_confidence': surface.get('confidence', 1.0)
                }
                matches.append(match)
                
            except Exception as e:
                print(f"⚠️ 計算 {object_label} 與 {surface['label']} 的匹配失敗: {e}")
                continue
        
        # 按分數排序並取前k個
        matches.sort(key=lambda x: x['compatibility_score'], reverse=True)
        top_matches = matches[:top_k]
        
        # 添加排名資訊
        for rank, match in enumerate(top_matches, 1):
            match['rank'] = rank
        
        return top_matches
    
# 主要API函數
def quick_semantic_match(object_info: Dict, surface_obj: Dict, 
                        scene_embedding: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
    """快速單次語意匹配"""
    matcher = SemanticMatcher()
    return matcher.calculate_semantic_compatibility(object_info, surface_obj, scene_embedding)


def find_best_placement(object_label: str, object_embedding: np.ndarray,
                       background_surfaces: List[Dict], top_k: int = 3,
                       scene_embedding: Optional[np.ndarray] = None) -> List[Dict]:
    """
    便利函數：為物件找到最佳放置位置
    
    Args:
        object_label: 物件標籤 (如 "cup", "book")
        object_embedding: 物件的CLIP嵌入向量 [512,]
        background_surfaces: 背景表面列表 [{'label': str, 'bbox': [x1,y1,x2,y2], ...}, ...]
        top_k: 返回前k個最佳匹配，預設3
        scene_embedding: 可選的場景嵌入向量
        
    Returns:
        List[Dict]: 前k個最佳放置位置及其bbox
    """
    processor = SemanticMatchingProcessor()
    return processor.find_best_surfaces(
        object_label=object_label,
        object_embedding=object_embedding,
        background_surfaces=background_surfaces,
        top_k=top_k,
        scene_embedding=scene_embedding
    )

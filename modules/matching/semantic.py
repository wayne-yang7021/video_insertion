
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
    """語意匹配處理器 - 負責協調語意匹配的邏輯流程"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.semantic_matcher = SemanticMatcher(device=device)
        self.matching_history = []  # 記錄匹配歷史
        
    def process_semantic_matching(self, insert_objects: List[Dict], 
                                background_surfaces: List[Dict],
                                scene_info: Optional[Dict] = None) -> List[Dict]:
        """
        處理語意匹配的主要流程
        
        Args:
            insert_objects: 要插入的物件列表
            background_surfaces: 背景表面物件列表  
            scene_info: 場景資訊（包含scene_embedding等）
            
        Returns:
            List[Dict]: 匹配結果列表，每個結果包含物件、表面、分數等資訊
        """
        matching_results = []
        scene_embedding = scene_info.get('scene_embedding') if scene_info else None
        
        for obj in insert_objects:
            obj_results = []
            
            for surface in background_surfaces:
                # 使用SemanticMatcher計算相容性
                compatibility_score, detailed_scores = self.semantic_matcher.calculate_semantic_compatibility(
                    object_info=obj,
                    surface_obj=surface,
                    scene_embedding=scene_embedding
                )
                
                result = {
                    'object': obj,
                    'surface': surface,
                    'compatibility_score': compatibility_score,
                    'detailed_scores': detailed_scores,
                    'timestamp': self._get_timestamp()
                }
                
                obj_results.append(result)
            
            # 按分數排序，取最佳匹配
            obj_results.sort(key=lambda x: x['compatibility_score'], reverse=True)
            matching_results.extend(obj_results)
            
            # 記錄匹配歷史
            self.matching_history.append({
                'object_id': obj.get('id', 'unknown'),
                'best_match': obj_results[0] if obj_results else None,
                'total_candidates': len(obj_results)
            })
        
        return matching_results
    
    def filter_matches_by_threshold(self, matching_results: List[Dict], 
                                  threshold: float = 0.5) -> List[Dict]:
        """根據閾值過濾匹配結果"""
        return [result for result in matching_results 
                if result['compatibility_score'] >= threshold]
    
    def get_best_matches(self, matching_results: List[Dict], 
                        top_k: int = 3) -> List[Dict]:
        """獲取最佳的K個匹配結果"""
        sorted_results = sorted(matching_results, 
                              key=lambda x: x['compatibility_score'], 
                              reverse=True)
        return sorted_results[:top_k]
    
    def analyze_matching_quality(self, matching_results: List[Dict]) -> Dict:
        """分析匹配品質"""
        if not matching_results:
            return {'status': 'no_matches', 'quality': 0.0}
        
        scores = [result['compatibility_score'] for result in matching_results]
        
        analysis = {
            'total_matches': len(matching_results),
            'average_score': np.mean(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'std_score': np.std(scores),
            'high_quality_matches': len([s for s in scores if s >= 0.7]),
            'medium_quality_matches': len([s for s in scores if 0.4 <= s < 0.7]),
            'low_quality_matches': len([s for s in scores if s < 0.4])
        }
        
        # 判斷整體品質
        if analysis['average_score'] >= 0.7:
            analysis['overall_quality'] = 'high'
        elif analysis['average_score'] >= 0.4:
            analysis['overall_quality'] = 'medium'
        else:
            analysis['overall_quality'] = 'low'
            
        return analysis
    
    def get_matching_history(self) -> List[Dict]:
        """獲取匹配歷史"""
        return self.matching_history
    
    def clear_history(self):
        """清除匹配歷史"""
        self.matching_history = []
    
    def _get_timestamp(self) -> str:
        """獲取時間戳"""
        import datetime
        return datetime.datetime.now().isoformat()


# 向後相容性：保留原有的函數介面
def _calculate_semantic_score(object_info: Dict, surface_obj: Dict, scene_embedding: np.ndarray) -> float:
    """🎯 核心創新：基於CLIP的語意匹配分數 (保留原有方法作為向後相容)"""
    try:
        # 使用新的SemanticMatcher
        matcher = SemanticMatcher()
        compatibility_score, _ = matcher.calculate_semantic_compatibility(
            object_info=object_info,
            surface_obj=surface_obj,
            scene_embedding=scene_embedding
        )
        return compatibility_score
        
    except Exception as e:
        print(f"⚠️ 語意分數計算失敗: {e}")
        return 0.3


# 便利函數：快速語意匹配
def quick_semantic_match(object_info: Dict, surface_obj: Dict, 
                        scene_embedding: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
    """快速語意匹配函數"""
    matcher = SemanticMatcher()
    return matcher.calculate_semantic_compatibility(object_info, surface_obj, scene_embedding)

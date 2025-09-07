
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
    
    def find_best_placement_positions(self, object_label: str, object_embedding: np.ndarray, 
                                     background_objects: List[Dict], top_k: int = 3,
                                     scene_embedding: Optional[np.ndarray] = None) -> List[Dict]:
        """
        為單一物件找到最佳的前k個放置位置
        
        Args:
            object_label: 物件標籤 (如 "cup", "book")
            object_embedding: 物件的CLIP嵌入向量 [512,]
            background_objects: 背景物件列表 [{'label': str, 'bbox': [x1,y1,x2,y2], ...}, ...]
            top_k: 返回前k個最佳匹配，預設3
            scene_embedding: 可選的場景嵌入向量
            
        Returns:
            List[Dict]: 前k個最佳放置位置，格式為:
            [
                {
                    'reference_object': str,
                    'placement_type': str,  # 'on_surface', 'beside', 'near'
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
        
        # 計算與每個背景物件的匹配分數和放置類型
        matches = []
        for i, bg_obj in enumerate(background_objects):
            try:
                # 計算相容性分數
                compatibility_score, detailed_scores = self.semantic_matcher.calculate_semantic_compatibility(
                    object_info=object_info,
                    surface_obj=bg_obj,
                    scene_embedding=scene_embedding
                )
                
                # 決定放置類型
                placement_type = self._determine_placement_type(object_label, bg_obj['label'])
                
                match = {
                    'reference_object': bg_obj['label'],
                    'placement_type': placement_type,
                    'bbox': bg_obj['bbox'],
                    'compatibility_score': compatibility_score,
                    'detailed_scores': detailed_scores,
                    'object_id': bg_obj.get('id', f'obj_{i}'),
                    'object_confidence': bg_obj.get('confidence', 1.0),
                    # 向後相容性欄位
                    'surface_label': bg_obj['label'],
                    'surface_id': bg_obj.get('id', f'obj_{i}'),
                    'surface_confidence': bg_obj.get('confidence', 1.0)
                }
                matches.append(match)
                
            except Exception as e:
                print(f"⚠️ 計算 {object_label} 與 {bg_obj['label']} 的匹配失敗: {e}")
                continue
        
        # 按分數排序並取前k個
        matches.sort(key=lambda x: x['compatibility_score'], reverse=True)
        top_matches = matches[:top_k]
        
        # 添加排名資訊
        for rank, match in enumerate(top_matches, 1):
            match['rank'] = rank
        
        return top_matches
    
    def _determine_placement_type(self, object_label: str, reference_label: str) -> str:
        """決定放置類型"""
        # 可以作為表面的物件
        surface_objects = ['dining table', 'chair', 'couch', 'bed', 'bench', 'sink', 'toilet']
        
        # 適合放在旁邊的組合
        beside_pairs = [
            ('person', ['chair', 'couch', 'bed']),
            ('backpack', ['chair', 'person']),
            ('umbrella', ['person', 'chair']),
            ('handbag', ['person', 'chair']),
            ('suitcase', ['person', 'bed']),
        ]
        
        # 檢查是否適合放在表面上
        if reference_label in surface_objects:
            return 'on_surface'
        
        # 檢查是否適合放在旁邊
        for obj, refs in beside_pairs:
            if object_label == obj and reference_label in refs:
                return 'beside'
        
        # 預設為附近
        return 'near'
    
# 主要API函數
def quick_semantic_match(object_info: Dict, surface_obj: Dict, 
                        scene_embedding: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
    """快速單次語意匹配"""
    matcher = SemanticMatcher()
    return matcher.calculate_semantic_compatibility(object_info, surface_obj, scene_embedding)


def find_best_placement(object_label: str, object_embedding: np.ndarray,
                       background_objects: List[Dict] = None, top_k: int = 3,
                       scene_embedding: Optional[np.ndarray] = None,
                       background_surfaces: List[Dict] = None,
                       image: Optional[np.ndarray] = None,
                       use_vlm: bool = True,
                       use_llm_enhanced: bool = False) -> List[Dict]:
    """
    便利函數：為物件找到最佳放置位置，可選擇使用VLM增強或LLM增強
    
    Args:
        object_label: 物件標籤 (如 "cup", "book")
        object_embedding: 物件的CLIP嵌入向量 [512,]
        background_objects: 背景物件列表 [{'label': str, 'bbox': [x1,y1,x2,y2], ...}, ...]
        top_k: 返回前k個最佳匹配，預設3
        scene_embedding: 可選的場景嵌入向量
        background_surfaces: (向後相容) 同background_objects
        image: 可選的場景圖片，用於VLM增強
        use_vlm: 是否使用VLM增強，預設True
        use_llm_enhanced: 是否使用LLM增強分析，預設False
        
    Returns:
        List[Dict]: 前k個最佳放置位置，包含放置類型和參考物件
    """
    # 向後相容性處理
    if background_surfaces is not None and background_objects is None:
        background_objects = background_surfaces
    elif background_objects is None:
        raise ValueError("必須提供 background_objects 或 background_surfaces")
    
    # 如果啟用LLM增強且提供了圖片，則使用LLM增強分析
    if use_llm_enhanced and image is not None:
        try:
            from modules.matching.llm_enhanced import create_llm_enhanced_matcher
            llm_matcher = create_llm_enhanced_matcher()
            
            if llm_matcher.is_enabled():
                enhanced_result = llm_matcher.enhanced_placement_analysis(
                    image, object_label, background_objects
                )
                print("✅ LLM增強分析完成")
                
                # 將單一結果轉換為列表格式以保持向後相容
                if enhanced_result and enhanced_result.get('reference_object') != 'none':
                    return [{
                        'reference_object': enhanced_result['reference_object'],
                        'bbox': enhanced_result['bbox'],
                        'compatibility_score': enhanced_result['score'],
                        'source': enhanced_result.get('source', 'llm_enhanced'),
                        'rank': 1,
                        'placement_type': 'llm_suggested'
                    }]
                else:
                    print("⚠️ LLM未找到合適建議，使用標準流程")
            else:
                print("⚠️ LLM增強功能未啟用，使用標準流程")
        except Exception as e:
            print(f"⚠️ LLM增強失敗，使用標準流程: {e}")
    
    # 執行現有的語意匹配
    processor = SemanticMatchingProcessor()
    semantic_matches = processor.find_best_placement_positions(
        object_label=object_label,
        object_embedding=object_embedding,
        background_objects=background_objects,
        top_k=top_k,
        scene_embedding=scene_embedding
    )
    
    # 如果啟用VLM且提供了圖片，則使用VLM增強
    if use_vlm and image is not None:
        try:
            from modules.validation.vlm.vlm import create_vlm_helper
            vlm_helper = create_vlm_helper()
            
            if vlm_helper.is_enabled():
                enhanced_matches = vlm_helper.enhance_matches(image, object_label, semantic_matches)
                print(f"✅ VLM增強完成，返回{len(enhanced_matches)}個建議")
                return enhanced_matches
            else:
                print("⚠️ VLM功能未啟用，使用原始語意匹配結果")
        except Exception as e:
            print(f"⚠️ VLM增強失敗，使用原始結果: {e}")
    
    return semantic_matches

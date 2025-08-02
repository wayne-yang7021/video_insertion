import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import clip
import torch


class SemanticMatcher:
    """完整的語意匹配器，整合多個判斷維度來評估背景物件和要插入物品的語意關聯性"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # 物理支撐關係規則
        self.support_rules = {
            "table": ["cup", "book", "laptop", "plate", "phone", "glass", "bottle", "pen", "notebook"],
            "desk": ["laptop", "book", "pen", "monitor", "keyboard", "mouse", "phone", "notebook"],
            "shelf": ["book", "decoration", "bottle", "plant", "frame", "clock", "vase"],
            "counter": ["appliance", "food", "utensil", "cup", "plate", "bottle", "jar"],
            "floor": ["chair", "table", "plant", "bag", "shoes", "box"],
            "bed": ["pillow", "book", "phone", "remote", "clothes"],
            "sofa": ["pillow", "remote", "book", "phone", "blanket", "cushion"],
            "nightstand": ["lamp", "clock", "book", "phone", "glass", "bottle"]
        }
        
        # 功能場景分組
        self.scene_groups = {
            "kitchen": ["cup", "plate", "spoon", "pot", "counter", "stove", "knife", "bowl", "glass"],
            "office": ["laptop", "book", "pen", "desk", "chair", "monitor", "keyboard", "mouse"],
            "living_room": ["sofa", "tv", "remote", "table", "cushion", "lamp", "plant"],
            "bedroom": ["bed", "pillow", "lamp", "nightstand", "clothes", "book"],
            "bathroom": ["towel", "soap", "toothbrush", "mirror", "sink"],
            "dining": ["table", "chair", "plate", "cup", "utensil", "napkin"]
        }
        
        # 尺寸分類
        self.size_categories = {
            "tiny": ["ring", "coin", "button", "key", "earring"],
            "small": ["cup", "phone", "book", "remote", "pen", "glass", "bottle"],
            "medium": ["laptop", "plate", "lamp", "clock", "vase", "bowl"],
            "large": ["chair", "monitor", "plant", "bag", "pillow"],
            "huge": ["table", "sofa", "bed", "desk", "tv"]
        }
        
        # 權重配置
        self.weights = {
            "clip_similarity": 0.35,
            "context_matching": 0.25,
            "physical_support": 0.20,
            "scene_coherence": 0.15,
            "size_compatibility": 0.05
        }
    
    def calculate_semantic_compatibility(self, object_info: Dict, surface_obj: Dict, 
                                       scene_embedding: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
        """
        計算綜合語意相容性分數
        
        Args:
            object_info: 要插入物件的資訊 {'primary_label': str, 'embedding': np.ndarray, ...}
            surface_obj: 背景表面物件資訊 {'label': str, 'bbox': [...], ...}
            scene_embedding: 場景的CLIP嵌入向量
            
        Returns:
            Tuple[float, Dict]: (總分數, 各維度詳細分數)
        """
        scores = {}
        
        try:
            # 1. CLIP語意相似度
            scores['clip_similarity'] = self._calculate_clip_similarity(object_info, surface_obj)
            
            # 2. 上下文匹配
            scores['context_matching'] = self._calculate_context_matching(
                object_info, surface_obj, scene_embedding
            )
            
            # 3. 物理支撐相容性
            scores['physical_support'] = self._calculate_physical_support(
                object_info['primary_label'], surface_obj['label']
            )
            
            # 4. 場景一致性
            scores['scene_coherence'] = self._calculate_scene_coherence(
                object_info['primary_label'], surface_obj['label']
            )
            
            # 5. 尺寸相容性
            scores['size_compatibility'] = self._calculate_size_compatibility(
                object_info['primary_label'], surface_obj['label']
            )
            
            # 計算加權總分
            total_score = sum(scores[key] * self.weights[key] for key in scores.keys())
            total_score = max(0.0, min(1.0, total_score))
            
            return total_score, scores
            
        except Exception as e:
            print(f"⚠️ 語意相容性計算失敗: {e}")
            return 0.3, {"error": str(e)}
    
    def _calculate_clip_similarity(self, object_info: Dict, surface_obj: Dict) -> float:
        """計算基於CLIP的語意相似度"""
        try:
            object_embedding = object_info['embedding']
            
            # 生成表面物件的語意描述
            surface_text = f"a {surface_obj['label']} that can hold objects"
            surface_embedding = self._get_clip_text_embedding(surface_text)
            
            # 計算相似度
            similarity = cosine_similarity([object_embedding], [surface_embedding])[0][0]
            return max(0.0, similarity)
            
        except Exception as e:
            print(f"⚠️ CLIP相似度計算失敗: {e}")
            return 0.3
    
    def _calculate_context_matching(self, object_info: Dict, surface_obj: Dict, 
                                  scene_embedding: Optional[np.ndarray]) -> float:
        """計算上下文匹配分數"""
        try:
            object_label = object_info['primary_label']
            surface_label = surface_obj['label']
            
            # 生成多種上下文描述
            contexts = [
                f"a {object_label} sitting on a {surface_label}",
                f"a {object_label} placed on top of a {surface_label}",
                f"a {surface_label} with a {object_label} on it",
                f"{object_label} and {surface_label} together in a room"
            ]
            
            context_scores = []
            for context in contexts:
                context_embedding = self._get_clip_text_embedding(context)
                
                if scene_embedding is not None:
                    # 與場景嵌入比較
                    score = cosine_similarity([context_embedding], [scene_embedding])[0][0]
                else:
                    # 使用上下文本身的語意強度
                    neutral_embedding = self._get_clip_text_embedding("objects in a room")
                    score = cosine_similarity([context_embedding], [neutral_embedding])[0][0]
                
                context_scores.append(max(0.0, score))
            
            return np.mean(context_scores)
            
        except Exception as e:
            print(f"⚠️ 上下文匹配計算失敗: {e}")
            return 0.3
    
    def _calculate_physical_support(self, object_label: str, surface_label: str) -> float:
        """計算物理支撐相容性"""
        try:
            # 檢查直接支撐關係
            if surface_label in self.support_rules:
                if object_label in self.support_rules[surface_label]:
                    return 1.0
            
            # 檢查語意相近的支撐關係
            for surface_type, supported_objects in self.support_rules.items():
                if self._is_similar_object(surface_label, surface_type):
                    if object_label in supported_objects:
                        return 0.8
                    if any(self._is_similar_object(object_label, obj) for obj in supported_objects):
                        return 0.6
            
            # 基本物理合理性檢查
            if self._basic_physics_check(object_label, surface_label):
                return 0.4
            
            return 0.1
            
        except Exception as e:
            print(f"⚠️ 物理支撐計算失敗: {e}")
            return 0.3
    
    def _calculate_scene_coherence(self, object_label: str, surface_label: str) -> float:
        """計算場景一致性分數"""
        try:
            object_scenes = self._get_object_scenes(object_label)
            surface_scenes = self._get_object_scenes(surface_label)
            
            # 計算共同場景
            common_scenes = object_scenes.intersection(surface_scenes)
            
            if common_scenes:
                # 有共同場景，分數基於共同場景數量
                return min(1.0, len(common_scenes) / 2.0)
            else:
                # 沒有共同場景，檢查是否為通用物件
                if self._is_universal_object(object_label) or self._is_universal_object(surface_label):
                    return 0.5
                return 0.2
                
        except Exception as e:
            print(f"⚠️ 場景一致性計算失敗: {e}")
            return 0.3
    
    def _calculate_size_compatibility(self, object_label: str, surface_label: str) -> float:
        """計算尺寸相容性"""
        try:
            object_size = self._get_size_category(object_label)
            surface_size = self._get_size_category(surface_label)
            
            if not object_size or not surface_size:
                return 0.5  # 未知尺寸，給予中等分數
            
            size_hierarchy = ["tiny", "small", "medium", "large", "huge"]
            obj_idx = size_hierarchy.index(object_size)
            surf_idx = size_hierarchy.index(surface_size)
            
            # 物件應該比支撐面小或相等
            if obj_idx <= surf_idx:
                return 1.0 - (surf_idx - obj_idx) * 0.1  # 尺寸差距越大，分數略降
            else:
                # 物件比支撐面大，分數大幅下降
                return max(0.1, 1.0 - (obj_idx - surf_idx) * 0.4)
                
        except Exception as e:
            print(f"⚠️ 尺寸相容性計算失敗: {e}")
            return 0.5
    
    def _get_clip_text_embedding(self, text: str) -> np.ndarray:
        """獲取文本的CLIP嵌入"""
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_embedding = self.clip_model.encode_text(text_tokens)
            return text_embedding.cpu().numpy()[0]
    
    def _get_object_scenes(self, object_label: str) -> set:
        """獲取物件可能出現的場景"""
        scenes = set()
        for scene, objects in self.scene_groups.items():
            if object_label in objects or any(self._is_similar_object(object_label, obj) for obj in objects):
                scenes.add(scene)
        return scenes
    
    def _get_size_category(self, object_label: str) -> Optional[str]:
        """獲取物件的尺寸分類"""
        for size, objects in self.size_categories.items():
            if object_label in objects:
                return size
        return None
    
    def _is_similar_object(self, obj1: str, obj2: str) -> bool:
        """檢查兩個物件是否語意相近"""
        # 簡單的語意相似性檢查
        similar_groups = [
            ["cup", "mug", "glass"],
            ["table", "desk", "counter"],
            ["book", "notebook", "magazine"],
            ["phone", "mobile", "smartphone"],
            ["laptop", "computer", "notebook"]
        ]
        
        for group in similar_groups:
            if obj1 in group and obj2 in group:
                return True
        return False
    
    def _is_universal_object(self, object_label: str) -> bool:
        """檢查是否為通用物件（可出現在多個場景）"""
        universal_objects = ["book", "phone", "cup", "bottle", "bag", "clock", "plant"]
        return object_label in universal_objects
    
    def _basic_physics_check(self, object_label: str, surface_label: str) -> bool:
        """基本物理合理性檢查"""
        # 檢查一些明顯不合理的組合
        impossible_combinations = [
            ("liquid", "vertical_surface"),
            ("heavy_object", "fragile_surface")
        ]
        
        # 這裡可以根據需要擴展更多物理規則
        return True  # 暫時返回True，可根據需要添加具體規則


class DepthMatcher:
    """深度匹配器 - 用於後續擴展深度相關的匹配邏輯"""
    
    def __init__(self):
        pass
    
    def calculate_depth_compatibility(self, object_info: Dict, surface_obj: Dict, depth_map: np.ndarray) -> float:
        """計算深度相容性分數"""
        # TODO: 實作深度匹配邏輯
        return 0.5
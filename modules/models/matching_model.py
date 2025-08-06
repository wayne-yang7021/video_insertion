import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import clip
import torch


class SemanticMatcher:
    """簡化的語意匹配器，專注於核心功能"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        
        # 簡化的權重配置 - 只保留最重要的3個維度
        self.weights = {
            "semantic_similarity": 0.4,    # CLIP語意相似度 + 上下文
            "physical_compatibility": 0.4,  # 物理支撐 + 穩定性
            "scene_coherence": 0.2          # 場景一致性
        }
        
        # 合併的支撐規則 - 簡化版本
        self.support_rules = self._create_support_rules()
        
        # 不穩定組合 - 關鍵安全規則
        self.unsafe_pairs = [
            ("sports ball", ["chair", "couch"]),
            ("wine glass", ["couch", "bed"]),
            ("cup", ["couch", "bed"]),
            ("laptop", ["sink", "toilet"]),
            ("cell phone", ["sink"])
        ]

    def _create_support_rules(self) -> Dict[str, List[str]]:
        """創建簡化的支撐規則"""
        return {
            # 主要表面
            "dining table": ["cup", "wine glass", "bottle", "bowl", "fork", "knife", "spoon", 
                           "book", "laptop", "mouse", "keyboard", "cell phone", "clock", "vase"],
            "chair": ["backpack", "handbag", "book", "laptop", "remote", "cell phone"],
            "couch": ["backpack", "book", "remote", "cell phone", "laptop"],
            "bed": ["book", "remote", "cell phone", "laptop"],
            "sink": ["cup", "wine glass", "bottle", "bowl", "toothbrush"],
            # 其他表面預設支撐小物件
            "default": ["remote", "cell phone", "book"]
        }

    def calculate_semantic_compatibility(self, object_info: Dict, surface_obj: Dict,
                                         scene_embedding: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
        """
        計算語意相容性分數 - 簡化版本
        
        Args:
            object_info: 物件資訊 {'primary_label': str, 'embedding': np.ndarray}
            surface_obj: 表面資訊 {'label': str, 'bbox': [...]}
            scene_embedding: 可選的場景嵌入向量
            
        Returns:
            Tuple[float, Dict]: (總分數, 詳細分數)
        """
        try:
            object_label = object_info['primary_label']
            surface_label = surface_obj['label']
            
            # 1. 語意相似度 (CLIP + 上下文)
            semantic_score = self._calculate_semantic_similarity(object_info, surface_obj, scene_embedding)
            
            # 2. 物理相容性 (支撐 + 穩定性)
            physical_score = self._calculate_physical_compatibility(object_label, surface_label)
            
            # 3. 場景一致性
            scene_score = self._calculate_scene_coherence(object_label, surface_label)
            
            # 組合分數
            scores = {
                'semantic_similarity': semantic_score,
                'physical_compatibility': physical_score,
                'scene_coherence': scene_score
            }
            
            # 計算總分
            total_score = sum(scores[key] * self.weights[key] for key in scores.keys())
            total_score = max(0.0, min(1.0, total_score))
            
            return total_score, scores
            
        except Exception as e:
            print(f"⚠️ 語意相容性計算失敗: {e}")
            return 0.3, {"error": str(e)}

    def _calculate_semantic_similarity(self, object_info: Dict, surface_obj: Dict, 
                                      scene_embedding: Optional[np.ndarray]) -> float:
        """計算語意相似度 (結合CLIP和上下文)"""
        try:
            object_label = object_info['primary_label']
            surface_label = surface_obj['label']
            object_embedding = object_info['embedding']
            
            # CLIP相似度
            surface_text = f"a {surface_label} for placing {object_label}"
            surface_embedding = self._get_clip_text_embedding(surface_text)
            clip_score = cosine_similarity([object_embedding], [surface_embedding])[0][0]
            
            # 上下文相似度
            context_text = f"{object_label} on {surface_label}"
            context_embedding = self._get_clip_text_embedding(context_text)
            
            if scene_embedding is not None:
                context_score = cosine_similarity([context_embedding], [scene_embedding])[0][0]
            else:
                context_score = clip_score  # 如果沒有場景，使用CLIP分數
            
            # 結合兩個分數
            return max(0.0, (clip_score + context_score) / 2.0)
            
        except Exception as e:
            print(f"⚠️ 語意相似度計算失敗: {e}")
            return 0.3

    def _calculate_physical_compatibility(self, object_label: str, surface_label: str) -> float:
        """計算物理相容性 (結合支撐和穩定性)"""
        try:
            # 檢查不安全組合
            for unsafe_obj, unsafe_surfaces in self.unsafe_pairs:
                if object_label == unsafe_obj and surface_label in unsafe_surfaces:
                    return 0.1  # 不安全組合給極低分
            
            # 檢查支撐關係
            support_score = 0.5  # 預設分數
            
            if surface_label in self.support_rules:
                if object_label in self.support_rules[surface_label]:
                    support_score = 1.0
            else:
                # 未知表面，檢查是否為小物件
                if object_label in self.support_rules["default"]:
                    support_score = 0.7
            
            # 特殊情況調整
            if object_label in ["laptop", "cell phone"] and surface_label in ["sink", "toilet"]:
                support_score = 0.1  # 電子產品避免水
            elif object_label in ["cup", "wine glass"] and surface_label == "dining table":
                support_score = 1.0  # 經典組合
            
            return support_score
            
        except Exception as e:
            print(f"⚠️ 物理相容性計算失敗: {e}")
            return 0.5

    def _calculate_scene_coherence(self, object_label: str, surface_label: str) -> float:
        """計算場景一致性分數 - 簡化版本"""
        try:
            # 簡化的場景規則
            scene_pairs = {
                # 廚房場景
                ("cup", "dining table"): 0.9,
                ("fork", "dining table"): 0.9,
                ("knife", "dining table"): 0.9,
                ("spoon", "dining table"): 0.9,
                ("bowl", "dining table"): 0.9,
                
                # 客廳場景
                ("remote", "couch"): 0.9,
                ("book", "couch"): 0.8,
                
                # 臥室場景
                ("book", "bed"): 0.8,
                ("cell phone", "bed"): 0.7,
                
                # 辦公場景
                ("laptop", "dining table"): 0.8,
                ("mouse", "dining table"): 0.8,
                ("keyboard", "dining table"): 0.8,
                
                # 衛生間場景
                ("toothbrush", "sink"): 0.9,
            }
            
            pair = (object_label, surface_label)
            if pair in scene_pairs:
                return scene_pairs[pair]
            
            # 通用物件給中等分數
            universal_objects = ["book", "cell phone", "cup", "remote"]
            if object_label in universal_objects:
                return 0.6
            
            return 0.4  # 預設分數
            
        except Exception as e:
            print(f"⚠️ 場景一致性計算失敗: {e}")
            return 0.5

    def _get_clip_text_embedding(self, text: str) -> np.ndarray:
        """獲取文本的CLIP嵌入"""
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_embedding = self.clip_model.encode_text(text_tokens)
            return text_embedding.cpu().numpy()[0]


class DepthMatcher:
    """深度匹配器 - 用於後續擴展深度相關的匹配邏輯"""

    def __init__(self):
        pass

    def calculate_depth_compatibility(self, object_info: Dict, surface_obj: Dict, depth_map: np.ndarray) -> float:
        """計算深度相容性分數"""
        # TODO: 實作深度匹配邏輯
        return 0.5

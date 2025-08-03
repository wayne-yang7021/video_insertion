import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import clip
import torch


class SemanticMatcher:
    """完整的語意匹配器，整合多個判斷維度來評估背景物件和要插入物品的語意關聯性"""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", device=device)

        # 物理支撐關係規則 (基於COCO 80類別)
        self.support_rules = {
            "dining table": ["cup", "wine glass", "bottle", "bowl", "fork", "knife", "spoon", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "book", "laptop", "mouse", "remote", "keyboard", "cell phone", "clock", "vase", "scissors"],
            "chair": ["backpack", "handbag", "book", "laptop", "remote", "cell phone", "teddy bear"],
            "couch": ["backpack", "handbag", "book", "remote", "cell phone", "laptop", "teddy bear"],
            "bed": ["book", "remote", "cell phone", "laptop", "teddy bear"],
            "bench": ["backpack", "handbag", "book", "remote", "cell phone"],
            "tv": ["remote"],
            "toilet": ["book", "cell phone", "toothbrush"],
            "sink": ["cup", "wine glass", "bottle", "bowl", "fork", "knife", "spoon", "toothbrush", "scissors"],
            "refrigerator": [],
            "microwave": [],
            "oven": [],
            "toaster": []
        }

        # 功能場景分組 (基於COCO 80類別)
        self.scene_groups = {
            "kitchen": ["cup", "wine glass", "bottle", "bowl", "fork", "knife", "spoon", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "microwave", "oven", "toaster", "sink", "refrigerator"],
            "office": ["laptop", "mouse", "keyboard", "book", "chair", "dining table", "cell phone", "clock", "scissors"],
            "living_room": ["couch", "tv", "remote", "dining table", "chair", "potted plant", "book", "clock", "vase"],
            "bedroom": ["bed", "chair", "book", "clock", "cell phone", "laptop", "teddy bear", "hair drier"],
            "bathroom": ["toilet", "sink", "toothbrush", "hair drier"],
            "dining": ["dining table", "chair", "cup", "wine glass", "bottle", "bowl", "fork", "knife", "spoon"],
            "outdoor": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"],
            "sports": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
            "personal": ["backpack", "umbrella", "handbag", "tie", "suitcase", "cell phone", "book", "scissors", "teddy bear"]
        }

        # 尺寸分類 (基於COCO 80類別)
        self.size_categories = {
            "tiny": ["fork", "knife", "spoon", "mouse", "remote", "cell phone", "scissors", "toothbrush"],
            "small": ["cup", "wine glass", "bottle", "bowl", "banana", "apple", "orange", "book", "clock", "vase", "baseball glove", "sports ball"],
            "medium": ["laptop", "keyboard", "backpack", "handbag", "suitcase", "frisbee", "baseball bat", "tennis racket", "potted plant", "teddy bear", "microwave", "toaster"],
            "large": ["chair", "tv", "bicycle", "skateboard", "surfboard", "umbrella", "oven", "refrigerator"],
            "huge": ["dining table", "couch", "bed", "toilet", "sink", "car", "motorcycle", "bus", "truck", "airplane", "train", "boat"]
        }

        # 權重配置
        self.weights = {
            "clip_similarity": 0.25,
            "context_matching": 0.20,
            "physical_support": 0.20,
            "scene_coherence": 0.15,
            "stability_assessment": 0.10,
            "usage_frequency": 0.05,
            "aesthetic_harmony": 0.05
        }
        
        # 穩定性評估規則
        self.unstable_combinations = [
            ("sports ball", "chair"),
            ("sports ball", "couch"),
            ("bottle", "couch"),
            ("wine glass", "couch"),
            ("cup", "couch"),
        ]
        
        # 使用頻率資料
        self.common_pairs = {
            ("cup", "dining table"): 0.9,
            ("laptop", "dining table"): 0.8,
            ("book", "bed"): 0.7,
            ("remote", "couch"): 0.9,
            ("cell phone", "dining table"): 0.8,
            ("wine glass", "dining table"): 0.9,
            ("fork", "dining table"): 0.9,
            ("knife", "dining table"): 0.9,
            ("spoon", "dining table"): 0.9,
            ("bowl", "dining table"): 0.8,
            ("book", "chair"): 0.6,
            ("laptop", "bed"): 0.5,
            ("teddy bear", "bed"): 0.9,
            ("toothbrush", "sink"): 0.9,
        }
        
        # 美學和諧規則
        self.aesthetic_groups = {
            "formal_dining": ["wine glass", "fork", "knife", "spoon", "dining table"],
            "casual_living": ["remote", "book", "cup", "couch"],
            "work_setup": ["laptop", "mouse", "keyboard", "dining table"],
            "personal_care": ["toothbrush", "hair drier", "sink", "toilet"],
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
            scores['clip_similarity'] = self._calculate_clip_similarity(
                object_info, surface_obj)

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

            # 5. 穩定性評估
            scores['stability_assessment'] = self._calculate_stability_assessment(
                object_info['primary_label'], surface_obj['label']
            )
            
            # 6. 使用頻率
            scores['usage_frequency'] = self._calculate_usage_frequency(
                object_info['primary_label'], surface_obj['label']
            )
            
            # 7. 美學和諧
            scores['aesthetic_harmony'] = self._calculate_aesthetic_harmony(
                object_info['primary_label'], surface_obj['label']
            )

            # 計算加權總分
            total_score = sum(scores[key] * self.weights[key]
                              for key in scores.keys())
            total_score = max(0.0, min(1.0, total_score))

            return total_score, scores

        except Exception as e:
            print(f"⚠️ 語意相容性計算失敗: {e}")
            return 0.3, {"error": str(e)}

    def _calculate_clip_similarity(self, object_info: Dict, surface_obj: Dict) -> float:
        """計算基於CLIP的語意相似度"""
        try:
            object_embedding = object_info['embedding']

            # 根據表面類型生成更豐富的描述
            surface_descriptions = {
                "dining table": "a dining table surface for placing food and objects",
                "chair": "a chair seat for sitting or placing small items",
                "couch": "a comfortable couch for sitting and placing personal items",
                "bed": "a bed for sleeping and placing personal belongings",
                "bench": "a bench for sitting and placing bags or books",
                "tv": "a television stand or surface for placing remote controls",
                "toilet": "a toilet area for placing personal hygiene items",
                "sink": "a sink area for placing dishes and cleaning items",
                "refrigerator": "a refrigerator surface for placing kitchen items",
                "microwave": "a microwave surface for placing kitchen utensils",
                "oven": "an oven surface for placing cooking items",
                "toaster": "a toaster area for placing kitchen accessories"
            }
            
            surface_label = surface_obj['label']
            if surface_label in surface_descriptions:
                surface_text = surface_descriptions[surface_label]
            else:
                surface_text = f"a {surface_label} that can hold objects"
            
            surface_embedding = self._get_clip_text_embedding(surface_text)

            # 計算相似度
            similarity = cosine_similarity(
                [object_embedding], [surface_embedding])[0][0]
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

            # 根據物件類型生成更合適的描述
            contexts = []
            
            # 基本描述
            contexts.extend([
                f"a {object_label} sitting on a {surface_label}",
                f"a {object_label} placed on top of a {surface_label}",
                f"a {surface_label} with a {object_label} on it"
            ])
            
            # 食物類特殊描述
            if object_label in ["banana", "apple", "sandwich", "orange", "pizza", "cake", "donut"]:
                contexts.extend([
                    f"eating {object_label} while sitting at a {surface_label}",
                    f"preparing to eat {object_label} from a {surface_label}"
                ])
            
            # 工作用品特殊描述
            if object_label in ["laptop", "book", "keyboard", "mouse"]:
                contexts.extend([
                    f"working with {object_label} on a {surface_label}",
                    f"using {object_label} while sitting at a {surface_label}"
                ])
            
            # 休閒用品特殊描述
            if object_label in ["remote", "cell phone", "book", "teddy bear"]:
                contexts.extend([
                    f"relaxing with {object_label} on a {surface_label}",
                    f"using {object_label} while resting on a {surface_label}"
                ])
            
            # 餐具特殊描述
            if object_label in ["fork", "knife", "spoon", "cup", "wine glass", "bowl"]:
                contexts.extend([
                    f"dining with {object_label} at a {surface_label}",
                    f"having a meal using {object_label} on a {surface_label}"
                ])

            context_scores = []
            for context in contexts:
                context_embedding = self._get_clip_text_embedding(context)

                if scene_embedding is not None:
                    # 與場景嵌入比較
                    score = cosine_similarity([context_embedding], [
                                              scene_embedding])[0][0]
                else:
                    # 使用上下文本身的語意強度
                    neutral_embedding = self._get_clip_text_embedding(
                        "objects in a room")
                    score = cosine_similarity([context_embedding], [
                                              neutral_embedding])[0][0]

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

    def _calculate_stability_assessment(self, object_label: str, surface_label: str) -> float:
        """計算穩定性評估分數"""
        try:
            # 檢查明顯不穩定的組合
            for unstable_obj, unstable_surf in self.unstable_combinations:
                if object_label == unstable_obj and surface_label == unstable_surf:
                    return 0.2
            
            # 液體容器特殊處理
            liquid_containers = ["cup", "wine glass", "bottle", "bowl"]
            if object_label in liquid_containers:
                stable_surfaces = ["dining table", "sink"]
                if surface_label in stable_surfaces:
                    return 1.0
                elif surface_label in ["chair", "bed"]:
                    return 0.4
                else:
                    return 0.6
            
            # 電子產品特殊處理
            electronics = ["laptop", "cell phone", "tv", "remote", "mouse", "keyboard"]
            if object_label in electronics:
                if surface_label == "sink":
                    return 0.1  # 電子產品避免接觸水
                elif surface_label in ["dining table", "chair", "couch", "bed"]:
                    return 0.9
                else:
                    return 0.6
            
            # 基於尺寸的穩定性檢查
            object_size = self._get_size_category(object_label)
            surface_size = self._get_size_category(surface_label)
            
            if object_size and surface_size:
                size_hierarchy = ["tiny", "small", "medium", "large", "huge"]
                obj_idx = size_hierarchy.index(object_size)
                surf_idx = size_hierarchy.index(surface_size)
                
                if obj_idx <= surf_idx:
                    return 1.0 - (surf_idx - obj_idx) * 0.05
                else:
                    return max(0.2, 1.0 - (obj_idx - surf_idx) * 0.3)
            
            return 0.7  # 預設中等穩定性
            
        except Exception as e:
            print(f"⚠️ 穩定性評估計算失敗: {e}")
            return 0.5
    
    def _calculate_usage_frequency(self, object_label: str, surface_label: str) -> float:
        """計算使用頻率分數"""
        try:
            # 檢查常見配對
            pair = (object_label, surface_label)
            if pair in self.common_pairs:
                return self.common_pairs[pair]
            
            # 檢查相似物件的配對
            for (common_obj, common_surf), score in self.common_pairs.items():
                if (self._is_similar_object(object_label, common_obj) and 
                    self._is_similar_object(surface_label, common_surf)):
                    return score * 0.8  # 相似配對給予較低分數
            
            # 基於場景一致性推斷使用頻率
            object_scenes = self._get_object_scenes(object_label)
            surface_scenes = self._get_object_scenes(surface_label)
            common_scenes = object_scenes.intersection(surface_scenes)
            
            if common_scenes:
                return 0.6  # 同場景物件有中等使用頻率
            else:
                return 0.3  # 不同場景物件使用頻率較低
                
        except Exception as e:
            print(f"⚠️ 使用頻率計算失敗: {e}")
            return 0.5
    
    def _calculate_aesthetic_harmony(self, object_label: str, surface_label: str) -> float:
        """計算美學和諧分數"""
        try:
            # 檢查是否屬於同一美學群組
            for group_name, items in self.aesthetic_groups.items():
                if object_label in items and surface_label in items:
                    return 0.9
            
            # 材質和風格匹配
            formal_objects = ["wine glass", "fork", "knife", "spoon"]
            formal_surfaces = ["dining table"]
            
            if object_label in formal_objects and surface_label in formal_surfaces:
                return 0.8
            
            casual_objects = ["remote", "book", "cup", "teddy bear"]
            casual_surfaces = ["couch", "bed", "chair"]
            
            if object_label in casual_objects and surface_label in casual_surfaces:
                return 0.7
            
            # 顏色和材質衝突檢查（簡化版）
            tech_objects = ["laptop", "tv", "cell phone", "mouse", "keyboard"]
            natural_surfaces = ["potted plant"]  # 如果有的話
            
            if object_label in tech_objects and surface_label in natural_surfaces:
                return 0.4  # 科技產品與自然元素搭配度較低
            
            return 0.6  # 預設中等和諧度
            
        except Exception as e:
            print(f"⚠️ 美學和諧計算失敗: {e}")
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
        # 基於COCO類別的語意相似性檢查
        similar_groups = [
            ["cup", "wine glass", "bottle"],
            ["fork", "knife", "spoon"],
            ["banana", "apple", "orange"],
            ["sandwich", "hot dog", "pizza"],
            ["laptop", "keyboard", "mouse"],
            ["chair", "couch", "bench"],
            ["car", "truck", "bus"],
            ["bicycle", "motorcycle"],
            ["baseball bat", "baseball glove", "tennis racket"],
            ["frisbee", "sports ball", "kite"],
            ["backpack", "handbag", "suitcase"]
        ]

        for group in similar_groups:
            if obj1 in group and obj2 in group:
                return True
        return False

    def _is_universal_object(self, object_label: str) -> bool:
        """檢查是否為通用物件（可出現在多個場景）"""
        universal_objects = ["book", "cell phone", "cup", "bottle",
                             "backpack", "clock", "potted plant", "chair", "remote"]
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

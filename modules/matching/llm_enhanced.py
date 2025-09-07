"""
LLM增強匹配系統
整合LLM來增強物件置入決策的準確性和合理性
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from PIL import Image

from modules.llm.client import create_llm_client, LLMClient
from modules.llm.prompts import PromptTemplates
from modules.llm.config import LLMConfig

# 設置日誌
logger = logging.getLogger(__name__)


class LLMObjectFilter:
    """使用LLM篩選偵測到的物件"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化物件篩選器
        
        Args:
            llm_client: 可選的LLM客戶端，如果為None則自動創建
        """
        self.llm_client = llm_client or create_llm_client()
        self.enabled = self.llm_client.is_enabled()
    
    def filter_objects(self, detected_objects: List[Dict], 
                      target_object: str) -> Tuple[List[Dict], str]:
        """
        篩選適合的參考物件
        
        Args:
            detected_objects: 偵測到的物件列表
            target_object: 目標物件
            
        Returns:
            Tuple[List[Dict], str]: (篩選後的物件, 篩選原因)
        """
        if not self.enabled:
            logger.warning("LLM物件篩選功能未啟用，返回原始物件列表")
            return detected_objects, "LLM功能未啟用，未進行篩選"
        
        if not detected_objects:
            return [], "沒有偵測到任何物件"
        
        try:
            print(f"\n🔍 LLM物件篩選 - 目標物件: {target_object}")
            print(f"原始物件數量: {len(detected_objects)}")
            
            # 創建篩選prompt
            prompt = PromptTemplates.format_object_filter_prompt(
                target_object, detected_objects
            )
            
            # 調用LLM
            response = self.llm_client.generate_text(prompt)
            
            if not response:
                logger.warning("LLM篩選無回應，返回原始物件列表")
                return detected_objects, "LLM無回應"
            
            print("\n🤖 LLM篩選原始回應:")
            print("=" * 50)
            print(response)
            print("=" * 50)
            
            # 解析回應
            result = self.llm_client.parse_json_response(response)
            
            if not result:
                logger.warning("LLM篩選回應解析失敗，返回原始物件列表")
                return detected_objects, "回應解析失敗"
            
            # 提取篩選結果
            suitable_objects = result.get('suitable_objects', [])
            removed_objects = result.get('removed_objects', [])
            reasoning = result.get('reasoning', '無具體原因')
            
            # 根據LLM建議篩選物件
            filtered_objects = self._apply_filter_results(
                detected_objects, suitable_objects, removed_objects
            )
            
            print(f"\n✅ 篩選完成:")
            print(f"保留物件: {len(filtered_objects)}")
            print(f"移除物件: {len(detected_objects) - len(filtered_objects)}")
            print(f"篩選原因: {reasoning}")
            
            return filtered_objects, reasoning
            
        except Exception as e:
            logger.error(f"LLM物件篩選失敗: {e}")
            return detected_objects, f"篩選失敗: {str(e)}"
    
    def _apply_filter_results(self, detected_objects: List[Dict],
                             suitable_objects: List[Dict],
                             removed_objects: List[Dict]) -> List[Dict]:
        """
        根據LLM建議應用篩選結果
        
        Args:
            detected_objects: 原始偵測物件
            suitable_objects: LLM認為適合的物件
            removed_objects: LLM認為應移除的物件
            
        Returns:
            List[Dict]: 篩選後的物件列表
        """
        # 提取適合和移除的物件標籤
        suitable_labels = {obj['label'] for obj in suitable_objects}
        removed_labels = {obj['label'] for obj in removed_objects}
        
        filtered_objects = []
        
        for obj in detected_objects:
            obj_label = obj['label']
            
            # 如果明確標記為移除，則跳過
            if obj_label in removed_labels:
                continue
            
            # 如果明確標記為適合，或者不在移除列表中，則保留
            if obj_label in suitable_labels or obj_label not in removed_labels:
                # 添加篩選原因
                for suitable_obj in suitable_objects:
                    if suitable_obj['label'] == obj_label:
                        obj['filter_reason'] = suitable_obj.get('reason', '適合作為參考')
                        break
                else:
                    obj['filter_reason'] = '預設保留'
                
                filtered_objects.append(obj)
        
        return filtered_objects


class EnhancedVLMPromptGenerator:
    """生成增強的VLM分析提示"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化提示生成器
        
        Args:
            llm_client: 可選的LLM客戶端
        """
        self.llm_client = llm_client or create_llm_client()
    
    def generate_enhanced_prompt(self, filtered_objects: List[Dict],
                               target_object: str,
                               filter_reasoning: str) -> str:
        """
        基於篩選結果生成詳細的VLM提示
        
        Args:
            filtered_objects: 篩選後的物件列表
            target_object: 目標物件
            filter_reasoning: LLM篩選的原因
            
        Returns:
            str: 增強的VLM提示
        """
        print(f"\n📝 生成增強VLM提示 - 目標物件: {target_object}")
        
        # 使用模板生成增強prompt
        enhanced_prompt = PromptTemplates.format_enhanced_vlm_prompt(
            target_object, filtered_objects, filter_reasoning
        )
        
        print("✅ 增強提示生成完成")
        return enhanced_prompt


class LLMResultValidator:
    """使用LLM驗證VLM結果的合理性"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化結果驗證器
        
        Args:
            llm_client: 可選的LLM客戶端
        """
        self.llm_client = llm_client or create_llm_client()
        self.enabled = self.llm_client.is_enabled()
    
    def validate_suggestions(self, vlm_suggestions: List[Dict],
                           filtered_objects: List[Dict],
                           target_object: str) -> List[Dict]:
        """
        驗證VLM建議的合理性
        
        Args:
            vlm_suggestions: VLM建議列表
            filtered_objects: 篩選後的物件列表
            target_object: 目標物件
            
        Returns:
            List[Dict]: 包含驗證結果的建議列表
        """
        if not self.enabled:
            logger.warning("LLM驗證功能未啟用，返回原始建議")
            return vlm_suggestions
        
        if not vlm_suggestions:
            return []
        
        print(f"\n🔍 LLM結果驗證 - 驗證{len(vlm_suggestions)}個建議")
        
        validated_suggestions = []
        
        for i, suggestion in enumerate(vlm_suggestions):
            try:
                print(f"\n驗證建議 {i+1}/{len(vlm_suggestions)}")
                
                # 創建驗證prompt
                prompt = PromptTemplates.format_validation_prompt(
                    suggestion, filtered_objects, target_object
                )
                
                # 調用LLM驗證
                response = self.llm_client.generate_text(prompt)
                
                if not response:
                    logger.warning(f"建議{i+1}驗證無回應，保持原始建議")
                    validated_suggestions.append(suggestion)
                    continue
                
                print(f"🤖 驗證回應 {i+1}:")
                print("-" * 30)
                print(response[:200] + "..." if len(response) > 200 else response)
                print("-" * 30)
                
                # 解析驗證結果
                validation_result = self.llm_client.parse_json_response(response)
                
                if validation_result:
                    # 將驗證結果添加到建議中
                    suggestion['llm_validation'] = validation_result
                    suggestion['is_validated'] = validation_result.get('is_reasonable', True)
                    suggestion['validation_confidence'] = validation_result.get('overall_confidence', 0.5)
                else:
                    logger.warning(f"建議{i+1}驗證結果解析失敗")
                    suggestion['is_validated'] = True  # 預設為有效
                    suggestion['validation_confidence'] = 0.5
                
                validated_suggestions.append(suggestion)
                
            except Exception as e:
                logger.error(f"建議{i+1}驗證失敗: {e}")
                suggestion['is_validated'] = True  # 預設為有效
                suggestion['validation_confidence'] = 0.5
                validated_suggestions.append(suggestion)
        
        print(f"\n✅ 驗證完成，{len(validated_suggestions)}個建議已驗證")
        return validated_suggestions


class IntelligentDecisionFusion:
    """融合多種分析結果的智能決策器"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        初始化決策融合器
        
        Args:
            llm_client: 可選的LLM客戶端
        """
        self.llm_client = llm_client or create_llm_client()
        self.enabled = self.llm_client.is_enabled()
    
    def fuse_results(self, semantic_matches: List[Dict],
                    vlm_suggestions: List[Dict],
                    llm_validations: Optional[List[Dict]] = None) -> Dict:
        """
        融合所有分析結果，返回最佳的單一建議
        
        Args:
            semantic_matches: 語意匹配結果
            vlm_suggestions: VLM建議結果
            llm_validations: LLM驗證結果（可選）
            
        Returns:
            Dict: 最終的最佳建議 {reference_object: str, bbox: List[int]}
        """
        print(f"\n🔄 智能決策融合")
        print(f"語意匹配: {len(semantic_matches)} 個")
        print(f"VLM建議: {len(vlm_suggestions)} 個")
        
        best_result = None
        best_score = 0
        
        # 評估VLM建議（優先級最高）
        for suggestion in vlm_suggestions:
            if suggestion.get('is_validated', True):
                score = suggestion.get('confidence', 0)
                # 如果有LLM驗證，調整分數
                if 'llm_validation' in suggestion:
                    validation = suggestion['llm_validation']
                    if validation.get('is_reasonable', True):
                        score *= validation.get('overall_confidence', 1.0)
                    else:
                        score *= 0.5  # 降低不合理建議的分數
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        'reference_object': suggestion.get('reference_object', 'unknown'),
                        'bbox': suggestion.get('bbox', []),
                        'score': score,
                        'source': 'vlm_validated'
                    }
        
        # 如果沒有好的VLM建議，使用語意匹配結果
        if best_result is None and semantic_matches:
            best_semantic = semantic_matches[0]  # 取最佳的語意匹配
            best_result = {
                'reference_object': best_semantic.get('reference_object', 'unknown'),
                'bbox': best_semantic.get('bbox', []),
                'score': best_semantic.get('compatibility_score', 0),
                'source': 'semantic'
            }
        
        if best_result:
            print(f"✅ 最佳建議: {best_result['reference_object']} (分數: {best_result['score']:.3f})")
        else:
            print("❌ 未找到合適的建議")
            best_result = {
                'reference_object': 'none',
                'bbox': [],
                'score': 0,
                'source': 'none'
            }
        
        return best_result


class LLMEnhancedMatcher:
    """主要的LLM增強匹配器 - 統一介面"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化LLM增強匹配器
        
        Args:
            config: 可選的LLM配置
        """
        self.config = config or LLMConfig()
        self.llm_client = create_llm_client()
        
        # 初始化各個組件
        self.object_filter = LLMObjectFilter(self.llm_client)
        self.prompt_generator = EnhancedVLMPromptGenerator(self.llm_client)
        self.result_validator = LLMResultValidator(self.llm_client)
        self.decision_fusion = IntelligentDecisionFusion(self.llm_client)
    
    def is_enabled(self) -> bool:
        """檢查LLM增強功能是否可用"""
        return self.llm_client.is_enabled()
    
    def enhanced_placement_analysis(self, image: np.ndarray, object_label: str,
                                  detected_objects: List[Dict]) -> List[Dict]:
        """
        完整的LLM增強分析流程
        
        Args:
            image: 場景圖片
            object_label: 目標物件標籤
            detected_objects: 偵測到的物件列表
            
        Returns:
            List[Dict]: 最終的增強分析結果
        """
        if not self.is_enabled():
            logger.warning("LLM增強功能未啟用")
            return []
        
        print("\n" + "🚀" * 20 + " LLM增強分析開始 " + "🚀" * 20)
        
        try:
            # 步驟1: 物件篩選
            if self.config.object_filtering:
                filtered_objects, filter_reasoning = self.object_filter.filter_objects(
                    detected_objects, object_label
                )
            else:
                filtered_objects = detected_objects
                filter_reasoning = "物件篩選功能已停用"
            
            # 步驟2: 生成增強VLM提示
            if self.config.prompt_enhancement:
                enhanced_prompt = self.prompt_generator.generate_enhanced_prompt(
                    filtered_objects, object_label, filter_reasoning
                )
            else:
                enhanced_prompt = None
            
            # 步驟3: 調用VLM分析
            from modules.validation.vlm.vlm import create_vlm_helper
            vlm_helper = create_vlm_helper()
            
            if vlm_helper.is_enabled():
                vlm_suggestions = vlm_helper.get_vlm_suggestions(
                    image, object_label, custom_prompt=enhanced_prompt
                )
            else:
                logger.warning("VLM功能未啟用")
                vlm_suggestions = []
            
            # 步驟4: LLM結果驗證
            if self.config.result_validation and vlm_suggestions:
                validated_suggestions = self.result_validator.validate_suggestions(
                    vlm_suggestions, filtered_objects, object_label
                )
            else:
                validated_suggestions = vlm_suggestions
            
            # 步驟5: 獲取語意匹配結果進行融合
            semantic_matches = self._get_semantic_matches(
                object_label, filtered_objects
            )
            
            # 步驟6: 智能決策融合
            if self.config.decision_fusion:
                best_result = self.decision_fusion.fuse_results(
                    semantic_matches, validated_suggestions
                )
            else:
                # 如果沒有融合，選擇最佳的單一結果
                if validated_suggestions:
                    best_suggestion = validated_suggestions[0]
                    best_result = {
                        'reference_object': best_suggestion.get('reference_object', 'unknown'),
                        'bbox': best_suggestion.get('bbox', []),
                        'score': best_suggestion.get('confidence', 0),
                        'source': 'vlm'
                    }
                elif semantic_matches:
                    best_semantic = semantic_matches[0]
                    best_result = {
                        'reference_object': best_semantic.get('reference_object', 'unknown'),
                        'bbox': best_semantic.get('bbox', []),
                        'score': best_semantic.get('compatibility_score', 0),
                        'source': 'semantic'
                    }
                else:
                    best_result = {
                        'reference_object': 'none',
                        'bbox': [],
                        'score': 0,
                        'source': 'none'
                    }
            
            print("\n" + "✅" * 20 + " LLM增強分析完成 " + "✅" * 20)
            print(f"🎯 最終建議: 將 {object_label} 放置在 {best_result['reference_object']}")
            print(f"📍 位置座標: {best_result['bbox']}")
            
            return best_result
            
        except Exception as e:
            logger.error(f"LLM增強分析失敗: {e}")
            return []
    
    def _get_semantic_matches(self, object_label: str, 
                            filtered_objects: List[Dict]) -> List[Dict]:
        """
        獲取語意匹配結果
        
        Args:
            object_label: 目標物件標籤
            filtered_objects: 篩選後的物件列表
            
        Returns:
            List[Dict]: 語意匹配結果
        """
        try:
            from modules.matching.semantic import find_best_placement
            import clip
            import torch
            
            # 載入CLIP模型
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, _ = clip.load("ViT-B/32", device=device)
            
            # 獲取物件嵌入
            with torch.no_grad():
                text_tokens = clip.tokenize([f"a {object_label}"]).to(device)
                text_embedding = clip_model.encode_text(text_tokens)
                object_embedding = text_embedding.cpu().numpy()[0]
            
            # 執行語意匹配
            semantic_matches = find_best_placement(
                object_label=object_label,
                object_embedding=object_embedding,
                background_objects=filtered_objects,
                top_k=3,
                use_vlm=False,
                use_llm_enhanced=False  # 避免遞迴調用
            )
            
            return semantic_matches
            
        except Exception as e:
            logger.error(f"獲取語意匹配結果失敗: {e}")
            return []


# 便利函數
def create_llm_enhanced_matcher(config: Optional[LLMConfig] = None) -> LLMEnhancedMatcher:
    """
    創建LLM增強匹配器實例
    
    Args:
        config: 可選的LLM配置
        
    Returns:
        LLMEnhancedMatcher: LLM增強匹配器實例
    """
    return LLMEnhancedMatcher(config)


def is_llm_enhanced_available() -> bool:
    """
    檢查LLM增強功能是否可用
    
    Returns:
        bool: LLM增強功能是否可用
    """
    matcher = create_llm_enhanced_matcher()
    return matcher.is_enabled()
"""
VLM輔助置入系統
使用Vision Language Model (特別是GEMINI API) 來增強物件置入位置建議
"""

import os
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from PIL import Image
import logging

# 設置日誌
logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. VLM功能將不可用。")


class VLMHelper:
    """VLM輔助器 - 使用GEMINI API提供智能的物件置入建議"""

    def __init__(self, api_key: Optional[str] = None):
        """初始化VLM輔助器"""
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.enabled = GEMINI_AVAILABLE and bool(self.api_key)

        if not self.enabled:
            logger.warning("VLM功能未啟用")
            return

        try:
            genai.configure(api_key=self.api_key)
            # 使用最新的Gemini 2.0模型
            self.client = genai.GenerativeModel("gemini-2.0-flash-exp")
            logger.info("VLM輔助器初始化成功")
        except Exception as e:
            logger.error(f"GEMINI API初始化失敗: {e}")
            self.enabled = False

    def is_enabled(self) -> bool:
        """檢查VLM功能是否可用"""
        return self.enabled

    def get_vlm_suggestions(self, image: Union[np.ndarray, Image.Image],
                            object_label: str,
                            custom_prompt: Optional[str] = None) -> List[Dict]:
        """
        獲取VLM的置入建議

        Args:
            image: 輸入圖片
            object_label: 要放置的物件標籤
            custom_prompt: 可選的自定義prompt (LLM增強生成)

        Returns:
            List[Dict]: VLM建議列表，包含標準化座標
        """
        if not self.enabled:
            return []

        try:
            # 轉換圖片格式
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # 等比例縮放
            width, height = pil_image.size
            if width > 1024 or height > 1024:
                if width > height:
                    new_width = 1024
                    new_height = int(height * (1024 / width))
                else:
                    new_height = 1024
                    new_width = int(width * (1024 / height))
                pil_image = pil_image.resize(
                    (new_width, new_height), Image.Resampling.LANCZOS)

            # 使用自定義prompt或創建標準prompt
            if custom_prompt:
                prompt = custom_prompt
                print("🔧 使用LLM增強prompt")
            else:
                prompt = f"""
請分析這張圖片，為放置一個{object_label}提供3個最佳位置建議。

對每個建議，請提供：
1. 位置描述
2. 選擇原因  
3. 信心分數 (0到1之間)
4. 參考物件
5. 建議的放置區域座標 (使用0-1標準化座標，格式: [x1, y1, x2, y2]，左上角是[0,0])

請以JSON格式回應：
{{
  "suggestions": [
    {{
      "position_description": "在桌子的右上角",
      "reasoning": "這個位置不會阻擋其他物件",
      "confidence": 0.85,
      "reference_object": "dining table",
      "bbox": [0.6, 0.2, 0.8, 0.4]
    }}
  ]
}}
"""
                print("📝 使用標準prompt")

            # 調用GEMINI API (帶重試機制)
            response = self._call_gemini_with_retry(pil_image, prompt)

            if not response or not response.text:
                print("❌ VLM無回應")
                return []

            # 無論如何都打印原始回應
            print("\n" + "="*60)
            print("🤖 VLM原始回應:")
            print("="*60)
            print(response.text)
            print("="*60)

            # 解析JSON回應
            import json
            try:
                # 清理回應文字，移除markdown代碼塊標記
                clean_text = response.text.strip()

                # 移除```json和```標記
                if clean_text.startswith('```json'):
                    clean_text = clean_text[7:]  # 移除```json
                if clean_text.startswith('```'):
                    clean_text = clean_text[3:]   # 移除```
                if clean_text.endswith('```'):
                    clean_text = clean_text[:-3]  # 移除結尾的```

                clean_text = clean_text.strip()

                print(f"\n🧹 清理後的JSON:")
                print("-" * 40)
                print(clean_text)
                print("-" * 40)

                result = json.loads(clean_text)
                suggestions = result.get("suggestions", [])

                # 驗證格式
                valid_suggestions = self._validate_suggestions(suggestions)
                print(f"\n✅ 成功解析{len(valid_suggestions)}個有效VLM建議")
                return valid_suggestions

            except json.JSONDecodeError as e:
                print(f"❌ VLM回應JSON解析失敗: {e}")
                print("🔧 嘗試更激進的清理...")

                # 更激進的文字清理
                import re

                # 尋找JSON對象
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        print(f"🎯 找到JSON對象:")
                        print(
                            json_str[:200] + "..." if len(json_str) > 200 else json_str)

                        result = json.loads(json_str)
                        suggestions = result.get("suggestions", [])

                        if suggestions:
                            print(f"✅ 成功解析{len(suggestions)}個建議")
                            return self._validate_suggestions(suggestions)
                    except json.JSONDecodeError:
                        pass

                print("嘗試從文字中提取信息...")
                # 如果JSON解析失敗，嘗試從文字中提取基本信息
                fallback_suggestions = self._extract_fallback_suggestions(
                    response.text)
                if fallback_suggestions:
                    print(f"✅ 從文字中提取了{len(fallback_suggestions)}個建議")
                    return fallback_suggestions

                return []

        except Exception as e:
            logger.error(f"VLM建議生成失敗: {e}")
            return []

    def enhance_matches(self, image: Union[np.ndarray, Image.Image], object_label: str,
                        existing_matches: List[Dict]) -> List[Dict]:
        """
        使用VLM增強現有的匹配結果

        Args:
            image: 輸入圖片
            object_label: 要放置的物件標籤
            existing_matches: 現有的語意匹配結果

        Returns:
            List[Dict]: 增強後的匹配結果
        """
        if not self.enabled:
            logger.debug("VLM功能未啟用，返回原始匹配結果")
            return existing_matches

        try:
            print("\n" + "🔄" * 20 + " VLM增強處理 " + "🔄" * 20)
            # 獲取VLM建議
            vlm_suggestions = self.get_vlm_suggestions(image, object_label)

            if not vlm_suggestions:
                print("⚠️ VLM未返回建議，使用原始匹配結果")
                return existing_matches

            # 將VLM建議轉換為與現有格式相容的格式
            vlm_matches = []
            image_shape = image.shape if isinstance(
                image, np.ndarray) else (image.height, image.width)

            for suggestion in vlm_suggestions:
                vlm_match = {
                    'reference_object': suggestion['reference_object'],
                    'placement_type': 'vlm_suggested',  # 標記為VLM建議
                    'bbox': self._convert_normalized_to_pixel_bbox(suggestion['bbox'], image_shape),
                    # 使用VLM的confidence作為分數
                    'compatibility_score': suggestion['confidence'],
                    'detailed_scores': {
                        'vlm_confidence': suggestion['confidence'],
                        'vlm_reasoning': suggestion['reasoning'],
                        'vlm_description': suggestion['position_description']
                    },
                    'rank': 0,  # 暫時設為0，後面重新排序
                    'source': 'vlm',  # 標記來源
                    'object_id': f"vlm_{suggestion['reference_object']}",
                    'object_confidence': suggestion['confidence']
                }
                vlm_matches.append(vlm_match)

            # 融合兩種結果
            all_matches = existing_matches.copy()

            # 為現有匹配添加來源標記
            for match in all_matches:
                match['source'] = 'semantic'

            # 添加VLM匹配
            all_matches.extend(vlm_matches)

            # 重新排序（按compatibility_score降序）
            all_matches.sort(
                key=lambda x: x['compatibility_score'], reverse=True)

            # 重新分配排名
            for rank, match in enumerate(all_matches, 1):
                match['rank'] = rank

            # 保持原來的數量，但優先顯示高分的結果
            result_count = len(existing_matches)
            enhanced_matches = all_matches[:result_count]

            logger.info(
                f"VLM增強完成：融合了{len(vlm_matches)}個VLM建議和{len(existing_matches)}個語意匹配")
            return enhanced_matches

        except Exception as e:
            logger.error(f"VLM增強失敗: {e}")
            return existing_matches

    def _convert_normalized_to_pixel_bbox(self, normalized_bbox: List[float],
                                          image_shape: Tuple[int, int]) -> List[int]:
        """
        將標準化座標轉換為像素座標

        Args:
            normalized_bbox: 標準化座標 [x1, y1, x2, y2] (0-1)
            image_shape: 圖片尺寸 (height, width) 或 (height, width, channels)

        Returns:
            List[int]: 像素座標 [x1, y1, x2, y2]
        """
        if len(image_shape) >= 2:
            height, width = image_shape[:2]
        else:
            # 如果是PIL Image的(width, height)格式
            width, height = image_shape

        x1, y1, x2, y2 = normalized_bbox
        return [
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height)
        ]

    def _call_gemini_with_retry(self, image, prompt, max_retries=3):
        """
        帶重試機制的GEMINI API調用

        Args:
            image: PIL圖片
            prompt: 提示文字
            max_retries: 最大重試次數

        Returns:
            API回應或None
        """
        import time

        for attempt in range(max_retries):
            try:
                response = self.client.generate_content([image, prompt])
                return response

            except Exception as e:
                error_msg = str(e)

                # 檢查是否是額度限制錯誤
                if "429" in error_msg or "quota" in error_msg.lower():
                    if "retry_delay" in error_msg:
                        # 提取建議的等待時間
                        import re
                        delay_match = re.search(
                            r'retry_delay.*?seconds: (\d+)', error_msg)
                        if delay_match:
                            delay = int(delay_match.group(1))
                        else:
                            delay = 30  # 預設等待30秒
                    else:
                        delay = 30

                    if attempt < max_retries - 1:
                        logger.warning(
                            f"API額度限制，等待{delay}秒後重試... (嘗試 {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("API額度已用完，請稍後再試或升級計劃")
                        return None
                else:
                    # 其他錯誤，短暫等待後重試
                    if attempt < max_retries - 1:
                        logger.warning(f"API調用失敗: {e}，2秒後重試...")
                        time.sleep(2)
                        continue
                    else:
                        logger.error(f"API調用最終失敗: {e}")
                        return None

        return None

    def _extract_fallback_suggestions(self, text: str) -> List[Dict]:
        """
        從非JSON格式的文字中提取建議信息

        Args:
            text: VLM的原始回應文字

        Returns:
            List[Dict]: 提取的建議列表
        """
        suggestions = []

        # 簡單的文字解析邏輯
        lines = text.split('\n')
        current_suggestion = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 嘗試提取關鍵信息
            if '位置' in line or 'position' in line.lower():
                current_suggestion['position_description'] = line
            elif '原因' in line or 'reason' in line.lower():
                current_suggestion['reasoning'] = line
            elif '信心' in line or 'confidence' in line.lower():
                # 嘗試提取數字
                import re
                numbers = re.findall(r'0\.\d+|\d+\.\d+', line)
                if numbers:
                    current_suggestion['confidence'] = float(numbers[0])
            elif '座標' in line or 'bbox' in line.lower() or '[' in line:
                # 嘗試提取座標
                import re
                coords = re.findall(r'0\.\d+|\d+\.\d+', line)
                if len(coords) >= 4:
                    current_suggestion['bbox'] = [float(x) for x in coords[:4]]

            # 如果收集到足夠信息，添加到建議列表
            if len(current_suggestion) >= 3:  # 至少有3個字段
                # 填充缺失字段
                if 'reference_object' not in current_suggestion:
                    current_suggestion['reference_object'] = 'table'  # 預設值
                if 'confidence' not in current_suggestion:
                    current_suggestion['confidence'] = 0.5  # 預設值
                if 'bbox' not in current_suggestion:
                    current_suggestion['bbox'] = [0.3, 0.3, 0.7, 0.7]  # 預設值

                suggestions.append(current_suggestion.copy())
                current_suggestion = {}

                if len(suggestions) >= 3:  # 最多3個建議
                    break

        return suggestions

    def _validate_suggestions(self, suggestions: List[Dict]) -> List[Dict]:
        """
        驗證和修復建議格式

        Args:
            suggestions: 原始建議列表

        Returns:
            List[Dict]: 驗證後的建議列表
        """
        valid_suggestions = []

        for i, suggestion in enumerate(suggestions):
            try:
                # 檢查必需字段
                required_fields = [
                    "position_description", "reasoning", "confidence", "reference_object", "bbox"]

                # 修復缺失字段
                if "position_description" not in suggestion:
                    suggestion["position_description"] = f"Position {i+1}"
                if "reasoning" not in suggestion:
                    suggestion["reasoning"] = "No reasoning provided"
                if "confidence" not in suggestion:
                    suggestion["confidence"] = 0.5
                if "reference_object" not in suggestion:
                    suggestion["reference_object"] = "unknown"
                if "bbox" not in suggestion:
                    suggestion["bbox"] = [0.3, 0.3, 0.7, 0.7]

                # 驗證和修復bbox
                bbox = suggestion["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    print(f"⚠️ 修復建議{i+1}的bbox格式")
                    suggestion["bbox"] = [0.3, 0.3, 0.7, 0.7]
                else:
                    # 確保座標在0-1範圍內
                    fixed_bbox = []
                    for coord in bbox:
                        if isinstance(coord, (int, float)):
                            fixed_bbox.append(max(0.0, min(1.0, float(coord))))
                        else:
                            fixed_bbox.append(0.5)
                    suggestion["bbox"] = fixed_bbox

                # 驗證confidence
                if not isinstance(suggestion["confidence"], (int, float)):
                    suggestion["confidence"] = 0.5
                else:
                    suggestion["confidence"] = max(
                        0.0, min(1.0, float(suggestion["confidence"])))

                valid_suggestions.append(suggestion)

            except Exception as e:
                print(f"⚠️ 跳過無效建議{i+1}: {e}")
                continue

        return valid_suggestions


# 便利函數
def create_vlm_helper(api_key: Optional[str] = None) -> VLMHelper:
    """創建VLM輔助器實例"""
    return VLMHelper(api_key=api_key)


def is_vlm_available() -> bool:
    """檢查VLM功能是否可用"""
    return GEMINI_AVAILABLE and bool(os.getenv("GEMINI_API_KEY"))

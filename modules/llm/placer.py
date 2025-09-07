"""
LLM物件放置器 - 主要運行邏輯
"""

import base64
import io
from typing import List, Dict, Optional, Union
import numpy as np
from PIL import Image
from .client import LLMClient
from .config import LLMConfig
from .prompts import PromptTemplates


class LLMPlacer:
    """LLM物件放置器 - 主要功能類"""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化LLM放置器

        Args:
            config: 可選的LLM配置
        """
        self.config = config or LLMConfig()
        self.client = LLMClient(self.config)

    def is_enabled(self) -> bool:
        """檢查LLM功能是否可用"""
        return self.client.is_enabled()

    def suggest_placement_from_images(self, background_image: Union[np.ndarray, Image.Image],
                                      target_object: Union[np.ndarray, Image.Image]) -> Optional[Dict]:
        """
        基於背景圖片和目標物件圖片分析，提供放置位置建議

        Args:
            background_image: 背景圖片 (numpy array 或 PIL Image)
            target_object: 要插入的物件圖片 (numpy array 或 PIL Image)

        Returns:
            Optional[Dict]: 放置建議結果，包含位置描述和推理
        """
        if not self.is_enabled():
            print("❌ LLM功能未啟用")
            return None

        try:
            print("🖼️ 分析背景圖片和目標物件，提供放置建議...")

            # 轉換圖片格式
            bg_image = self._convert_to_pil(background_image)
            obj_image = self._convert_to_pil(target_object)

            # 生成prompt
            prompt = PromptTemplates.format_image_placement_prompt()

            # 調用Gemini API，同時傳入背景圖片和物件圖片
            response = self._call_gemini_with_images_for_placement(
                prompt, bg_image, obj_image)

            if not response:
                print("❌ LLM無回應")
                return None

            print("🤖 LLM分析回應:")
            print("=" * 50)
            print(response)
            print("=" * 50)

            # 解析JSON回應
            result = self.client.parse_json_response(response)

            if result and 'placement_suggestion' in result:
                suggestion = result['placement_suggestion']
                print(
                    f"✅ 建議位置: {suggestion.get('position_description', '未知')}")
                print(
                    f"📍 參考物件: {', '.join(suggestion.get('reference_objects', []))}")
                print(f"🎯 信心度: {suggestion.get('confidence', 0):.2f}")
                return result
            else:
                print("❌ 無法解析LLM回應")
                return None

        except Exception as e:
            print(f"❌ 圖片分析失敗: {e}")
            return None

    def suggest_object_size(self, background_image: Union[np.ndarray, Image.Image],
                            object_to_insert: Union[np.ndarray, Image.Image],
                            detected_objects: List[Dict]) -> Optional[Dict]:
        """
        根據場景中已偵測物件的相對大小，建議插入物件的適當尺寸

        Args:
            background_image: 背景場景圖片
            object_to_insert: 要插入的物件圖片
            detected_objects: 已偵測到的物件列表，每個包含 {'label': str, 'bbox': [x1,y1,x2,y2]}

        Returns:
            Optional[Dict]: 大小建議結果，包含縮放比例和推理
        """
        if not self.is_enabled():
            print("❌ LLM功能未啟用")
            return None

        try:
            print("📏 分析物件的適當大小...")

            # 轉換圖片格式
            bg_image = self._convert_to_pil(background_image)
            obj_image = self._convert_to_pil(object_to_insert)

            # 生成大小分析prompt
            prompt = PromptTemplates.format_size_analysis_prompt(
                detected_objects)

            # 調用Gemini API，同時傳入背景圖片和物件圖片
            response = self._call_gemini_with_images(
                prompt, bg_image, obj_image)

            if not response:
                print("❌ LLM無回應")
                return None

            print("🤖 LLM大小分析回應:")
            print("=" * 50)
            print(response)
            print("=" * 50)

            # 解析JSON回應
            result = self.client.parse_json_response(response)

            if result and 'size_suggestion' in result:
                suggestion = result['size_suggestion']
                print(f"📐 建議寬度比例: {suggestion.get('width_scale', 1.0):.2f}")
                print(f"📐 建議高度比例: {suggestion.get('height_scale', 1.0):.2f}")
                print(f"📍 參考物件: {suggestion.get('reference_object', '未知')}")
                print(f"🎯 信心度: {suggestion.get('confidence', 0):.2f}")
                return result
            else:
                print("❌ 無法解析LLM大小分析回應")
                return None

        except Exception as e:
            print(f"❌ 大小分析失敗: {e}")
            return None

    def _convert_to_pil(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """轉換圖片為PIL格式"""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
        return image

    def _call_gemini_with_image(self, prompt: str, image: Image.Image) -> Optional[str]:
        """
        調用Gemini API處理圖片和文字

        Args:
            prompt: 文字提示
            image: PIL圖片

        Returns:
            Optional[str]: LLM回應
        """
        try:
            # 準備圖片和文字內容
            content = [prompt, image]

            # 調用Gemini API
            response = self.client.client.generate_content(content)
            return response.text if response else None

        except Exception as e:
            print(f"❌ Gemini圖片分析調用失敗: {e}")
            return None

    def _call_gemini_with_images(self, prompt: str, bg_image: Image.Image, obj_image: Image.Image) -> Optional[str]:
        """
        調用Gemini API處理多張圖片和文字 (用於大小分析)

        Args:
            prompt: 文字提示
            bg_image: 背景圖片
            obj_image: 物件圖片

        Returns:
            Optional[str]: LLM回應
        """
        try:
            # 準備多張圖片和文字內容
            content = [
                "背景場景圖片:",
                bg_image,
                "要插入的物件圖片:",
                obj_image,
                prompt
            ]

            # 調用Gemini API
            response = self.client.client.generate_content(content)
            return response.text if response else None

        except Exception as e:
            print(f"❌ Gemini多圖片分析調用失敗: {e}")
            return None

    def _call_gemini_with_images_for_placement(self, prompt: str, bg_image: Image.Image, obj_image: Image.Image) -> Optional[str]:
        """
        調用Gemini API處理多張圖片和文字 (用於位置建議)

        Args:
            prompt: 文字提示
            bg_image: 背景圖片
            obj_image: 物件圖片

        Returns:
            Optional[str]: LLM回應
        """
        try:
            # 準備多張圖片和文字內容
            content = [
                "背景場景圖片:",
                bg_image,
                "要插入的物件圖片:",
                obj_image,
                prompt
            ]

            # 調用Gemini API
            response = self.client.client.generate_content(content)
            return response.text if response else None

        except Exception as e:
            print(f"❌ Gemini位置分析調用失敗: {e}")
            return None

    def draw_placement_box(self, background_image: Union[np.ndarray, Image.Image],
                           object_to_insert: Union[np.ndarray, Image.Image],
                           placement_result: Dict,
                           size_result: Dict,
                           cumulative_scale: Dict = None) -> Optional[Image.Image]:
        """
        根據位置和大小建議，讓Gemini在圖片上畫出建議的插入位置框

        Args:
            background_image: 背景場景圖片
            object_to_insert: 要插入的物件圖片
            placement_result: 位置建議結果
            size_result: 大小建議結果

        Returns:
            Optional[Image.Image]: 畫有位置框的圖片
        """
        if not self.is_enabled():
            print("❌ LLM功能未啟用")
            return None

        try:
            print("🎯 請Gemini畫出建議的插入位置框...")

            # 轉換圖片格式
            bg_image = self._convert_to_pil(background_image)
            obj_image = self._convert_to_pil(object_to_insert)

            # 生成畫框prompt
            prompt = PromptTemplates.format_draw_box_prompt(
                placement_result, size_result)

            # 調用Gemini API
            response = self._call_gemini_with_images_for_drawing(
                prompt, bg_image, obj_image)

            if not response:
                print("❌ LLM無回應")
                return None

            print("🤖 LLM畫框回應:")
            print("=" * 50)
            print(response)
            print("=" * 50)

            # 檢查是否有來自調整建議的標準化座標
            if (placement_result and 'placement_suggestion' in placement_result and 
                'x_normalized' in placement_result['placement_suggestion']):
                # 使用調整建議中的標準化座標
                suggestion = placement_result['placement_suggestion']
                x_norm = suggestion.get('x_normalized', 0.5)
                y_norm = suggestion.get('y_normalized', 0.5)
                
                # 轉換為像素座標（物件底部中心）
                x = int(x_norm * bg_image.width)
                y = int(y_norm * bg_image.height)
                
                print(f"📍 使用調整建議座標: ({x_norm:.2f}, {y_norm:.2f}) -> ({x}, {y})")
            else:
                # 解析LLM回應中的座標信息
                result = self.client.parse_json_response(response)
                
                if result and 'placement_box' in result:
                    box_info = result['placement_box']
                    
                    # 處理標準化座標
                    x_norm = box_info.get('x_normalized', 0.5)
                    y_norm = box_info.get('y_normalized', 0.5)
                    
                    # 轉換為像素座標
                    x = int(x_norm * bg_image.width)
                    y = int(y_norm * bg_image.height)
                    
                    print(f"📍 使用LLM建議座標: ({x_norm:.2f}, {y_norm:.2f}) -> ({x}, {y})")
                else:
                    print("❌ 無法解析座標信息，使用預設位置")
                    x, y = bg_image.width // 2, bg_image.height // 2
            
            # 使用累積縮放比例
            if cumulative_scale:
                # 直接使用累積縮放比例
                final_width_scale = cumulative_scale.get('width', 1.0)
                final_height_scale = cumulative_scale.get('height', 1.0)
            else:
                # 如果沒有累積縮放，使用當前建議
                size_suggestion = size_result.get('size_suggestion', {}) if size_result else {}
                final_width_scale = size_suggestion.get('width_scale', 1.0)
                final_height_scale = size_suggestion.get('height_scale', 1.0)
            
            # 調整物件圖片大小
            obj_width = int(obj_image.width * final_width_scale)
            obj_height = int(obj_image.height * final_height_scale)
            resized_obj = obj_image.resize((obj_width, obj_height), Image.Resampling.LANCZOS)
            
            # 創建背景圖片的副本
            result_image = bg_image.copy()
            
            # 將調整大小後的物件貼到背景上
            # 座標是底部中心，需要轉換為左上角座標
            paste_x = x - obj_width // 2  # 中心對齊
            paste_y = y - obj_height      # 底部對齊
            
            # 確保座標不會超出圖片邊界
            paste_x = max(0, min(paste_x, result_image.width - obj_width))
            paste_y = max(0, min(paste_y, result_image.height - obj_height))
            
            # 處理背景去除和透明度
            if resized_obj.mode == 'RGBA':
                # 如果已經是RGBA，直接使用
                result_image.paste(resized_obj, (paste_x, paste_y), resized_obj)
            else:
                # 轉換為RGBA並去除背景
                resized_obj = resized_obj.convert('RGBA')
                
                # 去除綠色背景（或其他單色背景）
                data = resized_obj.getdata()
                new_data = []
                
                resized_obj.putdata(new_data)
                result_image.paste(resized_obj, (paste_x, paste_y), resized_obj)
            
            print(f"✅ 已插入物件: 位置({paste_x}, {paste_y}), 大小({obj_width}x{obj_height}), 縮放({final_width_scale:.2f}x{final_height_scale:.2f})")
            return result_image

        except Exception as e:
            print(f"❌ 畫框失敗: {e}")
            return None

    def refine_placement_and_size(self, boxed_image: Image.Image,
                                  object_to_insert: Union[np.ndarray, Image.Image]) -> Optional[Dict]:
        """
        將畫好框的結果再次餵給LLM，讓它分析框的位置和大小並給出調整建議

        Args:
            boxed_image: 畫有位置框的圖片
            object_to_insert: 要插入的物件圖片

        Returns:
            Optional[Dict]: 包含調整後的位置和大小建議
        """
        if not self.is_enabled():
            print("❌ LLM功能未啟用")
            return None

        try:
            print("🔄 請LLM分析框的位置和大小並給出調整建議...")

            # 轉換圖片格式
            obj_image = self._convert_to_pil(object_to_insert)

            # 生成調整prompt
            prompt = PromptTemplates.format_refinement_prompt()

            # 調用Gemini API
            response = self._call_gemini_with_images_for_refinement(
                prompt, boxed_image, obj_image)

            if not response:
                print("❌ LLM無回應")
                return None

            print("🤖 LLM調整建議回應:")
            print("=" * 50)
            print(response)
            print("=" * 50)

            # 解析JSON回應
            result = self.client.parse_json_response(response)

            if result:
                # 檢查是否包含位置建議
                if 'placement_suggestion' in result:
                    placement = result['placement_suggestion']
                    print(
                        f"✅ 調整後位置: {placement.get('position_description', '未知')}")
                    print(f"🎯 位置信心度: {placement.get('confidence', 0):.2f}")

                # 檢查是否包含大小建議
                if 'size_suggestion' in result:
                    size = result['size_suggestion']
                    print(
                        f"📐 調整後大小: {size.get('width_scale', 1.0):.2f} x {size.get('height_scale', 1.0):.2f}")
                    print(f"🎯 大小信心度: {size.get('confidence', 0):.2f}")

                return result
            else:
                print("❌ 無法解析LLM調整建議")
                return None

        except Exception as e:
            print(f"❌ 調整建議失敗: {e}")
            return None

    def _call_gemini_with_images_for_drawing(self, prompt: str, bg_image: Image.Image, obj_image: Image.Image) -> Optional[str]:
        """
        調用Gemini API處理畫框請求

        Args:
            prompt: 文字提示
            bg_image: 背景圖片
            obj_image: 物件圖片

        Returns:
            Optional[str]: LLM回應
        """
        try:
            content = [
                "背景場景圖片:",
                bg_image,
                "要插入的物件圖片:",
                obj_image,
                prompt
            ]

            response = self.client.client.generate_content(content)
            return response.text if response else None

        except Exception as e:
            print(f"❌ Gemini畫框調用失敗: {e}")
            return None

    def _call_gemini_with_images_for_refinement(self, prompt: str, boxed_image: Image.Image, obj_image: Image.Image) -> Optional[str]:
        """
        調用Gemini API處理調整建議請求

        Args:
            prompt: 文字提示
            boxed_image: 畫有框的圖片
            obj_image: 物件圖片

        Returns:
            Optional[str]: LLM回應
        """
        try:
            content = [
                "畫有建議位置框的場景圖片:",
                boxed_image,
                "要插入的物件圖片:",
                obj_image,
                prompt
            ]

            response = self.client.client.generate_content(content)
            return response.text if response else None

        except Exception as e:
            print(f"❌ Gemini調整建議調用失敗: {e}")
            return None

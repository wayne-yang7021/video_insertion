"""
LLM Prompt模板管理
"""

from typing import List, Dict


class PromptTemplates:
    """Prompt模板管理類"""

    # 基本放置分析Prompt
    BASIC_PLACEMENT_PROMPT = """
請分析以下場景，為 {target_object} 找出最適合的放置位置。

場景中的物件：
{object_list}

請考慮：
1. 物理穩定性 - 確保有適當支撐
2. 使用便利性 - 符合使用習慣
3. 安全性 - 避免危險位置

請以JSON格式回應：
{{
  "best_placement": {{
    "reference_object": "參考物件名稱",
    "reasoning": "選擇原因",
    "confidence": 0.0-1.0
  }}
}}
"""

    # 圖片分析放置建議Prompt
    IMAGE_PLACEMENT_PROMPT = """
請分析背景場景圖片和要插入的物件圖片，為這個物件提供最佳的放置位置建議。

請仔細觀察：
1. 背景場景中的物件和空間佈局
2. 要插入物件的特性和用途
3. 兩者之間的合理搭配關係

要求：
1. 物理合理性 - 確保有適當的支撐表面
2. 空間關係 - 用相對位置描述（例如：在桌子和椅子之間、在沙發前方、在書架旁邊）
3. 使用便利性 - 考慮實際使用場景
4. 視覺協調 - 與環境和諧

請以JSON格式回應：
{{
  "placement_suggestion": {{
    "position_description": "詳細的相對位置描述，例如：在桌子和椅子之間的地板上",
    "reference_objects": ["主要參考物件1", "主要參考物件2"],
    "reasoning": "選擇這個位置的詳細原因",
    "confidence": 0.0-1.0,
    "alternative_positions": [
      {{
        "position_description": "備選位置描述",
        "reasoning": "備選原因"
      }}
    ]
  }}
}}
"""

    @classmethod
    def format_placement_prompt(cls, target_object: str, detected_objects: List[Dict]) -> str:
        """
        格式化放置分析prompt

        Args:
            target_object: 目標物件
            detected_objects: 偵測到的物件列表

        Returns:
            str: 格式化的prompt
        """
        object_list = "\n".join([
            f"- {obj['label']} (信心度: {obj.get('confidence', 1.0):.2f})"
            for obj in detected_objects
        ])

        return cls.BASIC_PLACEMENT_PROMPT.format(
            target_object=target_object,
            object_list=object_list
        )

    @classmethod
    def format_image_placement_prompt(cls) -> str:
        """
        格式化圖片分析放置prompt

        Returns:
            str: 格式化的prompt
        """
        return cls.IMAGE_PLACEMENT_PROMPT

    # 大小分析Prompt
    SIZE_ANALYSIS_PROMPT = """
請分析背景場景圖片和要插入的物件圖片，為這個物件提供適當的大小建議。

場景中已偵測到的物件：
{detected_objects_info}

請根據以下因素分析：
1. 識別要插入的物件類型
2. 真實世界中的物件比例關係
3. 場景中相似或相關物件的大小作為參考
4. 透視效果和深度感知
5. 整體視覺協調性

分析要求：
- 仔細觀察要插入的物件圖片，識別其類型和特徵
- 比較要插入物件與場景中物件的相對大小
- 考慮物件在真實世界中的典型尺寸比例
- 提供合理的縮放建議（0.1-2.0倍）

請以JSON格式回應：
{{
  "size_suggestion": {{
    "object_type": "識別出的物件類型",
    "width_scale": 0.8,
    "height_scale": 0.8,
    "reference_object": "用作大小參考的場景物件",
    "reasoning": "詳細的大小調整原因，包括與參考物件的比較",
    "confidence": 0.0-1.0,
    "suggested_pixel_size": {{
      "width": 120,
      "height": 150
    }},
    "alternative_scales": [
      {{
        "width_scale": 0.6,
        "height_scale": 0.6,
        "reasoning": "更保守的大小選擇"
      }}
    ]
  }}
}}
"""

    @classmethod
    def format_size_analysis_prompt(cls, detected_objects: List[Dict]) -> str:
        """
        格式化大小分析prompt

        Args:
            detected_objects: 偵測到的物件列表，包含bbox信息

        Returns:
            str: 格式化的prompt
        """
        objects_info = []
        for obj in detected_objects:
            bbox = obj.get('bbox', [0, 0, 0, 0])
            width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
            height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0

            objects_info.append(
                f"- {obj['label']}: bbox {bbox} "
                f"(寬度: {width:.3f}, 高度: {height:.3f}, "
                f"信心度: {obj.get('confidence', 1.0):.2f})"
            )

        detected_objects_info = "\n".join(objects_info)

        return cls.SIZE_ANALYSIS_PROMPT.format(
            detected_objects_info=detected_objects_info
        )

    # 畫框Prompt
    DRAW_BOX_PROMPT = """
根據之前的位置和大小建議，請在背景場景圖片上標示出建議的插入位置。

之前的位置建議：
{placement_info}

之前的大小建議：
{size_info}

重要說明：
- 位置座標應該是物件底部中心的位置
- 使用標準化座標（0.0-1.0），其中(0,0)是圖片左上角，(1,1)是右下角
- 例如：x=0.5表示水平居中，y=0.8表示靠近底部

請以JSON格式回應：
{{
  "placement_box": {{
    "x_normalized": 0.4,
    "y_normalized": 0.6,
    "width_scale": 0.3,
    "height_scale": 0.3,
    "description": "物件底部中心位置的描述",
    "confidence": 0.9
  }}
}}
"""

    # 調整建議Prompt
    REFINEMENT_PROMPT = """
請觀察這張已經標示了建議位置框的場景圖片和要插入的物件圖片。

請分析：
1. 框的位置是否合適？
2. 框的大小是否與物件和場景協調？
3. 是否需要調整位置或大小？

重要說明：
- 請直接給出新的標準化位置座標（0.0-1.0）
- 位置是指物件底部中心的位置
- 大小請給出絕對的縮放比例（不是相對調整）
- 每次調整都應該有明顯的改變

請以JSON格式回應：
{{
  "placement_suggestion": {{
    "x_normalized": 0.45,
    "y_normalized": 0.65,
    "position_description": "調整後的位置描述",
    "reference_objects": ["參考物件1", "參考物件2"],
    "reasoning": "位置調整原因",
    "confidence": 0.0-1.0
  }},
  "size_suggestion": {{
    "object_type": "識別出的物件類型",
    "width_scale": 0.8,
    "height_scale": 0.8,
    "reference_object": "參考物件",
    "reasoning": "大小調整原因（絕對縮放比例）",
    "confidence": 0.0-1.0
  }}
}}
"""

    @classmethod
    def format_draw_box_prompt(cls, placement_result: Dict, size_result: Dict) -> str:
        """
        格式化畫框prompt

        Args:
            placement_result: 位置建議結果
            size_result: 大小建議結果

        Returns:
            str: 格式化的prompt
        """
        placement_info = ""
        if placement_result and 'placement_suggestion' in placement_result:
            p = placement_result['placement_suggestion']
            placement_info = f"位置: {p.get('position_description', '未知')}\n原因: {p.get('reasoning', '無')}\n信心度: {p.get('confidence', 0):.2f}"

        size_info = ""
        if size_result and 'size_suggestion' in size_result:
            s = size_result['size_suggestion']
            size_info = f"寬度縮放: {s.get('width_scale', 1.0):.2f}\n高度縮放: {s.get('height_scale', 1.0):.2f}\n參考物件: {s.get('reference_object', '未知')}\n原因: {s.get('reasoning', '無')}"

        return cls.DRAW_BOX_PROMPT.format(
            placement_info=placement_info,
            size_info=size_info
        )

    @classmethod
    def format_refinement_prompt(cls) -> str:
        """
        格式化調整建議prompt

        Returns:
            str: 格式化的prompt
        """
        return cls.REFINEMENT_PROMPT

# LLM增強匹配系統使用指南

## 概述

LLM增強匹配系統是一個多層智能判斷系統，整合了大語言模型(LLM)來增強物件置入決策的準確性和合理性。系統通過物件篩選、VLM提示增強、結果驗證和智能融合四個階段，提供最佳的物件放置建議。

## 系統架構

```
Detectron偵測 → LLM物件篩選 → 增強VLM提示 → VLM分析 → LLM結果驗證 → 智能融合決策
```

## 快速開始

### 1. 安裝依賴

```bash
pip install google-generativeai python-dotenv openai anthropic
```

### 2. 設置API密鑰

在`.env`文件中設置：

```bash
GEMINI_API_KEY=your_gemini_api_key_here
# 可選：
OPENAI_API_KEY=your_openai_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
```

### 3. 基本使用

```python
from modules.matching.semantic import find_best_placement
import numpy as np

# 使用LLM增強分析
enhanced_matches = find_best_placement(
    object_label="cup",
    object_embedding=your_clip_embedding,
    background_objects=detected_objects,
    image=scene_image,
    use_llm_enhanced=True  # 啟用LLM增強
)
```

### 4. 運行測試

```bash
# 比較三種方法的效果
python tests/run_llm_enhanced_test.py
```

## 詳細使用

### LLM增強匹配器

```python
from modules.matching.llm_enhanced import create_llm_enhanced_matcher

# 創建LLM增強匹配器
matcher = create_llm_enhanced_matcher()

# 檢查是否可用
if matcher.is_enabled():
    # 執行完整的增強分析
    results = matcher.enhanced_placement_analysis(
        image=scene_image,
        object_label="cup",
        detected_objects=background_objects
    )
```

### 單獨使用各個組件

#### 1. 物件篩選

```python
from modules.matching.llm_enhanced import LLMObjectFilter

filter = LLMObjectFilter()
filtered_objects, reasoning = filter.filter_objects(
    detected_objects, "cup"
)
```

#### 2. VLM提示增強

```python
from modules.matching.llm_enhanced import EnhancedVLMPromptGenerator

generator = EnhancedVLMPromptGenerator()
enhanced_prompt = generator.generate_enhanced_prompt(
    filtered_objects, "cup", filter_reasoning
)
```

#### 3. 結果驗證

```python
from modules.matching.llm_enhanced import LLMResultValidator

validator = LLMResultValidator()
validated_suggestions = validator.validate_suggestions(
    vlm_suggestions, filtered_objects, "cup"
)
```

## 配置選項

### 基本配置

```python
from modules.llm.config import LLMConfig

config = LLMConfig(
    provider="gemini",  # gemini, openai, claude
    object_filtering=True,
    prompt_enhancement=True,
    result_validation=True,
    decision_fusion=True
)
```

### 配置文件

編輯 `modules/configs/llm_config.yaml`：

```yaml
llm_enhanced_matching:
  enabled: true
  llm_provider: "gemini"
  
  object_filter:
    enabled: true
    strict_mode: false
    
  vlm_enhancement:
    enabled: true
    detailed_prompts: true
    
  result_validation:
    enabled: true
    min_confidence_threshold: 0.6
```

## 輸出格式

### 增強分析結果

```python
[
    {
        'reference_object': 'dining table',
        'placement_type': 'vlm_suggested',
        'bbox': [100, 150, 200, 250],
        'compatibility_score': 0.85,
        'source': 'vlm_validated',
        'rank': 1,
        'llm_validation': {
            'is_reasonable': True,
            'overall_confidence': 0.9,
            'reasoning': '位置合理，有適當支撐'
        }
    }
]
```

### 決策鏈信息

每個結果包含完整的決策過程：

- `source`: 結果來源 (semantic, vlm, vlm_validated)
- `filter_reason`: 物件篩選原因
- `llm_validation`: LLM驗證結果
- `fusion_priority`: 融合優先級

## 性能優化

### 1. 快取機制

LLM調用會自動快取結果，避免重複計算。

### 2. 批次處理

```python
# 處理多個物件
objects = ["cup", "book", "phone"]
results = {}

for obj in objects:
    results[obj] = matcher.enhanced_placement_analysis(
        image, obj, detected_objects
    )
```

### 3. 選擇性啟用功能

```python
config = LLMConfig(
    object_filtering=True,   # 啟用物件篩選
    prompt_enhancement=True, # 啟用提示增強
    result_validation=False, # 停用結果驗證（提升速度）
    decision_fusion=True     # 啟用決策融合
)
```

## 故障排除

### 常見問題

1. **API密鑰錯誤**
   ```
   ❌ LLM功能未啟用
   ```
   解決：檢查`.env`文件中的API密鑰設置

2. **模型不可用**
   ```
   ❌ 不支援的LLM提供商
   ```
   解決：確保安裝了對應的客戶端庫

3. **記憶體不足**
   ```
   ❌ CUDA out of memory
   ```
   解決：使用CPU模式或減少批次大小

### 調試模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 會顯示詳細的LLM調用信息
matcher = create_llm_enhanced_matcher()
```

## 最佳實踐

### 1. 場景特定配置

針對不同場景調整配置：

```python
# 廚房場景
kitchen_config = LLMConfig(
    object_filtering=True,  # 重要：過濾不適合的物件
    result_validation=True  # 重要：驗證安全性
)

# 客廳場景
living_room_config = LLMConfig(
    prompt_enhancement=True,  # 重要：考慮美觀性
    decision_fusion=True      # 重要：平衡多種因素
)
```

### 2. 錯誤處理

```python
try:
    results = matcher.enhanced_placement_analysis(
        image, object_label, detected_objects
    )
    if not results:
        # 降級到標準方法
        results = find_best_placement(
            object_label, embedding, detected_objects,
            use_llm_enhanced=False
        )
except Exception as e:
    print(f"LLM增強失敗: {e}")
    # 使用備用方案
```

### 3. 結果驗證

```python
for result in results:
    # 檢查LLM驗證結果
    if 'llm_validation' in result:
        validation = result['llm_validation']
        if not validation.get('is_reasonable', True):
            print(f"⚠️ 建議可能不合理: {validation.get('reasoning')}")
```

## API參考

### find_best_placement

主要的匹配函數，支援LLM增強。

```python
def find_best_placement(
    object_label: str,
    object_embedding: np.ndarray,
    background_objects: List[Dict] = None,
    top_k: int = 3,
    scene_embedding: Optional[np.ndarray] = None,
    background_surfaces: List[Dict] = None,
    image: Optional[np.ndarray] = None,
    use_vlm: bool = True,
    use_llm_enhanced: bool = False  # 新增參數
) -> List[Dict]
```

### LLMEnhancedMatcher

核心的LLM增強匹配器類。

```python
class LLMEnhancedMatcher:
    def enhanced_placement_analysis(
        self, 
        image: np.ndarray, 
        object_label: str,
        detected_objects: List[Dict]
    ) -> List[Dict]
```

## 更新日誌

### v1.0.0
- 初始版本
- 支援Gemini、OpenAI、Claude
- 完整的四階段增強流程
- 配置系統和錯誤處理

## 貢獻指南

歡迎提交問題和改進建議到項目倉庫。

## 授權

本項目採用MIT授權。
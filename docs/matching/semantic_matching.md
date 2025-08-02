# Semantic Matching 語意匹配模組使用說明

本文檔詳細說明語意匹配模組中各個函數的使用方法、輸入輸出要求和功能說明。

## 模組架構

```
modules/
├── models/matching_model.py    # 核心模型類
└── matching/semantic.py        # 邏輯處理層
```

---

## 1. SemanticMatcher 類 (modules/models/matching_model.py)

### 1.1 初始化

```python
SemanticMatcher(device: str = "cuda" if torch.cuda.is_available() else "cpu")
```

**功能說明**: 初始化語意匹配器，載入CLIP模型和知識庫

**輸入參數**:
- `device` (str, 可選): 計算設備，預設自動選擇GPU或CPU

**輸出**: SemanticMatcher 實例

**使用範例**:
```python
from modules.models.matching_model import SemanticMatcher

# 使用預設設備
matcher = SemanticMatcher()

# 指定使用CPU
matcher = SemanticMatcher(device="cpu")
```

---

### 1.2 主要函數

#### `calculate_semantic_compatibility()`

```python
calculate_semantic_compatibility(
    object_info: Dict, 
    surface_obj: Dict, 
    scene_embedding: Optional[np.ndarray] = None
) -> Tuple[float, Dict]
```

**功能說明**: 計算物件與表面的綜合語意相容性分數

**輸入參數**:
- `object_info` (Dict): 要插入物件的資訊
  ```python
  {
      'primary_label': str,        # 物件標籤，如 "cup", "book"
      'embedding': np.ndarray,     # 物件的CLIP嵌入向量 (shape: [512,])
      'id': str,                   # 物件ID (可選)
      # 其他物件屬性...
  }
  ```

- `surface_obj` (Dict): 背景表面物件資訊
  ```python
  {
      'label': str,                # 表面標籤，如 "table", "desk"
      'bbox': List[float],         # 邊界框座標 [x1, y1, x2, y2]
      'confidence': float,         # 檢測信心度 (可選)
      # 其他表面屬性...
  }
  ```

- `scene_embedding` (np.ndarray, 可選): 場景的CLIP嵌入向量 (shape: [512,])

**輸出**:
- `Tuple[float, Dict]`: (總分數, 詳細分數字典)
  ```python
  (
      0.75,  # 總分數 (0.0-1.0)
      {
          'clip_similarity': 0.8,      # CLIP語意相似度
          'context_matching': 0.7,     # 上下文匹配分數
          'physical_support': 1.0,     # 物理支撐相容性
          'scene_coherence': 0.6,      # 場景一致性
          'size_compatibility': 0.9    # 尺寸相容性
      }
  )
  ```

**使用範例**:
```python
object_info = {
    'primary_label': 'cup',
    'embedding': cup_embedding,  # shape: [512,]
    'id': 'obj_001'
}

surface_obj = {
    'label': 'table',
    'bbox': [100, 150, 300, 400],
    'confidence': 0.95
}

# 計算相容性
total_score, detailed_scores = matcher.calculate_semantic_compatibility(
    object_info=object_info,
    surface_obj=surface_obj,
    scene_embedding=scene_embedding  # 可選
)

print(f"總分數: {total_score:.2f}")
print(f"詳細分數: {detailed_scores}")
```

---

## 2. SemanticMatchingProcessor 類 (modules/matching/semantic.py)

### 2.1 初始化

```python
SemanticMatchingProcessor(device: str = "cuda" if torch.cuda.is_available() else "cpu")
```

**功能說明**: 初始化語意匹配處理器，內部包含SemanticMatcher實例

**輸入參數**:
- `device` (str, 可選): 計算設備

**輸出**: SemanticMatchingProcessor 實例

---

### 2.2 主要函數

#### `process_semantic_matching()`

```python
process_semantic_matching(
    insert_objects: List[Dict], 
    background_surfaces: List[Dict],
    scene_info: Optional[Dict] = None
) -> List[Dict]
```

**功能說明**: 批量處理語意匹配，為每個物件找到最佳的表面匹配

**輸入參數**:
- `insert_objects` (List[Dict]): 要插入的物件列表
  ```python
  [
      {
          'primary_label': 'cup',
          'embedding': np.ndarray,  # shape: [512,]
          'id': 'obj_001'
      },
      {
          'primary_label': 'book',
          'embedding': np.ndarray,
          'id': 'obj_002'
      }
  ]
  ```

- `background_surfaces` (List[Dict]): 背景表面物件列表
  ```python
  [
      {
          'label': 'table',
          'bbox': [100, 150, 300, 400],
          'confidence': 0.95
      },
      {
          'label': 'shelf',
          'bbox': [50, 200, 200, 350],
          'confidence': 0.88
      }
  ]
  ```

- `scene_info` (Dict, 可選): 場景資訊
  ```python
  {
      'scene_embedding': np.ndarray,  # shape: [512,]
      'scene_type': 'office',         # 場景類型 (可選)
      'lighting': 'natural'           # 光照條件 (可選)
  }
  ```

**輸出**:
- `List[Dict]`: 匹配結果列表
  ```python
  [
      {
          'object': {...},                    # 原始物件資訊
          'surface': {...},                   # 匹配的表面資訊
          'compatibility_score': 0.85,       # 相容性分數
          'detailed_scores': {...},          # 詳細分數
          'timestamp': '2024-01-01T12:00:00'  # 時間戳
      },
      # 更多匹配結果...
  ]
  ```

**使用範例**:
```python
from modules.matching.semantic import SemanticMatchingProcessor

processor = SemanticMatchingProcessor()

# 準備資料
insert_objects = [
    {
        'primary_label': 'cup',
        'embedding': cup_embedding,
        'id': 'obj_001'
    }
]

background_surfaces = [
    {
        'label': 'table',
        'bbox': [100, 150, 300, 400],
        'confidence': 0.95
    }
]

scene_info = {
    'scene_embedding': scene_embedding
}

# 執行匹配
results = processor.process_semantic_matching(
    insert_objects=insert_objects,
    background_surfaces=background_surfaces,
    scene_info=scene_info
)

for result in results:
    print(f"物件: {result['object']['primary_label']}")
    print(f"表面: {result['surface']['label']}")
    print(f"分數: {result['compatibility_score']:.2f}")
```

---

#### `filter_matches_by_threshold()`

```python
filter_matches_by_threshold(
    matching_results: List[Dict], 
    threshold: float = 0.5
) -> List[Dict]
```

**功能說明**: 根據閾值過濾匹配結果

**輸入參數**:
- `matching_results` (List[Dict]): 匹配結果列表
- `threshold` (float): 分數閾值，預設0.5

**輸出**:
- `List[Dict]`: 過濾後的匹配結果

**使用範例**:
```python
# 只保留分數大於等於0.7的匹配
high_quality_matches = processor.filter_matches_by_threshold(
    matching_results=results,
    threshold=0.7
)
```

---

#### `get_best_matches()`

```python
get_best_matches(
    matching_results: List[Dict], 
    top_k: int = 3
) -> List[Dict]
```

**功能說明**: 獲取最佳的K個匹配結果

**輸入參數**:
- `matching_results` (List[Dict]): 匹配結果列表
- `top_k` (int): 返回的最佳匹配數量，預設3

**輸出**:
- `List[Dict]`: 按分數排序的前K個匹配結果

**使用範例**:
```python
# 獲取最佳的5個匹配
best_matches = processor.get_best_matches(
    matching_results=results,
    top_k=5
)
```

---

#### `analyze_matching_quality()`

```python
analyze_matching_quality(matching_results: List[Dict]) -> Dict
```

**功能說明**: 分析匹配品質統計資訊

**輸入參數**:
- `matching_results` (List[Dict]): 匹配結果列表

**輸出**:
- `Dict`: 品質分析報告
  ```python
  {
      'total_matches': 10,              # 總匹配數
      'average_score': 0.65,            # 平均分數
      'max_score': 0.95,                # 最高分數
      'min_score': 0.25,                # 最低分數
      'std_score': 0.18,                # 分數標準差
      'high_quality_matches': 3,        # 高品質匹配數 (>=0.7)
      'medium_quality_matches': 5,      # 中等品質匹配數 (0.4-0.7)
      'low_quality_matches': 2,         # 低品質匹配數 (<0.4)
      'overall_quality': 'medium'       # 整體品質評級
  }
  ```

**使用範例**:
```python
quality_report = processor.analyze_matching_quality(results)
print(f"整體品質: {quality_report['overall_quality']}")
print(f"平均分數: {quality_report['average_score']:.2f}")
```

---

#### `get_matching_history()` 和 `clear_history()`

```python
get_matching_history() -> List[Dict]
clear_history() -> None
```

**功能說明**: 獲取和清除匹配歷史記錄

**輸出** (get_matching_history):
- `List[Dict]`: 歷史記錄列表
  ```python
  [
      {
          'object_id': 'obj_001',
          'best_match': {...},          # 最佳匹配結果
          'total_candidates': 5         # 候選表面數量
      }
  ]
  ```

**使用範例**:
```python
# 獲取歷史
history = processor.get_matching_history()

# 清除歷史
processor.clear_history()
```

---

## 3. 便利函數

### 3.1 `quick_semantic_match()`

```python
quick_semantic_match(
    object_info: Dict, 
    surface_obj: Dict, 
    scene_embedding: Optional[np.ndarray] = None
) -> Tuple[float, Dict]
```

**功能說明**: 快速單次語意匹配，不需要初始化處理器

**輸入輸出**: 與 `SemanticMatcher.calculate_semantic_compatibility()` 相同

**使用範例**:
```python
from modules.matching.semantic import quick_semantic_match

score, details = quick_semantic_match(
    object_info=object_info,
    surface_obj=surface_obj,
    scene_embedding=scene_embedding
)
```

---

### 3.2 `_calculate_semantic_score()` (向後相容)

```python
_calculate_semantic_score(
    object_info: Dict, 
    surface_obj: Dict, 
    scene_embedding: np.ndarray
) -> float
```

**功能說明**: 保留的舊版本介面，僅返回總分數

**輸入輸出**: 與新版本類似，但只返回float分數

---

## 4. 資料格式要求

### 4.1 CLIP嵌入向量
- **格式**: `np.ndarray`
- **形狀**: `[512,]` (ViT-B/32模型)
- **數據類型**: `float32`
- **範圍**: 通常在 [-1, 1] 之間

### 4.2 邊界框格式
- **格式**: `List[float]` 或 `np.ndarray`
- **內容**: `[x1, y1, x2, y2]`
- **座標系**: 左上角為原點，x向右，y向下

### 4.3 物件標籤
- **格式**: `str`
- **建議**: 使用英文小寫，如 "cup", "table", "book"
- **支援的標籤**: 參考模型中的知識庫定義

---

## 5. 錯誤處理

所有函數都包含異常處理機制：

- **CLIP模型錯誤**: 返回預設分數 0.3
- **嵌入向量格式錯誤**: 返回預設分數 0.3
- **標籤不存在**: 使用基本物理檢查，返回較低分數
- **記憶體不足**: 自動降級到CPU計算

---

## 6. 性能優化建議

1. **批量處理**: 使用 `SemanticMatchingProcessor` 而非單次調用
2. **設備選擇**: GPU加速CLIP計算，CPU處理邏輯運算
3. **嵌入快取**: 預先計算並快取常用物件的嵌入向量
4. **閾值過濾**: 及早過濾低分數匹配，減少後續計算

---

## 7. 常見問題

**Q: 如何添加新的物件類型？**
A: 在 `SemanticMatcher` 的知識庫中添加相應的規則定義

**Q: 分數總是很低怎麼辦？**
A: 檢查物件標籤是否在支援列表中，或調整權重配置

**Q: 如何自定義權重？**
A: 修改 `SemanticMatcher.weights` 字典中的權重值

**Q: 支援中文標籤嗎？**
A: 目前主要支援英文標籤，中文需要額外的翻譯處理
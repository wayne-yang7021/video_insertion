# Semantic Matching 語意匹配模組使用說明

本文檔說明簡化後的語意匹配模組的使用方法和架構。

## 模組架構

```
modules/
├── models/matching_model.py    # 核心匹配器 (SemanticMatcher)
└── matching/semantic.py        # API介面層
```

## 核心設計理念

簡化後的系統專注於3個核心維度：
1. **語意相似度** (40%) - CLIP嵌入相似度 + 上下文匹配
2. **物理相容性** (40%) - 支撐關係 + 穩定性安全檢查  
3. **場景一致性** (20%) - 物件在特定場景中的合理性

---

## 主要API函數

### 1. `find_best_placement()` - 主要API

```python
find_best_placement(
    object_label: str,
    object_embedding: np.ndarray,
    background_objects: List[Dict],
    top_k: int = 3,
    scene_embedding: Optional[np.ndarray] = None
) -> List[Dict]
```

**功能說明**: 為單一物件找到最佳的前k個放置位置（包括表面上、旁邊或附近）

**輸入參數**:
- `object_label` (str): 物件標籤，如 "cup", "book", "laptop"
- `object_embedding` (np.ndarray): 物件的CLIP嵌入向量 [512,]
- `background_objects` (List[Dict]): 背景物件列表
  ```python
  [
      {
          'label': 'dining table',
          'bbox': [100, 150, 500, 400],
          'confidence': 0.96,  # 可選
          'id': 'table_001'    # 可選
      },
      # 更多物件...
  ]
  ```
- `top_k` (int): 返回前k個最佳匹配，預設3
- `scene_embedding` (np.ndarray, 可選): 場景嵌入向量

**輸出**:
```python
[
    {
        'reference_object': 'dining table',
        'placement_type': 'on_surface',  # 'on_surface', 'beside', 'near'
        'bbox': [100, 150, 500, 400],
        'compatibility_score': 0.783,
        'rank': 1,
        'detailed_scores': {
            'semantic_similarity': 0.75,
            'physical_compatibility': 0.90,
            'scene_coherence': 0.80
        },
        'object_id': 'table_001',
        'object_confidence': 0.96
    },
    # 更多匹配結果...
]
```

**使用範例**:
```python
from modules.matching.semantic import find_best_placement
import clip
import torch

# 載入CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device)

def get_clip_embedding(text: str):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        return clip_model.encode_text(tokens).cpu().numpy()[0]

# 使用API
best_placements = find_best_placement(
    object_label="cup",
    object_embedding=get_clip_embedding("a coffee cup"),
    background_objects=[
        {'label': 'dining table', 'bbox': [100, 150, 500, 400]},
        {'label': 'chair', 'bbox': [200, 300, 300, 500]},
        {'label': 'person', 'bbox': [600, 200, 900, 450]}
    ],
    top_k=3
)

# 輸出結果
for match in best_placements:
    print(f"{match['rank']}. {match['reference_object']} ({match['placement_type']}): {match['compatibility_score']:.3f}")
```

### 2. `quick_semantic_match()` - 單次匹配

```python
quick_semantic_match(
    object_info: Dict,
    surface_obj: Dict,
    scene_embedding: Optional[np.ndarray] = None
) -> Tuple[float, Dict]
```

**功能說明**: 快速計算單一物件與表面的相容性

**輸入參數**:
- `object_info` (Dict): 物件資訊
  ```python
  {
      'primary_label': 'cup',
      'embedding': np.ndarray,  # shape: [512,]
      'id': 'obj_001'
  }
  ```
- `surface_obj` (Dict): 表面資訊
  ```python
  {
      'label': 'dining table',
      'bbox': [100, 150, 300, 400]
  }
  ```
- `scene_embedding` (np.ndarray, 可選): 場景嵌入向量

**輸出**:
```python
(
    0.75,  # 總分數 (0.0-1.0)
    {
        'semantic_similarity': 0.8,      # 語意相似度
        'physical_compatibility': 0.9,   # 物理相容性
        'scene_coherence': 0.6          # 場景一致性
    }
)
```

**使用範例**:
```python
from modules.matching.semantic import quick_semantic_match

score, details = quick_semantic_match(
    object_info={'primary_label': 'cup', 'embedding': cup_embedding},
    surface_obj={'label': 'dining table', 'bbox': [100, 150, 300, 400]}
)

print(f"相容性分數: {score:.3f}")
print(f"詳細分數: {details}")
```

---

## 核心架構說明

### SemanticMatcher 類 (modules/models/matching_model.py)

**主要功能**:
- 載入CLIP模型進行語意理解
- 整合3個評分維度計算相容性分數
- 包含安全規則避免不合理的放置

**核心方法**:
- `calculate_semantic_compatibility()`: 計算物件與表面的相容性分數
- `_calculate_semantic_similarity()`: 結合CLIP和上下文的語意分析
- `_calculate_physical_compatibility()`: 物理支撐和穩定性檢查
- `_calculate_scene_coherence()`: 場景一致性評估

**為什麼需要**: 提供準確的物件放置建議，避免不安全或不合理的組合

**如何使用**: 通常不直接使用，而是透過API函數調用

### SemanticMatchingProcessor 類 (modules/matching/semantic.py)

**主要功能**:
- 提供簡化的API介面
- 處理批量匹配和結果排序

**核心方法**:
- `find_best_surfaces()`: 為單一物件找到最佳表面
- `get_best_matches()`: 從結果中提取前k個最佳匹配

**為什麼需要**: 簡化使用介面，隱藏複雜的內部邏輯

**如何使用**: 透過便利函數 `find_best_placement_surfaces()` 調用

---

## 評分維度說明

### 1. 語意相似度 (40%)
- **CLIP相似度**: 物件與表面描述的向量相似度
- **上下文匹配**: 物件在表面上的情境描述與場景的匹配度
- **目的**: 確保語意上的合理性

### 2. 物理相容性 (40%)
- **支撐關係**: 表面是否能物理支撐該物件
- **安全檢查**: 避免不穩定或危險的組合 (如酒杯放沙發上)
- **目的**: 確保物理上的可行性和安全性

### 3. 場景一致性 (20%)
- **場景匹配**: 物件與表面在特定場景中的合理性
- **常見組合**: 基於真實世界的常見搭配
- **目的**: 提高放置的自然度和真實感

---

## 資料格式要求

- **物件標籤**: COCO 80類別名稱 (如 "cup", "dining table", "cell phone")
- **CLIP嵌入**: shape [512,] 的 numpy array
- **邊界框**: [x1, y1, x2, y2] 格式的座標列表
- **分數範圍**: 0.0-1.0，越高表示相容性越好

---

## 使用建議

1. **主要API**: 使用 `find_best_placement()` 獲取最佳放置位置
2. **單次測試**: 使用 `quick_semantic_match()` 測試特定配對
3. **分數解讀**: >0.7 高品質，0.4-0.7 中等，<0.4 不建議
4. **安全考量**: 系統會自動降低不安全組合的分數
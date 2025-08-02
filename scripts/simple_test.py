#!/usr/bin/env python3
"""
簡單的語意匹配測試，不依賴外部庫
"""

import sys
import os

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_hardcoded_rules():
    """測試硬編碼規則的邏輯"""
    print("🧪 測試硬編碼規則邏輯")
    print("=" * 50)
    
    # 模擬現有的規則
    support_rules = {
        "table": ["cup", "book", "laptop", "plate", "phone", "glass", "bottle", "pen", "notebook"],
        "desk": ["laptop", "book", "pen", "monitor", "keyboard", "mouse", "phone", "notebook"],
        "shelf": ["book", "decoration", "bottle", "plant", "frame", "clock", "vase"],
        "couch": ["pillow", "remote", "book", "phone", "blanket", "cushion"],
    }
    
    scene_groups = {
        "living_room": ["sofa", "tv", "remote", "table", "cushion", "lamp", "plant"],
        "office": ["laptop", "book", "pen", "desk", "chair", "monitor", "keyboard", "mouse"],
    }
    
    size_categories = {
        "small": ["cup", "phone", "book", "remote", "pen", "glass", "bottle"],
        "medium": ["laptop", "plate", "lamp", "clock", "vase", "bowl"],
        "large": ["chair", "monitor", "plant", "bag", "pillow"],
        "huge": ["table", "sofa", "bed", "desk", "tv"]
    }
    
    # 你的detection結果
    detections = [
        {"label": "potted plant", "confidence": 1.00},
        {"label": "book", "confidence": 0.95},
        {"label": "couch", "confidence": 0.95},
        {"label": "vase", "confidence": 0.83}
    ]
    
    print("檢查物件在規則中的匹配情況：")
    print("-" * 30)
    
    for detection in detections:
        obj_label = detection["label"]
        print(f"\n📦 物件: {obj_label}")
        
        # 檢查支撐規則
        can_be_supported_by = []
        for surface, supported_objects in support_rules.items():
            if obj_label in supported_objects:
                can_be_supported_by.append(surface)
        
        if can_be_supported_by:
            print(f"  ✅ 可被支撐: {', '.join(can_be_supported_by)}")
        else:
            print(f"  ❌ 不在支撐規則中")
        
        # 檢查場景分組
        appears_in_scenes = []
        for scene, objects in scene_groups.items():
            if obj_label in objects:
                appears_in_scenes.append(scene)
        
        if appears_in_scenes:
            print(f"  🏠 出現場景: {', '.join(appears_in_scenes)}")
        else:
            print(f"  🏠 不在場景分組中")
        
        # 檢查尺寸分類
        size_category = None
        for size, objects in size_categories.items():
            if obj_label in objects:
                size_category = size
                break
        
        if size_category:
            print(f"  📏 尺寸分類: {size_category}")
        else:
            print(f"  📏 沒有尺寸分類")


if __name__ == "__main__":
    test_hardcoded_rules()
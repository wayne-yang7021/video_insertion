#!/usr/bin/env python3
"""
測試語意匹配功能的腳本
使用實際的detection結果測試SemanticMatcher的表現
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 先測試import是否正常
try:
    from modules.matching.semantic import SemanticMatchingProcessor, quick_semantic_match
    from modules.models.matching_model import SemanticMatcher
    print("✅ 成功導入語意匹配模組")
except ImportError as e:
    print(f"❌ 導入模組失敗: {e}")
    sys.exit(1)


def create_mock_embedding(text: str) -> np.ndarray:
    """創建模擬的嵌入向量（不使用真實CLIP）"""
    # 使用簡單的hash來創建一致的假嵌入向量
    np.random.seed(hash(text) % 2**32)
    return np.random.randn(512).astype(np.float32)


def parse_detection_results() -> Tuple[List[Dict], List[Dict]]:
    """
    解析detection結果，分為要插入的物件和背景表面

    Detection結果：
    - potted plant (1.00) → box: [512.57, 454.22, 753.74, 624.50]
    - book (0.95) → box: [94.39, 874.68, 226.27, 930.45]
    - couch (0.95) → box: [176.92, 668.90, 807.60, 1223.07]
    - couch (0.86) → box: [3.10, 955.39, 306.17, 1268.15]
    - vase (0.83) → box: [759.70, 497.21, 830.03, 625.19]
    """

    # 原始detection資料
    detections = [
        {"label": "potted plant", "confidence": 1.00,
            "bbox": [512.57, 454.22, 753.74, 624.50]},
        {"label": "book", "confidence": 0.95,
            "bbox": [94.39, 874.68, 226.27, 930.45]},
        {"label": "couch", "confidence": 0.95,
            "bbox": [176.92, 668.90, 807.60, 1223.07]},
        {"label": "couch", "confidence": 0.86,
            "bbox": [3.10, 955.39, 306.17, 1268.15]},
        {"label": "vase", "confidence": 0.83,
            "bbox": [759.70, 497.21, 830.03, 625.19]}
    ]

    # 分類：小物件作為要插入的物件，大物件作為背景表面
    insert_objects = []
    background_surfaces = []

    for i, detection in enumerate(detections):
        # 生成模擬嵌入向量
        embedding = create_mock_embedding(detection["label"])

        # 根據物件類型分類
        if detection["label"] in ["book", "vase", "potted plant"]:
            # 小物件 - 作為要插入的物件
            insert_objects.append({
                "id": f"insert_{i}",
                "primary_label": detection["label"],
                "embedding": embedding,
                "confidence": detection["confidence"],
                "original_bbox": detection["bbox"]
            })
        else:
            # 大物件 - 作為背景表面
            background_surfaces.append({
                "id": f"surface_{i}",
                "label": detection["label"],
                "bbox": detection["bbox"],
                "confidence": detection["confidence"]
            })

    return insert_objects, background_surfaces


def test_individual_matches():
    """測試個別物件間的匹配分數"""
    print("=" * 60)
    print("🧪 測試個別物件匹配")
    print("=" * 60)

    insert_objects, background_surfaces = parse_detection_results()

    # 如果沒有背景表面，創建一些測試用的表面
    if not background_surfaces:
        test_surfaces = [
            {"label": "table", "bbox": [100, 100, 400, 300]},
            {"label": "shelf", "bbox": [50, 200, 200, 500]},
            {"label": "desk", "bbox": [200, 150, 600, 400]},
            {"label": "floor", "bbox": [0, 800, 1000, 1200]}
        ]

        background_surfaces = []
        for i, surface in enumerate(test_surfaces):
            background_surfaces.append({
                "id": f"test_surface_{i}",
                "label": surface["label"],
                "bbox": surface["bbox"],
                "confidence": 0.9
            })

    # 測試每個插入物件與每個表面的匹配
    for obj in insert_objects:
        print(
            f"\n📦 測試物件: {obj['primary_label']} (confidence: {obj['confidence']:.2f})")
        print("-" * 40)

        for surface in background_surfaces:
            try:
                score, details = quick_semantic_match(
                    object_info=obj,
                    surface_obj=surface
                )

                print(f"  與 {surface['label']} 的匹配:")
                print(f"    總分數: {score:.3f}")
                print(f"    詳細分數:")
                for key, value in details.items():
                    if key != 'error':
                        print(f"      {key}: {value:.3f}")
                print()

            except Exception as e:
                print(f"  ❌ 與 {surface['label']} 匹配失敗: {e}")


def test_batch_processing():
    """測試批量處理功能"""
    print("=" * 60)
    print("🔄 測試批量處理")
    print("=" * 60)

    insert_objects, background_surfaces = parse_detection_results()

    # 如果沒有背景表面，使用測試表面
    if not background_surfaces:
        test_surfaces = [
            {"label": "table", "bbox": [100, 100, 400, 300]},
            {"label": "shelf", "bbox": [50, 200, 200, 500]}
        ]

        background_surfaces = []
        for i, surface in enumerate(test_surfaces):
            background_surfaces.append({
                "id": f"test_surface_{i}",
                "label": surface["label"],
                "bbox": surface["bbox"],
                "confidence": 0.9
            })

    # 使用SemanticMatchingProcessor進行批量處理
    processor = SemanticMatchingProcessor()

    try:
        results = processor.process_semantic_matching(
            insert_objects=insert_objects,
            background_surfaces=background_surfaces
        )

        print(f"✅ 成功處理 {len(results)} 個匹配結果")

        # 分析匹配品質
        quality_analysis = processor.analyze_matching_quality(results)
        print(f"\n📊 匹配品質分析:")
        print(f"  總匹配數: {quality_analysis['total_matches']}")
        print(f"  平均分數: {quality_analysis['average_score']:.3f}")
        print(f"  最高分數: {quality_analysis['max_score']:.3f}")
        print(f"  最低分數: {quality_analysis['min_score']:.3f}")
        print(f"  整體品質: {quality_analysis['overall_quality']}")
        print(f"  高品質匹配: {quality_analysis['high_quality_matches']}")
        print(f"  中等品質匹配: {quality_analysis['medium_quality_matches']}")
        print(f"  低品質匹配: {quality_analysis['low_quality_matches']}")

        # 顯示最佳匹配
        best_matches = processor.get_best_matches(results, top_k=5)
        print(f"\n🏆 前5個最佳匹配:")
        for i, match in enumerate(best_matches, 1):
            obj_label = match['object']['primary_label']
            surface_label = match['surface']['label']
            score = match['compatibility_score']
            print(f"  {i}. {obj_label} → {surface_label}: {score:.3f}")

    except Exception as e:
        print(f"❌ 批量處理失敗: {e}")
        import traceback
        traceback.print_exc()


def test_unknown_objects():
    """測試未知物件的處理"""
    print("=" * 60)
    print("🔍 測試未知物件處理")
    print("=" * 60)

    # 測試一些可能不在預定義規則中的物件
    unknown_objects = ["smartphone", "headphones", "mug", "notebook"]
    test_surfaces = [
        {"label": "table", "bbox": [100, 100, 400, 300]},
        {"label": "shelf", "bbox": [50, 200, 200, 500]}
    ]

    for obj_label in unknown_objects:
        print(f"\n📱 測試未知物件: {obj_label}")
        print("-" * 30)

        # 創建物件資訊
        embedding = create_mock_embedding(obj_label)
        object_info = {
            "primary_label": obj_label,
            "embedding": embedding,
            "id": f"unknown_{obj_label}"
        }

        for surface in test_surfaces:
            try:
                score, details = quick_semantic_match(
                    object_info=object_info,
                    surface_obj=surface
                )

                print(f"  與 {surface['label']} 的匹配: {score:.3f}")

            except Exception as e:
                print(f"  ❌ 與 {surface['label']} 匹配失敗: {e}")


def main():
    """主測試函數"""
    print("🚀 開始語意匹配測試")
    print("⚠️  注意：使用模擬嵌入向量進行測試")

    try:
        # 測試個別匹配
        test_individual_matches()

        # 測試批量處理
        test_batch_processing()

        # 測試未知物件
        test_unknown_objects()

        print("\n" + "=" * 60)
        print("✅ 所有測試完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

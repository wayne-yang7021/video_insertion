#!/usr/bin/env python3
"""
測試新的placement API - 模擬真實使用情境
輸入：單一物件 + 背景表面列表
輸出：前n個最佳匹配的表面和bbox
"""

from modules.matching.semantic import find_best_placement_surfaces, SemanticMatchingProcessor
import sys
import os
import numpy as np
import torch
import clip
from typing import Dict, List

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_placement_api():
    """測試新的placement API"""
    print("🚀 測試物件放置API")
    print("="*60)

    # 初始化CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)

    def get_clip_embedding(text: str) -> np.ndarray:
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(device)
            text_embedding = clip_model.encode_text(text_tokens)
            return text_embedding.cpu().numpy()[0]

    # 測試案例1: 要插入一個杯子
    print("\n📋 測試案例1: 插入杯子")
    print("-" * 40)

    object_label = "cup"
    object_embedding = get_clip_embedding("a coffee cup")

    # 模擬detection結果 - 背景中的表面物件
    background_surfaces = [
        {
            'label': 'dining table',
            'bbox': [100, 150, 500, 400],
            'confidence': 0.96,
            'id': 'table_001'
        },
        {
            'label': 'chair',
            'bbox': [200, 300, 300, 500],
            'confidence': 0.85,
            'id': 'chair_001'
        },
        {
            'label': 'couch',
            'bbox': [600, 200, 900, 450],
            'confidence': 0.91,
            'id': 'couch_001'
        },
        {
            'label': 'sink',
            'bbox': [50, 100, 200, 250],
            'confidence': 0.88,
            'id': 'sink_001'
        }
    ]

    # 使用新API找到最佳放置位置
    best_surfaces = find_best_placement_surfaces(
        object_label=object_label,
        object_embedding=object_embedding,
        background_surfaces=background_surfaces,
        top_k=3
    )

    print(f"要插入的物件: {object_label}")
    print(f"背景表面數量: {len(background_surfaces)}")
    print(f"\n🏆 前3個最佳放置位置:")

    for match in best_surfaces:
        print(f"  {match['rank']}. {match['surface_label']}")
        print(f"     分數: {match['compatibility_score']:.3f}")
        print(f"     bbox: {match['bbox']}")
        print(f"     表面ID: {match['surface_id']}")
        print()

    # 測試案例2: 要插入一本書
    print("\n📋 測試案例2: 插入書本")
    print("-" * 40)

    object_label = "book"
    object_embedding = get_clip_embedding("a book")

    # 不同的背景場景
    background_surfaces = [
        {
            'label': 'bed',
            'bbox': [100, 500, 800, 900],
            'confidence': 0.89,
            'id': 'bed_001'
        },
        {
            'label': 'dining table',
            'bbox': [200, 150, 600, 400],
            'confidence': 0.94,
            'id': 'table_002'
        },
        {
            'label': 'toilet',
            'bbox': [50, 600, 150, 800],
            'confidence': 0.82,
            'id': 'toilet_001'
        },
        {
            'label': 'chair',
            'bbox': [300, 300, 400, 500],
            'confidence': 0.87,
            'id': 'chair_002'
        }
    ]

    best_surfaces = find_best_placement_surfaces(
        object_label=object_label,
        object_embedding=object_embedding,
        background_surfaces=background_surfaces,
        top_k=3
    )

    print(f"要插入的物件: {object_label}")
    print(f"背景表面數量: {len(background_surfaces)}")
    print(f"\n🏆 前3個最佳放置位置:")

    for match in best_surfaces:
        print(f"  {match['rank']}. {match['surface_label']}")
        print(f"     分數: {match['compatibility_score']:.3f}")
        print(f"     bbox: {match['bbox']}")
        print(f"     關鍵因子:")

        # 顯示前3個最重要的評分因子
        sorted_scores = sorted(match['detailed_scores'].items(),
                               key=lambda x: x[1], reverse=True)
        for key, value in sorted_scores[:3]:
            if key != 'error':
                print(f"       {key}: {value:.3f}")
        print()

    # 測試案例3: 邊界情況 - 不穩定的物件
    print("\n📋 測試案例3: 不穩定物件 (wine glass)")
    print("-" * 40)

    object_label = "wine glass"
    object_embedding = get_clip_embedding("a wine glass")

    background_surfaces = [
        {
            'label': 'couch',
            'bbox': [100, 200, 400, 450],
            'confidence': 0.91,
            'id': 'couch_002'
        },
        {
            'label': 'dining table',
            'bbox': [500, 150, 800, 400],
            'confidence': 0.96,
            'id': 'table_003'
        },
        {
            'label': 'chair',
            'bbox': [200, 300, 300, 500],
            'confidence': 0.85,
            'id': 'chair_003'
        }
    ]

    best_surfaces = find_best_placement_surfaces(
        object_label=object_label,
        object_embedding=object_embedding,
        background_surfaces=background_surfaces,
        top_k=3
    )

    print(f"要插入的物件: {object_label}")
    print(f"\n🏆 前3個最佳放置位置:")

    for match in best_surfaces:
        stability_score = match['detailed_scores'].get(
            'stability_assessment', 0)
        safety_indicator = "✅ 安全" if stability_score >= 0.7 else "⚠️ 不穩定" if stability_score >= 0.4 else "❌ 危險"

        print(
            f"  {match['rank']}. {match['surface_label']} - {safety_indicator}")
        print(f"     總分數: {match['compatibility_score']:.3f}")
        print(f"     穩定性: {stability_score:.3f}")
        print(f"     bbox: {match['bbox']}")
        print()


def test_api_with_scene_context():
    """測試帶場景上下文的API"""
    print("\n" + "="*60)
    print("🎬 測試帶場景上下文的API")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)

    def get_clip_embedding(text: str) -> np.ndarray:
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(device)
            text_embedding = clip_model.encode_text(text_tokens)
            return text_embedding.cpu().numpy()[0]

    # 廚房場景
    object_label = "cup"
    object_embedding = get_clip_embedding("a coffee cup")
    scene_embedding = get_clip_embedding("a modern kitchen with dining area")

    background_surfaces = [
        {
            'label': 'dining table',
            'bbox': [100, 150, 500, 400],
            'confidence': 0.96,
            'id': 'kitchen_table'
        },
        {
            'label': 'sink',
            'bbox': [50, 100, 200, 250],
            'confidence': 0.88,
            'id': 'kitchen_sink'
        },
        {
            'label': 'couch',
            'bbox': [600, 200, 900, 450],
            'confidence': 0.91,
            'id': 'living_couch'
        }
    ]

    # 不帶場景上下文
    results_no_context = find_best_placement_surfaces(
        object_label=object_label,
        object_embedding=object_embedding,
        background_surfaces=background_surfaces,
        top_k=3
    )

    # 帶場景上下文
    results_with_context = find_best_placement_surfaces(
        object_label=object_label,
        object_embedding=object_embedding,
        background_surfaces=background_surfaces,
        top_k=3,
        scene_embedding=scene_embedding
    )

    print("📊 場景上下文對比:")
    print("\n不帶場景上下文:")
    for match in results_no_context:
        print(
            f"  {match['rank']}. {match['surface_label']}: {match['compatibility_score']:.3f}")

    print("\n帶廚房場景上下文:")
    for match in results_with_context:
        print(
            f"  {match['rank']}. {match['surface_label']}: {match['compatibility_score']:.3f}")


def main():
    """主測試函數"""
    try:
        test_placement_api()
        test_api_with_scene_context()

        print("\n" + "="*60)
        print("✅ 所有API測試完成")
        print("="*60)

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

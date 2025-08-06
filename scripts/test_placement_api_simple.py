#!/usr/bin/env python3
"""
簡化的placement API測試
"""

import sys
import os
import numpy as np
import torch
import clip

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.matching.semantic import find_best_placement

def test_api():
    """測試新的API"""
    print("🚀 測試 find_best_placement API")
    print("="*50)
    
    # 初始化CLIP模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    
    def get_clip_embedding(text: str) -> np.ndarray:
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(device)
            text_embedding = clip_model.encode_text(text_tokens)
            return text_embedding.cpu().numpy()[0]
    
    # 測試案例
    object_label = "cup"
    object_embedding = get_clip_embedding("a coffee cup")
    
    background_surfaces = [
        {'label': 'dining table', 'bbox': [100, 150, 500, 400], 'id': 'table_001'},
        {'label': 'chair', 'bbox': [200, 300, 300, 500], 'id': 'chair_001'},
        {'label': 'couch', 'bbox': [600, 200, 900, 450], 'id': 'couch_001'},
        {'label': 'sink', 'bbox': [50, 100, 200, 250], 'id': 'sink_001'}
    ]
    
    # 使用API
    best_placements = find_best_placement(
        object_label=object_label,
        object_embedding=object_embedding,
        background_surfaces=background_surfaces,
        top_k=1
    )
    
    print(f"要插入的物件: {object_label}")
    print(f"背景表面數量: {len(background_surfaces)}")
    print(f"\n🏆 前3個最佳放置位置:")
    
    for match in best_placements:
        print(f"  {match['rank']}. {match['surface_label']}")
        print(f"     分數: {match['compatibility_score']:.3f}")
        print(f"     bbox: {match['bbox']}")
        print(f"     表面ID: {match['surface_id']}")
        print()

if __name__ == "__main__":
    test_api()
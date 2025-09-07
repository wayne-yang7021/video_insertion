#!/usr/bin/env python3
"""
測試LLM大小建議功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from modules.llm.placer import LLMPlacer


def create_test_background():
    """創建測試背景圖片"""
    # 創建一個簡單的測試場景
    img = Image.new('RGB', (800, 600), color='lightblue')
    return img


def create_test_object():
    """創建測試物件圖片"""
    # 創建一個簡單的測試物件（杯子）
    img = Image.new('RGB', (200, 300), color='white')
    return img


def test_size_suggestion():
    """測試大小建議功能"""
    print("🧪 測試LLM大小建議功能")
    print("=" * 50)
    
    # 初始化LLM放置器
    placer = LLMPlacer()
    
    if not placer.is_enabled():
        print("❌ LLM功能未啟用，請檢查API密鑰配置")
        return
    
    # 準備測試數據
    background_image = create_test_background()
    object_image = create_test_object()
    
    # 模擬已偵測到的物件
    detected_objects = [
        {
            'label': 'dining table',
            'bbox': [0.2, 0.4, 0.8, 0.9],  # 標準化座標
            'confidence': 0.95
        },
        {
            'label': 'chair',
            'bbox': [0.1, 0.3, 0.3, 0.8],
            'confidence': 0.88
        },
        {
            'label': 'book',
            'bbox': [0.5, 0.45, 0.6, 0.55],
            'confidence': 0.82
        }
    ]
    
    object_label = "cup"
    
    print(f"📏 為 {object_label} 分析適當大小...")
    print(f"🔍 場景中的物件: {[obj['label'] for obj in detected_objects]}")
    
    # 調用大小建議功能
    result = placer.suggest_object_size(
        background_image=background_image,
        object_to_insert=object_image,
        detected_objects=detected_objects,
        object_label=object_label
    )
    
    if result:
        print("\n✅ 大小建議結果:")
        print("-" * 30)
        
        size_suggestion = result.get('size_suggestion', {})
        print(f"寬度縮放: {size_suggestion.get('width_scale', 1.0):.2f}")
        print(f"高度縮放: {size_suggestion.get('height_scale', 1.0):.2f}")
        print(f"參考物件: {size_suggestion.get('reference_object', '未知')}")
        print(f"調整原因: {size_suggestion.get('reasoning', '無')}")
        print(f"信心度: {size_suggestion.get('confidence', 0):.2f}")
        
        # 如果有建議的像素大小
        pixel_size = size_suggestion.get('suggested_pixel_size', {})
        if pixel_size:
            print(f"建議像素大小: {pixel_size.get('width', 0)} x {pixel_size.get('height', 0)}")
        
        # 如果有備選方案
        alternatives = size_suggestion.get('alternative_scales', [])
        if alternatives:
            print("\n🔄 備選大小方案:")
            for i, alt in enumerate(alternatives, 1):
                print(f"  {i}. 縮放 {alt.get('width_scale', 1.0):.2f}x{alt.get('height_scale', 1.0):.2f} - {alt.get('reasoning', '無原因')}")
        
    else:
        print("❌ 無法獲得大小建議")


def test_with_real_scenario():
    """使用更真實的場景測試"""
    print("\n🏠 測試真實場景大小建議")
    print("=" * 50)
    
    placer = LLMPlacer()
    
    if not placer.is_enabled():
        print("❌ LLM功能未啟用")
        return
    
    # 模擬客廳場景的偵測結果
    living_room_objects = [
        {
            'label': 'sofa',
            'bbox': [0.1, 0.3, 0.7, 0.8],
            'confidence': 0.92
        },
        {
            'label': 'coffee table',
            'bbox': [0.3, 0.6, 0.6, 0.9],
            'confidence': 0.89
        },
        {
            'label': 'tv',
            'bbox': [0.2, 0.1, 0.8, 0.4],
            'confidence': 0.95
        },
        {
            'label': 'remote control',
            'bbox': [0.45, 0.65, 0.5, 0.7],
            'confidence': 0.78
        }
    ]
    
    # 測試不同物件的大小建議
    test_objects = [
        ("vase", "花瓶"),
        ("lamp", "檯燈"),
        ("cushion", "抱枕")
    ]
    
    background_image = create_test_background()
    
    for obj_label, obj_name in test_objects:
        print(f"\n📐 分析 {obj_name} ({obj_label}) 的適當大小...")
        
        object_image = create_test_object()  # 簡化，實際應該是不同的物件圖片
        
        result = placer.suggest_object_size(
            background_image=background_image,
            object_to_insert=object_image,
            detected_objects=living_room_objects,
            object_label=obj_label
        )
        
        if result and 'size_suggestion' in result:
            suggestion = result['size_suggestion']
            print(f"  ✅ 建議縮放: {suggestion.get('width_scale', 1.0):.2f} x {suggestion.get('height_scale', 1.0):.2f}")
            print(f"  📍 參考: {suggestion.get('reference_object', '未知')}")
        else:
            print(f"  ❌ 無法獲得 {obj_name} 的大小建議")


if __name__ == "__main__":
    print("🚀 開始測試LLM大小建議功能")
    
    try:
        # 基本功能測試
        test_size_suggestion()
        
        # 真實場景測試
        test_with_real_scenario()
        
        print("\n🎉 測試完成！")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
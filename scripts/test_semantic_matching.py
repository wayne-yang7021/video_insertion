#!/usr/bin/env python3
"""
完全模擬真實情境的語意匹配測試腳本
使用真實CLIP模型和實際detection結果格式進行測試
"""

import sys
import os
import numpy as np
import torch
import clip
from typing import Dict, List, Tuple

# 添加專案根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 導入模組
try:
    from modules.matching.semantic import SemanticMatchingProcessor, quick_semantic_match
    from modules.models.matching_model import SemanticMatcher
    print("✅ 成功導入語意匹配模組")
except ImportError as e:
    print(f"❌ 導入模組失敗: {e}")
    sys.exit(1)


class RealSemanticMatchingTester:
    """真實情境的語意匹配測試器"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 使用設備: {self.device}")
        
        # 載入真實CLIP模型
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("✅ 成功載入CLIP模型")
        except Exception as e:
            print(f"❌ CLIP模型載入失敗: {e}")
            sys.exit(1)
        
        # 初始化語意匹配處理器
        self.processor = SemanticMatchingProcessor(device=self.device)
        print("✅ 成功初始化語意匹配處理器")
    
    def get_real_clip_embedding(self, text: str) -> np.ndarray:
        """獲取真實的CLIP文字嵌入向量"""
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(self.device)
            text_embedding = self.clip_model.encode_text(text_tokens)
            return text_embedding.cpu().numpy()[0]
    
    def create_real_detection_data(self) -> Tuple[List[Dict], List[Dict], Dict]:
        """創建模擬真實detection結果的測試資料"""
        
        # 模擬真實的detection結果 - 要插入的物件
        insert_objects = [
            {
                'primary_label': 'cup',
                'embedding': self.get_real_clip_embedding('a coffee cup'),
                'id': 'detected_cup_001',
                'confidence': 0.95,
                'original_bbox': [120, 200, 180, 280],
                'area': 3600,
                'detection_source': 'detectron2'
            },
            {
                'primary_label': 'book',
                'embedding': self.get_real_clip_embedding('a book'),
                'id': 'detected_book_001', 
                'confidence': 0.88,
                'original_bbox': [300, 400, 380, 450],
                'area': 4000,
                'detection_source': 'detectron2'
            },
            {
                'primary_label': 'laptop',
                'embedding': self.get_real_clip_embedding('a laptop computer'),
                'id': 'detected_laptop_001',
                'confidence': 0.92,
                'original_bbox': [450, 300, 650, 450],
                'area': 30000,
                'detection_source': 'detectron2'
            },
            {
                'primary_label': 'wine glass',
                'embedding': self.get_real_clip_embedding('a wine glass'),
                'id': 'detected_wineglass_001',
                'confidence': 0.87,
                'original_bbox': [200, 250, 230, 320],
                'area': 2100,
                'detection_source': 'detectron2'
            }
        ]
        
        # 模擬真實的detection結果 - 背景表面
        background_surfaces = [
            {
                'label': 'dining table',
                'bbox': [50, 150, 500, 400],
                'confidence': 0.96,
                'id': 'detected_table_001',
                'area': 112500,
                'material_hint': 'wood',
                'detection_source': 'detectron2'
            },
            {
                'label': 'chair', 
                'bbox': [200, 300, 300, 500],
                'confidence': 0.85,
                'id': 'detected_chair_001',
                'area': 20000,
                'material_hint': 'fabric',
                'detection_source': 'detectron2'
            },
            {
                'label': 'couch',
                'bbox': [600, 200, 900, 450],
                'confidence': 0.91,
                'id': 'detected_couch_001',
                'area': 75000,
                'material_hint': 'leather',
                'detection_source': 'detectron2'
            },
            {
                'label': 'bed',
                'bbox': [100, 500, 800, 900],
                'confidence': 0.89,
                'id': 'detected_bed_001',
                'area': 280000,
                'material_hint': 'fabric',
                'detection_source': 'detectron2'
            }
        ]
        
        # 模擬場景資訊
        scene_info = {
            'scene_embedding': self.get_real_clip_embedding('a modern living room with dining area'),
            'scene_type': 'living_dining_room',
            'lighting_condition': 'natural_daylight',
            'room_size': 'medium',
            'style': 'modern',
            'total_objects_detected': len(insert_objects) + len(background_surfaces)
        }
        
        return insert_objects, background_surfaces, scene_info


    def test_comprehensive_matching(self):
        """完整的語意匹配測試"""
        print("\n" + "="*60)
        print("🧪 完整語意匹配測試")
        print("="*60)
        
        # 獲取真實測試資料
        insert_objects, background_surfaces, scene_info = self.create_real_detection_data()
        
        print(f"📦 要插入的物件數量: {len(insert_objects)}")
        for obj in insert_objects:
            print(f"  - {obj['primary_label']} (confidence: {obj['confidence']:.2f})")
        
        print(f"\n🏠 背景表面數量: {len(background_surfaces)}")
        for surf in background_surfaces:
            print(f"  - {surf['label']} (confidence: {surf['confidence']:.2f})")
        
        print(f"\n🎬 場景類型: {scene_info['scene_type']}")
        
        # 執行批量匹配
        try:
            results = self.processor.process_semantic_matching(
                insert_objects=insert_objects,
                background_surfaces=background_surfaces,
                scene_info=scene_info
            )
            
            print(f"\n✅ 成功生成 {len(results)} 個匹配結果")
            
            # 顯示詳細結果
            self._display_detailed_results(results)
            
            # 分析匹配品質
            self._analyze_matching_quality(results)
            
            # 獲取最佳匹配
            self._show_best_matches(results)
            
            return results
            
        except Exception as e:
            print(f"❌ 批量匹配失敗: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_individual_pairs(self):
        """測試個別物件配對"""
        print("\n" + "="*60)
        print("🔍 個別配對測試")
        print("="*60)
        
        # 測試特定的物件-表面配對
        test_pairs = [
            ('cup', 'dining table'),
            ('laptop', 'dining table'),
            ('book', 'bed'),
            ('wine glass', 'couch'),
            ('book', 'chair')
        ]
        
        for obj_label, surf_label in test_pairs:
            print(f"\n📋 測試配對: {obj_label} → {surf_label}")
            print("-" * 40)
            
            # 創建測試資料
            object_info = {
                'primary_label': obj_label,
                'embedding': self.get_real_clip_embedding(f'a {obj_label}'),
                'id': f'test_{obj_label}'
            }
            
            surface_obj = {
                'label': surf_label,
                'bbox': [100, 100, 400, 300],
                'confidence': 0.9
            }
            
            try:
                score, details = quick_semantic_match(
                    object_info=object_info,
                    surface_obj=surface_obj
                )
                
                print(f"  總分數: {score:.3f}")
                print(f"  詳細分數:")
                for key, value in details.items():
                    if key != 'error':
                        print(f"    {key}: {value:.3f}")
                
                # 評估結果
                if score >= 0.7:
                    print(f"  ✅ 高品質匹配")
                elif score >= 0.4:
                    print(f"  ⚠️ 中等品質匹配")
                else:
                    print(f"  ❌ 低品質匹配")
                    
            except Exception as e:
                print(f"  ❌ 配對測試失敗: {e}")
    
    def test_edge_cases(self):
        """測試邊界情況"""
        print("\n" + "="*60)
        print("🚨 邊界情況測試")
        print("="*60)
        
        edge_cases = [
            # 不穩定的組合
            ('sports ball', 'chair', '球類在椅子上不穩定'),
            ('wine glass', 'couch', '酒杯在沙發上不穩定'),
            
            # 不合理的組合
            ('laptop', 'sink', '電子產品避免接觸水'),
            ('cell phone', 'toilet', '手機在馬桶附近不衛生'),
            
            # 尺寸不匹配
            ('dining table', 'cup', '桌子不能放在杯子上'),
            
            # 高品質匹配
            ('remote', 'couch', '遙控器和沙發是經典組合'),
            ('toothbrush', 'sink', '牙刷和洗手台是完美匹配')
        ]
        
        for obj_label, surf_label, description in edge_cases:
            print(f"\n🧪 {description}")
            print(f"   配對: {obj_label} → {surf_label}")
            
            object_info = {
                'primary_label': obj_label,
                'embedding': self.get_real_clip_embedding(f'a {obj_label}'),
                'id': f'edge_test_{obj_label}'
            }
            
            surface_obj = {
                'label': surf_label,
                'bbox': [100, 100, 400, 300],
                'confidence': 0.9
            }
            
            try:
                score, details = quick_semantic_match(
                    object_info=object_info,
                    surface_obj=surface_obj
                )
                
                print(f"   分數: {score:.3f}")
                
                # 分析關鍵因子
                key_factors = ['stability_assessment', 'physical_support', 'usage_frequency']
                for factor in key_factors:
                    if factor in details:
                        print(f"   {factor}: {details[factor]:.3f}")
                        
            except Exception as e:
                print(f"   ❌ 測試失敗: {e}")
    
    def _display_detailed_results(self, results: List[Dict]):
        """顯示詳細的匹配結果"""
        print(f"\n📊 詳細匹配結果:")
        print("-" * 80)
        
        for i, result in enumerate(results[:10]):  # 只顯示前10個結果
            obj_label = result['object']['primary_label']
            surf_label = result['surface']['label']
            score = result['compatibility_score']
            
            print(f"\n{i+1:2d}. {obj_label:12s} → {surf_label:15s} | 分數: {score:.3f}")
            
            # 顯示前3個最重要的分數
            sorted_scores = sorted(result['detailed_scores'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for key, value in sorted_scores[:3]:
                if key != 'error':
                    print(f"     {key:20s}: {value:.3f}")
    
    def _analyze_matching_quality(self, results: List[Dict]):
        """分析匹配品質"""
        quality_analysis = self.processor.analyze_matching_quality(results)
        
        print(f"\n📈 匹配品質分析:")
        print(f"  總匹配數: {quality_analysis['total_matches']}")
        print(f"  平均分數: {quality_analysis['average_score']:.3f}")
        print(f"  最高分數: {quality_analysis['max_score']:.3f}")
        print(f"  最低分數: {quality_analysis['min_score']:.3f}")
        print(f"  標準差: {quality_analysis['std_score']:.3f}")
        print(f"  整體品質: {quality_analysis['overall_quality']}")
        print(f"  高品質匹配 (≥0.7): {quality_analysis['high_quality_matches']}")
        print(f"  中等品質匹配 (0.4-0.7): {quality_analysis['medium_quality_matches']}")
        print(f"  低品質匹配 (<0.4): {quality_analysis['low_quality_matches']}")
    
    def _show_best_matches(self, results: List[Dict]):
        """顯示最佳匹配"""
        best_matches = self.processor.get_best_matches(results, top_k=5)
        
        print(f"\n🏆 前5個最佳匹配:")
        for i, match in enumerate(best_matches, 1):
            obj_label = match['object']['primary_label']
            surf_label = match['surface']['label']
            score = match['compatibility_score']
            print(f"  {i}. {obj_label:12s} → {surf_label:15s}: {score:.3f}")
    
    def run_all_tests(self):
        """執行所有測試"""
        print("🚀 開始完整的語意匹配測試")
        print("⚠️  使用真實CLIP模型和完整功能")
        
        try:
            # 1. 完整匹配測試
            results = self.test_comprehensive_matching()
            
            # 2. 個別配對測試
            self.test_individual_pairs()
            
            # 3. 邊界情況測試
            self.test_edge_cases()
            
            print("\n" + "="*60)
            print("✅ 所有測試完成")
            print("="*60)
            
            return results
            
        except Exception as e:
            print(f"\n❌ 測試過程中發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            return []


def main():
    """主測試函數"""
    tester = RealSemanticMatchingTester()
    results = tester.run_all_tests()
    return results


if __name__ == "__main__":
    main()

"""
LLM輔助增強系統測試腳本
展示LLM如何在每個階段輔助增強物件置入決策
"""

import torch
import clip
from modules.detection.detection import ObjectDetector
from modules.models.detectron_models import DetectronModel
from modules.validation.vlm.vlm import create_vlm_helper
from modules.matching.llm_enhanced import create_llm_enhanced_matcher
from modules.matching.semantic import find_best_placement
import os
import sys
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

# 載入.env文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 添加項目根目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_separator(title: str):
    """打印分隔線"""
    print("\n" + "="*60)
    print(f" {title} ")
    print("="*60)


def load_image() -> Tuple[Image.Image, np.ndarray]:
    """載入測試圖片"""
    image_path = "data/living-room.jpg"

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到圖片: {image_path}")

    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)

    print(f"✅ 載入圖片: {image_path}")
    print(f"   尺寸: {pil_image.size}")

    return pil_image, image_array


def detect_objects(image: Image.Image) -> Tuple[List[Dict], List[Dict]]:
    """執行物件偵測"""
    print("\n🔍 執行物件偵測...")

    # 初始化Detectron2模型
    detectron_model = DetectronModel()
    detectron_model.load_detectron()
    detector = ObjectDetector(detectron_model)

    # 執行偵測
    detections = detector.detect_objects_in_image(image)

    # 轉換格式
    background_objects = []
    for i, det in enumerate(detections):
        if det['score'] > 0.5:  # 只保留高信心度的偵測
            obj = {
                'label': det['label'],
                'bbox': [int(x) for x in det['box']],  # [x1, y1, x2, y2]
                'confidence': det['score'],
                'id': f"{det['label']}_{i}"
            }
            background_objects.append(obj)

    print(f"✅ 偵測到 {len(background_objects)} 個高信心度物件:")
    for obj in background_objects:
        print(f"   - {obj['label']}: {obj['confidence']:.2f}")

    return detections, background_objects


def get_clip_embedding(text: str, clip_model, device: str) -> np.ndarray:
    """獲取CLIP文本嵌入"""
    with torch.no_grad():
        text_tokens = clip.tokenize([text]).to(device)
        text_embedding = clip_model.encode_text(text_tokens)
        return text_embedding.cpu().numpy()[0]


def print_results(results: List[Dict], title: str):
    """格式化打印結果"""
    print(f"\n{title} ({len(results)} 個結果):")
    print("-" * 50)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('reference_object', 'Unknown')}")
        print(
            f"   分數: {result.get('compatibility_score', result.get('confidence', 0)):.3f}")
        print(f"   來源: {result.get('source', 'Unknown')}")
        print(f"   座標: {result.get('bbox', [])}")

        # 顯示詳細信息
        if 'llm_validation' in result:
            validation = result['llm_validation']
            print(f"   LLM驗證: {validation.get('is_reasonable', 'N/A')}")
            print(f"   驗證信心: {validation.get('overall_confidence', 0):.3f}")

        if 'filter_reason' in result:
            print(f"   篩選原因: {result['filter_reason']}")

        print()


def test_llm_assisted_pipeline(image_array: np.ndarray, object_label: str,
                               background_objects: List[Dict]) -> Dict:
    """測試完整的LLM輔助pipeline"""
    print_separator("LLM輔助增強Pipeline")

    llm_matcher = create_llm_enhanced_matcher()
    if not llm_matcher.is_enabled():
        print("❌ LLM增強功能未啟用")
        return {}

    print(f"🎯 目標物件: {object_label}")
    print(f"📊 原始偵測物件: {len(background_objects)} 個")

    # 展示每個LLM輔助階段
    pipeline_results = {}

    # 階段1: LLM物件篩選
    print("\n" + "🔍" * 20 + " 階段1: LLM物件篩選 " + "🔍" * 20)
    filtered_objects, filter_reasoning = llm_matcher.object_filter.filter_objects(
        background_objects, object_label
    )
    pipeline_results['filtered_objects'] = filtered_objects
    pipeline_results['filter_reasoning'] = filter_reasoning

    print(f"✅ 篩選完成: {len(background_objects)} → {len(filtered_objects)} 個物件")
    print(f"📝 篩選邏輯: {filter_reasoning}")

    # 階段2: LLM增強VLM提示
    print("\n" + "📝" * 20 + " 階段2: LLM增強VLM提示 " + "📝" * 20)
    enhanced_prompt = llm_matcher.prompt_generator.generate_enhanced_prompt(
        filtered_objects, object_label, filter_reasoning
    )
    pipeline_results['enhanced_prompt'] = enhanced_prompt

    print("✅ 增強提示生成完成")
    print(f"📏 提示長度: {len(enhanced_prompt)} 字符")

    # 階段3: VLM分析（使用增強提示）
    print("\n" + "🤖" * 20 + " 階段3: VLM分析（使用增強提示） " + "🤖" * 20)
    from modules.validation.vlm.vlm import create_vlm_helper
    vlm_helper = create_vlm_helper()

    if vlm_helper.is_enabled():
        vlm_suggestions = vlm_helper.get_vlm_suggestions(
            image_array, object_label, custom_prompt=enhanced_prompt
        )
        pipeline_results['vlm_suggestions'] = vlm_suggestions
        print(f"✅ VLM分析完成: {len(vlm_suggestions)} 個建議")
    else:
        print("❌ VLM功能未啟用")
        vlm_suggestions = []

    # 階段4: LLM結果驗證
    print("\n" + "🔍" * 20 + " 階段4: LLM結果驗證 " + "🔍" * 20)
    if vlm_suggestions:
        validated_suggestions = llm_matcher.result_validator.validate_suggestions(
            vlm_suggestions, filtered_objects, object_label
        )
        pipeline_results['validated_suggestions'] = validated_suggestions
        print(f"✅ 結果驗證完成: {len(validated_suggestions)} 個建議已驗證")
    else:
        validated_suggestions = []

    # 階段5: 智能決策融合
    print("\n" + "🔄" * 20 + " 階段5: 智能決策融合 " + "🔄" * 20)

    # 獲取語意匹配結果作為基準
    semantic_matches = llm_matcher._get_semantic_matches(
        object_label, filtered_objects)

    final_results = llm_matcher.decision_fusion.fuse_results(
        semantic_matches, validated_suggestions
    )
    pipeline_results['final_results'] = final_results

    print(f"✅ 決策融合完成: {len(final_results)} 個最終建議")

    return pipeline_results


def show_pipeline_summary(pipeline_results: Dict):
    """顯示pipeline總結"""
    print_separator("LLM輔助Pipeline總結")

    print("📋 各階段成果:")
    print(
        f"1. 物件篩選: {len(pipeline_results.get('filtered_objects', []))} 個適合物件")
    print(
        f"2. 提示增強: {len(pipeline_results.get('enhanced_prompt', ''))} 字符的詳細提示")
    print(
        f"3. VLM分析: {len(pipeline_results.get('vlm_suggestions', []))} 個VLM建議")
    print(
        f"4. 結果驗證: {len(pipeline_results.get('validated_suggestions', []))} 個驗證建議")
    print(f"5. 決策融合: {len(pipeline_results.get('final_results', []))} 個最終建議")

    print("\n🎯 最終建議:")
    final_results = pipeline_results.get('final_results', [])

    for i, result in enumerate(final_results, 1):
        print(f"\n建議 {i}:")
        print(f"  📍 位置: {result.get('reference_object', 'Unknown')}")
        print(
            f"  📊 分數: {result.get('compatibility_score', result.get('confidence', 0)):.3f}")
        print(f"  🏷️ 來源: {result.get('source', 'Unknown')}")
        print(f"  📐 座標: {result.get('bbox', [])}")

        # 顯示LLM驗證信息
        if 'llm_validation' in result:
            validation = result['llm_validation']
            print(f"  ✅ LLM驗證: {validation.get('is_reasonable', 'N/A')}")
            print(f"  🎯 驗證信心: {validation.get('overall_confidence', 0):.3f}")
            if validation.get('reasoning'):
                print(f"  💭 驗證原因: {validation['reasoning']}")

    print("\n💡 LLM輔助價值:")
    print("- 🔍 智能篩選: 過濾不適合的參考物件")
    print("- 📝 提示增強: 提供更詳細的場景上下文")
    print("- 🔍 結果驗證: 多維度評估建議的合理性")
    print("- 🔄 智能融合: 整合多種方法的優勢")


def main():
    """主函數"""
    print("🔍 LLM增強匹配系統比較測試")
    print("=" * 60)

    # 檢查API密鑰
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ 未找到GEMINI_API_KEY")
        return

    print(f"✅ API密鑰: {gemini_key[:10]}...")

    try:
        # 載入圖片
        pil_image, image_array = load_image()

        # 執行物件偵測
        detections, background_objects = detect_objects(pil_image)

        if not background_objects:
            print("❌ 未偵測到任何物件")
            return

        # 初始化CLIP模型
        print("\n📥 載入CLIP模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=device)
        print(f"✅ CLIP模型載入完成")

        # 測試物件
        object_label = "cup"
        print(f"\n🎯 測試物件: {object_label}")

        # 執行LLM輔助pipeline
        pipeline_results = test_llm_assisted_pipeline(
            image_array, object_label, background_objects
        )

        # 顯示pipeline總結
        show_pipeline_summary(pipeline_results)

        print_separator("測試完成")
        print("🎉 LLM輔助系統展示完成！")
        print("\n💡 LLM在每個階段的輔助作用:")
        print("1. 🔍 物件篩選 - 智能過濾不適合的參考物件")
        print("2. 📝 提示增強 - 為VLM提供更詳細的分析指導")
        print("3. 🔍 結果驗證 - 多維度評估VLM建議的合理性")
        print("4. 🔄 決策融合 - 整合多種方法提供最佳建議")

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

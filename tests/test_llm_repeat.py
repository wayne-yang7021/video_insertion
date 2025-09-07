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
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)
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
                'id': f"{det['label']}_{i}",
                'mask': det['mask'].tolist()
            }
            background_objects.append(obj)

    print(f"✅ 偵測到 {len(background_objects)} 個高信心度物件:")
    for obj in background_objects:
        print(f"   - {obj['label']}: {obj['confidence']:.2f}")

    return detections, background_objects


def main():
    """主函數"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("❌ 未找到GEMINI_API_KEY")
        return

    try:
        # 載入圖片
        pil_image, image_array = load_image()

        # 執行物件偵測
        detections, background_objects = detect_objects(pil_image)

        if not background_objects:
            print("❌ 未偵測到任何信心度大於0.5的物件")
            return

        # 初始化CLIP模型
        print("\n📥 載入CLIP模型...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = clip.load("ViT-B/32", device=device)
        print(f"✅ CLIP模型載入完成")

        # 初始化LLM增強匹配器
        llm_enhanced_matcher = create_llm_enhanced_matcher()

        # 載入要插入的物件圖片
        object_path = "data/cola.png"
        if not os.path.exists(object_path):
            print(f"❌ 找不到物件圖片: {object_path}")
            return

        object_img = Image.open(object_path)
        print(f"✅ 載入物件圖片: {object_path}")
        print(f"   尺寸: {object_img.size}")

        # 初始化LLM Placer
        from modules.llm.placer import LLMPlacer
        placer = LLMPlacer()
        print_separator("測試圖片分析放置建議")
        placement_result = placer.suggest_placement_from_images(
            pil_image, object_img)

        if placement_result:
            print("✅ 獲得放置建議")
        else:
            print("❌ 無法獲得放置建議")

        # 測試第二個功能：物件大小建議
        print_separator("測試物件大小建議")

        # 轉換bbox格式為標準化座標
        normalized_objects = []
        img_width, img_height = pil_image.size
        for obj in background_objects:
            bbox = obj['bbox']
            normalized_bbox = [
                bbox[0] / img_width,   # x1
                bbox[1] / img_height,  # y1
                bbox[2] / img_width,   # x2
                bbox[3] / img_height   # y2
            ]
            normalized_objects.append({
                'label': obj['label'],
                'bbox': normalized_bbox,
                'confidence': obj['confidence']
            })

        size_result = placer.suggest_object_size(
            background_image=pil_image,
            object_to_insert=object_img,
            detected_objects=normalized_objects
        )

        if size_result:
            print("✅ 獲得大小建議")
        else:
            print("❌ 無法獲得大小建議")

        # 測試迭代調整功能
        if placement_result and size_result:
            current_placement = placement_result
            current_size = size_result

            # 初始化累積縮放，使用LLM建議的初始大小
            initial_width_scale = current_size['size_suggestion'].get('width_scale', 1.0)
            initial_height_scale = current_size['size_suggestion'].get('height_scale', 1.0)
            cumulative_scale = {'width': initial_width_scale, 'height': initial_height_scale}
            
            print(f"📏 初始縮放比例: {cumulative_scale['width']:.2f} x {cumulative_scale['height']:.2f}")

            # 進行3次迭代調整
            for iteration in range(5):
                print(f"\n🔄 第 {iteration + 1} 次調整...")

                # 畫出位置框
                boxed_image = placer.draw_placement_box(
                    background_image=pil_image,
                    object_to_insert=object_img,
                    placement_result=current_placement,
                    size_result=current_size,
                    cumulative_scale=cumulative_scale
                )

                if boxed_image:
                    # 保存圖片
                    output_path = f"output_iteration_{iteration + 1}.jpg"
                    boxed_image.save(output_path)
                    print(f"� 已保第存圖片: {output_path}")

                    # 獲得調整建議
                    refined_result = placer.refine_placement_and_size(
                        boxed_image=boxed_image,
                        object_to_insert=object_img
                    )

                    if refined_result:
                        print(f"✅ 第 {iteration + 1} 次調整完成")

                        # 更新建議
                        if 'placement_suggestion' in refined_result:
                            current_placement = {
                                'placement_suggestion': refined_result['placement_suggestion']}
                        if 'size_suggestion' in refined_result:
                            new_size = refined_result['size_suggestion']
                            current_size = {
                                'size_suggestion': new_size}

                            # 更新累積縮放
                            new_width_scale = new_size.get('width_scale', 1.0)
                            new_height_scale = new_size.get(
                                'height_scale', 1.0)

                            print(
                                f"📊 LLM建議調整: {new_width_scale:.2f} x {new_height_scale:.2f}")
                            print(
                                f"📊 調整前累積: {cumulative_scale['width']:.2f} x {cumulative_scale['height']:.2f}")

                            # 累積縮放 = 當前累積縮放 * 新的調整比例
                            cumulative_scale['width'] *= new_width_scale
                            cumulative_scale['height'] *= new_height_scale

                            print(
                                f"🔄 調整後累積: {cumulative_scale['width']:.2f} x {cumulative_scale['height']:.2f}")
                    else:
                        print(f"❌ 第 {iteration + 1} 次調整失敗")
                        break
                else:
                    print(f"❌ 第 {iteration + 1} 次畫框失敗")
                    break

            print("\n🎉 迭代調整完成！")
        else:
            print("❌ 缺少位置或大小建議，無法進行迭代測試")

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

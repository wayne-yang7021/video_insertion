# import sys
# import os
# import cv2

# ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
# if ROOT_PATH not in sys.path:
#     sys.path.append(ROOT_PATH)
    
# sys.path.append("external/sam2")  # 讓它找到 sam2 module
# import torch
# import numpy as np
# from PIL import Image
# from external.sam2.sam2.build_sam import build_sam2
# from external.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
# from external.sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# from utils.visualization.visualize_segmentations import visualize_segmentations  # 你前面寫的那個函數

# checkpoint = "checkpoints/sam2.1_hiera_small.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"   # ← 這行是關鍵

# sam_model = build_sam2(model_cfg, checkpoint)
# predictor = SAM2ImagePredictor(sam_model)

# image_path = "data/pictures/living-room.jpg"
# image = Image.open(image_path).convert("RGB")
# image_np = np.array(image)

# predictor.set_image(image_np)

# with torch.no_grad():
#     masks, scores, logits = predictor.predict(
#         point_coords=None,
#         point_labels=None,
#         multimask_output=True
#     )

# results = []
# for i in range(len(masks)):
#     m = masks[i]
#     if m.ndim == 3 and m.shape[0] == 1:
#         m = m[0]
#     if m.dtype != np.bool_:
#         m = m > 0.5
#     results.append({"mask": m, "score": float(scores[i])})

# import json

# def mask_stats_and_crops(image_np, results, save_dir="./output/mask_crops"):
#     os.makedirs(save_dir, exist_ok=True)
#     H, W = image_np.shape[:2]
#     stats = []

#     for i, r in enumerate(results):
#         m = r["mask"]
#         # 標準化成 2D bool，確保索引正確
#         m = np.squeeze(np.array(m))
#         if m.dtype != np.bool_:
#             m = m > 0.5
#         if m.shape != (H, W):
#             m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
#         if not m.any():
#             continue

#         # 面積（像素數）
#         area = int(m.sum())

#         # 外接框
#         ys, xs = np.where(m)
#         x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
#         bbox = [x1, y1, x2, y2]

#         # 裁圖（含視覺化：背景置黑，只保留物件）
#         crop_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR).copy()
#         crop_mask = np.zeros_like(crop_bgr)
#         crop_bgr[~m] = 0
#         crop_mask[m] = crop_bgr[m]
#         crop = crop_mask[y1:y2+1, x1:x2+1]
#         crop_path = os.path.join(save_dir, f"mask_{i:03d}.png")
#         cv2.imwrite(crop_path, crop)

#         stats.append({
#             "id": i,
#             "score": float(r.get("score", np.nan)),
#             "area_px": area,
#             "bbox_xyxy": bbox,
#             "crop_path": crop_path
#         })

#     # 也存成 json 檔，方便檢查
#     with open(os.path.join(save_dir, "mask_summary.json"), "w") as f:
#         json.dump(stats, f, indent=2, ensure_ascii=False)

#     return stats

# image_np_rgb = np.array(image)  # PIL → np RGB（你前面已有）
# stats = mask_stats_and_crops(image_np_rgb, results)
# print(f"共偵測到 {len(stats)} 個 mask")
# for s in stats[:5]:
#     print(s)  # 先看看前幾個


# os.makedirs("./output", exist_ok=True)
# visualize_segmentations(image, results, output_path="./output/sam2_result.jpg")

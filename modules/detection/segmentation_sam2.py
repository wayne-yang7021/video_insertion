# modules/detection/segmentation_sam2.py
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from modules.models.segmentation_model import (
    SAM2Config, load_sam2_predictor, build_amg
)

# ---------------- Dataclasses（你指定的介面） ----------------
@dataclass
class MaskItem:
    id: int
    mask: np.ndarray          # (H,W) bool
    score: float
    bbox_xyxy: Tuple[int,int,int,int]
    crop_path: Optional[str] = None

@dataclass
class SegmentationResult:
    masks: List[MaskItem]
    vis_path: str
    summary_json: str
    crops_dir: str


# ---------------- 視覺化與工具 ----------------
_PALETTE = [
    (255, 99, 71), (30, 144, 255), (60, 179, 113), (255, 165, 0),
    (148, 0, 211), (0, 206, 209), (220, 20, 60), (154, 205, 50),
]

def _to_2d_bool_mask(mask, H, W, thresh=0.5):
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.array(mask)
    mask = np.squeeze(mask)
    if mask.ndim != 2 or mask.shape != (H, W):
        mask = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return (mask > thresh) if mask.dtype != np.bool_ else mask


def _visualize_segmentations(
    image: Image.Image,
    segmentations: List[Dict],
    output_path: str,
    fill_alpha: float = 0.5,
    draw_outline: bool = True,
    outline_thickness: int = 2,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    result = img.copy()

    for i, seg in enumerate(segmentations):
        m = _to_2d_bool_mask(seg["mask"], H, W)
        if not m.any():
            continue
        color = _PALETTE[i % len(_PALETTE)]
        roi_src = result[m]
        roi_col = np.empty_like(roi_src); roi_col[:] = color
        blended = cv2.addWeighted(roi_src, 1.0 - fill_alpha, roi_col, fill_alpha, 0.0)
        result[m] = blended
        if draw_outline:
            mask_u8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, thickness=outline_thickness)
        ys, xs = np.where(m)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            txt = f"#{i}"
            if "score" in seg and seg["score"] is not None:
                txt += f" {seg['score']:.2f}"
            cv2.putText(result, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    cv2.imwrite(output_path, result)
    return output_path


def _stats_and_crops(image_np: np.ndarray, results: List[Dict], save_dir: str) -> List[MaskItem]:
    os.makedirs(save_dir, exist_ok=True)
    H, W = image_np.shape[:2]
    items: List[MaskItem] = []
    for i, r in enumerate(results):
        m = _to_2d_bool_mask(r["mask"], H, W)
        if not m.any():
            continue
        ys, xs = np.where(m)
        area = int(m.sum())
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        crop_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR).copy()
        crop_bgr[~m] = 0
        crop = crop_bgr[y1:y2+1, x1:x2+1]
        crop_path = os.path.join(save_dir, f"mask_{i:03d}.png")
        cv2.imwrite(crop_path, crop)

        items.append(MaskItem(
            id=i,
            mask=m.astype(bool),
            score=float(r.get("score", 0.0)),
            bbox_xyxy=(x1, y1, x2, y2),
            crop_path=crop_path
        ))
    return items


def _grid_prompt_segment(predictor, image_np, points_per_side=24, min_area=200):
    H, W = image_np.shape[:2]
    ys = np.linspace(16, H-16, points_per_side)
    xs = np.linspace(16, W-16, points_per_side)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2).astype(np.float32)

    results = []
    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=grid,
            point_labels=np.ones(len(grid), dtype=np.int32),
            multimask_output=True
        )
    for i in range(len(masks)):
        m = masks[i]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        m_bin = (m > 0.5).astype(np.uint8)
        if m_bin.sum() < min_area:
            continue
        results.append({"mask": m_bin.astype(bool), "score": float(scores[i])})
    return results


# ---------------- 對外主函式 ----------------
def segment_image_sam2(
    image: Image.Image,
    out_dir: str,
    sam2_cfg: Optional[SAM2Config] = None,
    amg_params: Optional[dict] = None,
    grid_params: Optional[dict] = None,
) -> SegmentationResult:
    """
    先嘗試 AMG（若可用）；失敗則用「鋪點」提示 fallback。
    會輸出：
      - 可視化：<out_dir>/sam2_result.jpg
      - 裁圖：  <out_dir>/mask_crops/mask_***.png
      - 摘要：  <out_dir>/mask_crops/mask_summary.json
    並回傳 SegmentationResult。
    """
    os.makedirs(out_dir, exist_ok=True)
    sam2_cfg = sam2_cfg or SAM2Config()
    amg_params = amg_params or dict(
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        box_nms_thresh=0.7,
        crop_n_layers=0,
    )
    grid_params = grid_params or dict(points_per_side=24, min_area=200)

    # 建模與 predictor
    sam_model, predictor = load_sam2_predictor(sam2_cfg)

    # 準備影像
    image_np = np.array(image)
    predictor.set_image(image_np)

    # AMG or fallback
    results: List[Dict]
    amg = build_amg(sam_model, **amg_params) if amg_params is not None else None
    if amg is not None:
        print("➡️ 使用 AutomaticMaskGenerator")
        with torch.no_grad():
            amg_masks = amg.generate(image_np)  # list[dict]
        results = [
            {"mask": m["segmentation"], "score": float(m.get("predicted_iou", 0.0))}
            for m in amg_masks
        ]
    else:
        print("⚠️ 找不到 AMG，改用鋪點提示 fallback")
        results = _grid_prompt_segment(predictor, image_np, **grid_params)

    # 裁圖 + 結構化
    crops_dir = os.path.join(out_dir, "mask_crops")
    mask_items = _stats_and_crops(image_np, results, save_dir=crops_dir)

    # summary.json
    summary_json = os.path.join(crops_dir, "mask_summary.json")
    with open(summary_json, "w") as f:
        json.dump(
            [
                dict(
                    id=m.id,
                    score=m.score,
                    area_px=int(m.mask.sum()),
                    bbox_xyxy=list(m.bbox_xyxy),
                    crop_path=m.crop_path,
                )
                for m in mask_items
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    # 視覺化
    vis_path = os.path.join(out_dir, "sam2_result.jpg")
    _visualize_segmentations(image, results, output_path=vis_path)

    return SegmentationResult(
        masks=mask_items,
        vis_path=vis_path,
        summary_json=summary_json,
        crops_dir=crops_dir,
    )

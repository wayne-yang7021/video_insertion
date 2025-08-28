import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict

_PALETTE = [
    (255, 99, 71),    # tomato
    (30, 144, 255),   # dodger blue
    (60, 179, 113),   # medium sea green
    (255, 165, 0),    # orange
    (148, 0, 211),    # dark violet
    (0, 206, 209),    # dark turquoise
    (220, 20, 60),    # crimson
    (154, 205, 50),   # yellow green
]

def _to_2d_bool_mask(mask, H, W, thresh=0.5):
    # 支援 torch / numpy / list，並 squeeze 到 2D
    if hasattr(mask, "detach"):  # torch tensor
        mask = mask.detach().cpu().numpy()
    mask = np.array(mask)

    # squeeze 除掉單通道
    mask = np.squeeze(mask)

    # 若尺寸跟影像不符，嘗試用最近鄰 resize（避免插值灰階）
    if mask.ndim != 2 or mask.shape != (H, W):
        mask_resized = cv2.resize(mask.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask

    # 轉成 boolean（若已經是 0/1 uint8 也可）
    if mask_resized.dtype == np.bool_:
        m = mask_resized
    else:
        m = mask_resized > thresh
    return m

def visualize_segmentations(
    image: Image.Image,
    segmentations: List[Dict],
    output_path: str = "./output/segmentation_result.jpg",
    fill_alpha: float = 0.5,   # 填色透明度
    draw_outline: bool = True, # 畫邊界更清楚
    outline_thickness: int = 2
) -> None:
    """
    將分割遮罩畫在圖片上：半透明填色 + (可選)邊界。
    segmentations: 每個 dict 至少包含 'mask'；可選 'score'
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # PIL RGB -> OpenCV BGR
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    result = img.copy()

    for i, seg in enumerate(segmentations):
        raw_mask = seg["mask"]
        score = seg.get("score", None)

        # 1) 標準化為 2D bool mask（H×W）
        m = _to_2d_bool_mask(raw_mask, H, W)
        if not m.any():
            continue  # 空遮罩跳過

        # 2) 選顏色（固定調色盤）
        color = _PALETTE[i % len(_PALETTE)]
        color_arr = np.empty_like(result)
        color_arr[:, :] = color  # BGR

        # 3) 單次混合（只混合遮罩內的像素）
        roi_src = result[m]
        roi_col = color_arr[m]
        blended = cv2.addWeighted(roi_src, 1.0 - fill_alpha, roi_col, fill_alpha, 0.0)
        result[m] = blended

        # 4) 畫邊界（更清楚）
        if draw_outline:
            mask_u8 = (m.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, thickness=outline_thickness)

        # 5) (可選)在質心標記 score
        if score is not None:
            ys, xs = np.where(m)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(
                    result, f"{score:.2f}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )

    cv2.imwrite(output_path, result)
    print(f"✅ 分割結果已儲存至：{output_path}")

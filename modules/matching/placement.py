# modules/matching/placement/placement.py
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal

import numpy as np
import torch
import cv2
from PIL import Image

# --- SAM2 predictor 由你的共用載入器提供 ---
from modules.models.segmentation_model import SAM2Config, load_sam2_predictor

# 可選：NMS（也可以用純 numpy 寫）
try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None  # 下面會 fallback 到 numpy 版

# 可選後端：OWL-ViT（transformers）
_OWL_OK = False
try:
    from transformers import OwlViTProcessor, OwlViTForObjectDetection 
    _OWL_OK = True
except Exception:
    pass

# 可選後端：Grounding-DINO
_GDINO_OK = False
try:
    # 依你環境調整 import，這裡提供常見介面占位
    from groundingdino.util.inference import load_model as gdino_load_model, predict as gdino_predict  
    _GDINO_OK = True
except Exception:
    pass


@dataclass
class Candidate:
    id: int
    label: str
    score: float
    box_xyxy: Tuple[int, int, int, int]
    point_xy: Tuple[int, int]
    mask: Optional[np.ndarray] = None   # (H,W) bool
    iou_with_box: Optional[float] = None


def _xywh_to_xyxy(x, y, w, h) -> Tuple[float, float, float, float]:
    return x, y, x + w, y + h


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    # boxes: [N,4] xyxy
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def _run_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if boxes.size == 0:
        return []
    if tv_nms is not None:
        b = torch.from_numpy(boxes.astype(np.float32))
        s = torch.from_numpy(scores.astype(np.float32))
        keep = tv_nms(b, s, iou_thr).cpu().numpy().tolist()
        return keep
    return _nms_numpy(boxes, scores, iou_thr)


# -------------------- 檢索後端：OWL-ViT --------------------
class OWLVitBackend:
    def __init__(self, model_name: str = "google/owlvit-base-patch32", device: Optional[str] = None):
        if not _OWL_OK:
            raise ImportError("transformers 的 OWL‑ViT 未安裝，請先安裝 transformers/torchvision。")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def detect(self, image: Image.Image, queries: List[str], score_thr: float = 0.25):
        # OWL‑ViT 支援多句子；每句裡可以逗號分多詞
        texts = [", ".join(queries)]
        inputs = self.processor(text=texts, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (H,W)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]
        boxes = results["boxes"].cpu().numpy()    # xyxy in pixels
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        # label 對應到輸入 texts 的片段，這裡簡單映射成 queries[0] 內詞彙
        # 為簡潔，先全部標成 queries 中的第一個詞或 "object"
        label_names = []
        for _ in labels:
            label_names.append("object")
        # 過濾
        keep = scores >= score_thr
        return boxes[keep], scores[keep], label_names


# -------------------- 檢索後端：Grounding-DINO --------------------
class GroundingDINObackend:
    def __init__(self, config_path: str, weights_path: str, box_thr: float = 0.25, text_thr: float = 0.25, device: Optional[str] = None):
        if not _GDINO_OK:
            raise ImportError("找不到 Grounding‑DINO 模組，請先安裝或改用 OWL‑ViT 後端。")
        self.model = gdino_load_model(config_path, weights_path)
        self.box_thr = box_thr
        self.text_thr = text_thr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def detect(self, image: Image.Image, queries: List[str], score_thr: float = 0.25):
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        text = ", ".join(queries)
        boxes, logits, phrases = gdino_predict(
            model=self.model,
            image=img_bgr,
            caption=text,
            box_threshold=self.box_thr,
            text_threshold=self.text_thr
        )
        # boxes 是歸一化 xyxy（0~1）
        H, W = image.size[1], image.size[0]
        boxes_px = boxes.copy()
        boxes_px[:, [0, 2]] *= W
        boxes_px[:, [1, 3]] *= H
        scores = logits  # 已是置信度
        labels = [ph if isinstance(ph, str) else "object" for ph in phrases]
        keep = scores >= score_thr
        return boxes_px[keep], scores[keep], [labels[i] for i in np.where(keep)[0]]


# -------------------- 用 SAM2 做 referring segmentation --------------------
def _refine_with_sam2_grid(predictor, image_np: np.ndarray, box_xyxy: Tuple[int, int, int, int],
                           points_per_side: int = 8, min_area: int = 150) -> Optional[np.ndarray]:
    """
    在 bbox 內鋪點當正提示，合成一張乾淨的 mask（union）。
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_np.shape[1]-1, x2), min(image_np.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    xs = np.linspace(x1, x2, points_per_side, endpoint=True)
    ys = np.linspace(y1, y2, points_per_side, endpoint=True)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2).astype(np.float32)

    with torch.no_grad():
        masks, scores, _ = predictor.predict(
            point_coords=grid,
            point_labels=np.ones(len(grid), dtype=np.int32),
            multimask_output=True
        )
    H, W = image_np.shape[:2]
    merged = np.zeros((H, W), dtype=np.uint8)
    for i in range(len(masks)):
        m = masks[i]
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        m_bin = (m > 0.5).astype(np.uint8)
        merged = np.maximum(merged, m_bin)
    if merged.sum() < min_area:
        return None
    return merged.astype(bool)


def _bbox_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _mask_centroid(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0
    return int(xs.mean()), int(ys.mean())


# -------------------- 對外主接口 --------------------
class PlacementProposer:
    """
    open-vocab 檢索 + SAM2 精修，輸出候選插入點。
    """

    def __init__(
        self,
        detector_backend: Literal["owlvit", "groundingdino"] = "owlvit",
        detector_cfg: Optional[dict] = None,
        sam2_cfg: Optional[SAM2Config] = None,
        nms_iou_thr: float = 0.5,
        score_thr: float = 0.25,
        min_area_px: int = 150,
    ):
        self.score_thr = score_thr
        self.nms_iou_thr = nms_iou_thr
        self.min_area_px = min_area_px

        # 檢索器
        detector_cfg = detector_cfg or {}
        if detector_backend == "owlvit":
            if not _OWL_OK:
                raise RuntimeError("選了 OWL‑ViT 但 transformers 未安裝。")
            self.detector = OWLVitBackend(**detector_cfg)
        elif detector_backend == "groundingdino":
            if not _GDINO_OK:
                raise RuntimeError("選了 Grounding‑DINO 但模組未安裝。")
            self.detector = GroundingDINObackend(**detector_cfg)
        else:
            raise ValueError("detector_backend 必須是 'owlvit' 或 'groundingdino'")

        # SAM2
        self.sam2_cfg = sam2_cfg or SAM2Config()
        self.sam_model, self.predictor = load_sam2_predictor(self.sam2_cfg)

    def propose(
        self,
        image: Image.Image,
        queries: List[str],
        max_candidates: int = 20,
        use_mask_centroid: bool = True,
        debug_vis_path: Optional[str] = None
    ) -> List[Candidate]:
        """
        1) open-vocab 檢索出 bbox
        2) bbox 內用 SAM2 鋪點做精修 mask
        3) NMS + 過濾，回傳候選清單（含插入點）
        """
        # 偵測
        boxes_px, scores, labels = self.detector.detect(image, queries, score_thr=self.score_thr)
        if boxes_px.shape[0] == 0:
            return []

        # NMS
        keep = _run_nms(boxes_px.astype(np.float32), scores.astype(np.float32), self.nms_iou_thr)
        boxes_px = boxes_px[keep]
        scores = scores[keep]
        labels = [labels[i] for i in keep]
        if len(scores) > max_candidates:
            idx = np.argsort(-scores)[:max_candidates]
            boxes_px = boxes_px[idx]
            scores = scores[idx]
            labels = [labels[i] for i in idx]

        # SAM2 精修
        image_np = np.array(image)
        self.predictor.set_image(image_np)
        cands: List[Candidate] = []
        H, W = image_np.shape[:2]

        for i, (box, sc, lb) in enumerate(zip(boxes_px, scores, labels)):
            x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            mask = _refine_with_sam2_grid(self.predictor, image_np, (x1, y1, x2, y2), points_per_side=8, min_area=self.min_area_px)
            if mask is not None and mask.sum() >= self.min_area_px:
                px, py = _mask_centroid(mask) if use_mask_centroid else _bbox_center((x1, y1, x2, y2))
                # 簡單 iou: mask bbox vs 原 bbox
                ys, xs = np.where(mask)
                bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                inter_x1, inter_y1 = max(x1, bx1), max(y1, by1)
                inter_x2, inter_y2 = min(x2, bx2), min(y2, by2)
                inter = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
                area_a = (x2 - x1 + 1) * (y2 - y1 + 1)
                area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
                iou = inter / (area_a + area_b - inter + 1e-6)
            else:
                mask = None
                px, py = _bbox_center((x1, y1, x2, y2))
                iou = None

            cands.append(Candidate(
                id=i,
                label=lb,
                score=float(sc),
                box_xyxy=(x1, y1, x2, y2),
                point_xy=(px, py),
                mask=mask,
                iou_with_box=iou
            ))

        # Debug 視覺化（可選）
        if debug_vis_path is not None:
            os.makedirs(os.path.dirname(debug_vis_path), exist_ok=True)
            vis = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            for c in cands:
                x1, y1, x2, y2 = c.box_xyxy
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.circle(vis, c.point_xy, 4, (0, 0, 255), -1)
                txt = f"{c.label}:{c.score:.2f}"
                if c.iou_with_box is not None:
                    txt += f" iou:{c.iou_with_box:.2f}"
                cv2.putText(vis, txt, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                if c.mask is not None:
                    vis[c.mask] = vis[c.mask] * 0.5 + np.array([60, 180, 60]) * 0.5
            cv2.imwrite(debug_vis_path, vis)

        return cands

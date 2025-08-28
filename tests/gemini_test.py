# -*- coding: utf-8 -*-
"""
整合 Detectron + MiDaS 與 Gemini 的 VPP（視覺置入）流程：
1) 你現有流程：偵測 -> 深度 -> 匯出 bboxes 與 bbox_depths
2) Gemini：
   - 先從產品照萃取產品屬性（JSON）
   - 再用背景圖 + 深度圖 + 場景結構（尺寸/偵測框/平均深度/信心）+ 產品屬性 + 你的客製化 prompt
     生成最佳置入位置（bbox/polygon）、理由、光線建議等（JSON）
3) 視覺化：把 Gemini 回傳的 bbox/多邊形直接畫到背景圖上

需求：
- pip install google-generativeai pillow opencv-python
- GEMINI_API_KEY 環境變數
"""

import os
import sys
import json
import re
from typing import Any, Dict, List, Tuple, Union

# ====== 你的專案既有引用 ======
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from modules.models.depth_match_model import DepthEstimator
from modules.matching.depth_matching import DepthHandler

from utils.get_video_first_frame import extract_first_frame
from utils.visualization.visualize_depth import (
    overlay_depth_map, visualize_depth_grayscale, visualize_depth_segmented, draw_bbox_depth_labels
)

from modules.models.detectron_models import DetectronModel
from modules.detection.detection import ObjectDetector
from utils.visualization.visualize_detection import visualize_detections

# ====== 其他常用套件 ======
import cv2
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import google.generativeai as genai

load_dotenv()

# ===================== 基礎工具 =====================
def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

def _coerce_json(text: str) -> Dict[str, Any]:
    """從模型輸出粗暴抽 JSON（即使外面包了文字/反引號，盡量抓第一段 JSON）。"""
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return {}

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ===================== 視覺化工具 =====================
def draw_bbox(image_rgb: np.ndarray,
              bbox_xyxy_norm: List[float],
              label: str = "proposed",
              color: Tuple[int,int,int] = (255, 0, 0),
              alpha: float = 0.25) -> np.ndarray:
    """
    在 RGB 影像上以半透明方式畫出相對座標 bbox（[x1,y1,x2,y2], 0~1）。
    """
    h, w = image_rgb.shape[:2]
    x1 = int(max(0, min(1, bbox_xyxy_norm[0])) * w)
    y1 = int(max(0, min(1, bbox_xyxy_norm[1])) * h)
    x2 = int(max(0, min(1, bbox_xyxy_norm[2])) * w)
    y2 = int(max(0, min(1, bbox_xyxy_norm[3])) * h)

    overlay = image_rgb.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
    blended = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

    cv2.rectangle(blended, (x1, y1), (x2, y2), color, thickness=2)
    cv2.putText(blended, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return blended

def draw_polygon(image_rgb: np.ndarray,
                 polygon_norm: List[List[float]],
                 label: str = "mask",
                 color: Tuple[int,int,int] = (0, 128, 255),
                 alpha: float = 0.25) -> np.ndarray:
    """
    在 RGB 影像上以半透明方式畫出相對座標 polygon（[[x,y], ...], 0~1）。
    """
    h, w = image_rgb.shape[:2]
    pts = []
    for x,y in polygon_norm:
        px = int(max(0, min(1, x)) * w)
        py = int(max(0, min(1, y)) * h)
        pts.append([px, py])
    pts = np.array(pts, dtype=np.int32)

    overlay = image_rgb.copy()
    cv2.fillPoly(overlay, [pts], color)
    blended = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

    cv2.polylines(blended, [pts], isClosed=True, color=color, thickness=2)
    if pts.size > 0:
        x1, y1 = pts[0]
        cv2.putText(blended, label, (x1, max(0, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return blended

def draw_point(image_rgb: np.ndarray,
               center_norm: List[float],
               label: str = "best-point",
               color: Tuple[int,int,int] = (255, 64, 0),
               radius: int = 8,
               thickness: int = -1) -> np.ndarray:
    """
    在 RGB 影像上畫出相對座標中心點（[x,y], 0~1）。
    """
    h, w = image_rgb.shape[:2]
    x = int(max(0, min(1, center_norm[0])) * w)
    y = int(max(0, min(1, center_norm[1])) * h)

    out = image_rgb.copy()
    cv2.circle(out, (x, y), radius, color, thickness)
    cv2.circle(out, (x, y), radius+3, color, 2)
    cv2.putText(out, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

#============================================================
def _clip_int(v, lo, hi):  # 小工具
    return int(max(lo, min(hi, v)))

def compute_depth_stats(depth_map: np.ndarray, box_xyxy_abs: List[float]) -> Dict[str, float]:
    """對單一 bbox 的深度做統計（使用原始深度值，不正規化）。"""
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy_abs]
    x1 = _clip_int(x1, 0, w-1); x2 = _clip_int(x2, 0, w-1)
    y1 = _clip_int(y1, 0, h-1); y2 = _clip_int(y2, 0, h-1)
    if x2 <= x1 or y2 <= y1:
        return {"min": 0, "p10": 0, "median": 0, "p90": 0, "max": 0, "mean": 0, "std": 0}

    crop = depth_map[y1:y2, x1:x2].astype(np.float32)
    # 避免 NaN 影響
    crop = crop[np.isfinite(crop)]
    if crop.size == 0:
        return {"min": 0, "p10": 0, "median": 0, "p90": 0, "max": 0, "mean": 0, "std": 0}

    return {
        "min": float(np.min(crop)),
        "p10": float(np.percentile(crop, 10)),
        "median": float(np.median(crop)),
        "p90": float(np.percentile(crop, 90)),
        "max": float(np.max(crop)),
        "mean": float(np.mean(crop)),
        "std": float(np.std(crop)),
    }

def make_object_depth_thumbnails(
    depth_map: np.ndarray,
    detections: List[Dict[str, Any]],
    img_size: Tuple[int, int],
    max_tiles: int = 24,
    tile_size: int = 128,
) -> List[Tuple[str, Image.Image]]:
    """
    為每個偵測框裁切深度區塊 -> 以「每塊自身 min-max」正規化為 0~255 灰階縮圖（tile），
    並回傳 [(描述文字, PIL.Image), ...]。描述包含 id/label/conf/bbox_rel/depth_stats。
    為避免成本，預設最多送 24 個。
    """
    W, H = img_size
    # 以 bbox 面積大到小排序，優先送大的
    items = []
    for i, det in enumerate(detections):
        box = ensure_xyxy(det["box"])
        area = max(0.0, (box[2]-box[0]) * (box[3]-box[1]))
        items.append((i, det, box, area))
    items.sort(key=lambda x: x[3], reverse=True)
    items = items[:max_tiles]

    tiles = []
    for i, det, box_abs, _ in items:
        # 統計
        stats = compute_depth_stats(depth_map, box_abs)
        # 相對座標
        box_rel = normalize_box_xyxy(box_abs, W, H)

        # 產生縮圖（以 crop 自己的 min/max 做對比拉伸）
        x1, y1, x2, y2 = [int(round(v)) for v in box_abs]
        x1 = _clip_int(x1, 0, W-1); x2 = _clip_int(x2, 0, W-1)
        y1 = _clip_int(y1, 0, H-1); y2 = _clip_int(y2, 0, H-1)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = depth_map[y1:y2, x1:x2].astype(np.float32)
        crop = crop[np.isfinite(crop)]
        if crop.size == 0:
            continue
        # 重新抓一次完整矩形（含非有限值），便於可視化
        crop_full = depth_map[y1:y2, x1:x2].astype(np.float32)
        mn, mx = np.min(crop), np.max(crop)
        norm = (crop_full - mn) / (mx - mn + 1e-6)
        tile = (norm * 255.0).clip(0, 255).astype(np.uint8)
        tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tile_pil = Image.fromarray(tile)

        label = det.get("label") or det.get("name") or "object"
        conf = float(det.get("score") or det.get("confidence") or 0.0)
        desc = json.dumps({
            "type": "object_depth_tile",
            "id": i,
            "label": label,
            "confidence": round(conf, 6),
            "bbox_xyxy": [round(v, 6) for v in box_rel],
            "depth_stats": stats
        }, ensure_ascii=False)
        tiles.append((desc, tile_pil))
    return tiles




# ===================== Gemini 設定與提示 =====================
API_KEY = os.getenv("GEMINI_API_KEY")
# API_KEY = "AIzaSyBYJtg5_W8gf5ZfkQzmsQJk56ZMzFvEsJU"
if not API_KEY:
    raise RuntimeError("請先設定環境變數 GEMINI_API_KEY")

DEFAULT_MODEL = "gemini-1.5-pro"  # 也可用 "gemini-1.5-pro"
genai.configure(api_key=API_KEY)

def _build_model(temperature: float = 0.4, top_p: float = 0.95, top_k: int = 40, max_tokens: int = 2048):
    generation_config = genai.GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_tokens,
    )
    return genai.GenerativeModel(
        model_name=DEFAULT_MODEL,
        generation_config=generation_config,
        # 可選：讓輸出偏向 JSON（若 SDK 版本支援，可取消註解）
        # system_instruction="回覆必須是純 JSON，不要額外說明文字。",
    )

PRODUCT_EXTRACT_PROMPT = """
你是一位產品識別專家。只根據下面的「產品照片」分析該物件屬性，並輸出**純 JSON**。

請輸出欄位：
- object_name: (字符串)
- category: (字符串)
- brand: (字符串或空字串)
- colors: (字串陣列)
- materials: (字串陣列)
- shape: (字符串)
- size_hint: (字符串)
- text_on_package: (字串陣列)
- notable_features: (字串陣列)
- orientation_constraints: (字符串)
- fragility: (字符串)
"""

# PLACEMENT_PROMPT_TEMPLATE = """
# 你是 VPP（Visual Product Placement）助手。請基於「背景照片、深度圖、場景結構 JSON、產品屬性 JSON」，
# 輸出**純 JSON**的置入建議。回答繁體中文。

# [場景結構 JSON]
# - 這是一段由系統提供的 JSON，包含背景圖尺寸、偵測框、相對座標與平均深度、信心等。
# - 你要以相對座標（0~1）來回傳最終建議位置，並確保與背景圖尺寸對應。

# [產品屬性 JSON]
# - 這是一段由系統提供的 JSON，描述要置入的物件屬性、品牌可見性等。

# [使用者客製化需求]
# {user_prompt}

# [輸出 JSON 結構]
# {{
#   "product": {{
#     "object_name": "<string>",
#     "category": "<string>",
#     "brand": "<string>"
#   }},
#   "best_placement": {{
#     "location_description": "<string>",
#     "reasoning": "<string>",
#     "bbox_xyxy": [<float 0-1>, <float 0-1>, <float 0-1>, <float 0-1>],   // [x1,y1,x2,y2] 相對座標
#     "polygon": [[<float>,<float>], ...] | null,                           // 可選，相對座標
#     "rotation_degrees": <float> | 0,
#     "scale_hint": "<string>",                                             // 例：「約佔寬度 8–10%」
#     "size_estimate": {{                       
#       "width_ratio": <float>,                                             
#       "height_ratio": <float>                                             
#     }},
#     "brand_visibility": <float>,                                          // 0~1
#     "occlusion_risk": <float>,                                            // 0~1
#     "confidence": <float>,                                                // 0~1
#     "lighting_notes": "<string>"
#   }},
#   "alternatives": [
#     {{
#       "location_description": "<string>",
#       "reasoning": "<string>",
#       "bbox_xyxy": [<float 0-1>, <float 0-1>, <float 0-1>, <float 0-1>],
#       "polygon": [[<float>,<float>], ...] | null,
#       "rotation_degrees": <float> | 0,
#       "scale_hint": "<string>",
#       "size_estimate": { "width_ratio": <float>, "height_ratio": <float> },  
#       "brand_visibility": <float>,
#       "occlusion_risk": <float>,
#       "confidence": <float>,
#       "lighting_notes": "<string>"
#     }}
#   ],
#   "avoid_zones": [
#     {{
#       "area_description": "<string>",
#       "reason": "<string>"
#     }}
#   ],
#   "summary_zh": "<string>"
# }}

# [重要規則]
# - 僅輸出上述 JSON，不要加解釋或程式碼區塊標記。
# - 你會看到每個偵測物件的「深度縮圖」（對應 bbox 區塊的灰階深度），以及場景 JSON 的深度統計（min/median/p90 等）。
# - 請利用「深度縮圖 + 統計 + bbox 面積」推斷這些物件在場景中的相對距離與大致尺寸，據此估計要置入產品的合理實體大小與畫面佔比（size_estimate）。
# - 若場景存在明顯參考物（例如桌子、沙發、螢幕等），優先用它們作為比例尺來計算建議寬高佔比。
# - 深度圖僅為輔助：較亮=較近（若你判斷不同，請明確說明），避免把物件放在會穿插遮擋的區域。
# - 優先選擇：不要放在已經有東西的位置、不遮擋主體、品牌清楚、光線佳、構圖自然。
# """
PLACEMENT_PROMPT_TEMPLATE = """
你是 VPP（Visual Product Placement）助手。請基於「背景照片、深度圖、場景結構 JSON、產品屬性 JSON、每個物件的深度縮圖」，
輸出**純 JSON**，只需回傳**單一最適置入點**（該點視為要置入物的中心）。回答繁體中文。

[場景結構 JSON]
- 系統提供：背景圖尺寸、偵測框、相對座標、深度統計、信心等。
- 你回傳的中心點必須是相對座標（0~1，對應背景圖）。

[產品屬性 JSON]
- 系統提供：要置入之產品的屬性（品牌、形狀、可視性需求、擺放限制…）。

[使用者客製化需求]
{user_prompt}

[輸出 JSON 結構]
{{
  "product": {{
    "object_name": "<string>",
    "category": "<string>",
    "brand": "<string>"
  }},
  "best_point": {{
    "center": [<float 0-1>, <float 0-1>],             // 單一最適置入點（中心點，相對座標）
    "reasoning": "<string>",                           // 為何選這個點：不遮擋主體、品牌可見、光線佳、自然、動線等
    "support_surface": "<string|null>",                // 若落在桌面/地面/層板等，請敘述；否則 null
    "brand_visibility": <float>,                       // 0~1，越高越清楚
    "occlusion_risk": <float>,                         // 0~1，越高越容易被遮擋
    "confidence": <float>,                             // 0~1，整體建議信心
    "lighting_notes": "<string>"                       // 光線/反光/陰影建議
  }},
  "avoid_zones": [
    {{ "area_description": "<string>", "reason": "<string>" }}
  ],
  "summary_zh": "<string>"
}}

[重要規則]
- 僅輸出上述 JSON，請勿多加解釋或程式碼區塊標記。
- 你會看到每個偵測物件的「深度縮圖」與深度統計（min/median/p90 等）。
- 請利用「深度縮圖 + 統計 + bbox/面積/位置」判斷遮擋風險與可用空間，選出**單一最佳中心點**。
- 嚴格避免落在已被其他物件佔據的位置（若無法確定，寧可偏向空曠平面），避免靠近人臉/螢幕/敏感主體。
- 優先在可支撐物體的平面（桌面、地面、層板）上；構圖自然、品牌可見度高、光線良好為佳。
"""


# ===================== 轉換/準備資料 =====================
def to_pil(image_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def depth_to_pil(depth_map: np.ndarray, target_size: Tuple[int,int]=None) -> Image.Image:
    """把深度 map 正規化到 0~255，轉灰階 PIL，必要時 resize。"""
    dm = depth_map.astype(np.float32)
    dm = dm - np.nanmin(dm)
    rng = np.nanmax(dm) + 1e-6
    dm = (dm / rng * 255.0).clip(0, 255).astype(np.uint8)
    if target_size:
        dm = cv2.resize(dm, target_size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(dm)

def normalize_box_xyxy(bbox_xyxy_abs: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = bbox_xyxy_abs
    return [x1/width, y1/height, x2/width, y2/height]

def ensure_xyxy(box: Union[List[float], Tuple[float,float,float,float], Dict[str,float]]) -> List[float]:
    """支援多種 bbox 格式 -> [x1,y1,x2,y2]"""
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    if isinstance(box, dict):
        keys = ["x1","y1","x2","y2"]
        if all(k in box for k in keys):
            return [float(box[k]) for k in keys]
    raise ValueError(f"無法解析 bbox 格式: {box}")

def build_scene_context_json(
    width: int,
    height: int,
    detections: List[Dict[str, Any]],
    bbox_depths: List[float],
    depth_map: np.ndarray,  
) -> Dict[str, Any]:
    items = []
    for i, det in enumerate(detections):
        label = det.get("label") or det.get("name") or "object"
        score = float(det.get("score") or det.get("confidence") or 0.0)
        box_abs = ensure_xyxy(det.get("box"))
        box_rel = normalize_box_xyxy(box_abs, width, height)

        stats = compute_depth_stats(depth_map, box_abs)
        x1, y1, x2, y2 = box_rel
        items.append({
            "id": i,
            "label": label,
            "confidence": score,
            "bbox_xyxy_abs": box_abs,
            "bbox_xyxy": [round(v, 6) for v in box_rel],
            "bbox_xywh": [round(x1, 6), round(y1, 6), round(max(0.0, x2-x1), 6), round(max(0.0, y2-y1), 6)],
            "avg_depth": float(bbox_depths[i]) if i < len(bbox_depths) else None,
            "depth_stats": stats,  # ← 新增
            "area_ratio": round(max(0.0, (box_abs[2]-box_abs[0])*(box_abs[3]-box_abs[1]) / (width*height)), 6)
        })
    return {
        "image_size": {"width": width, "height": height},
        "depth_is_brighter_if_closer": True,  # 可視需要保留
        "objects": items
    }


# ===================== Gemini 兩階段呼叫 =====================
def gemini_extract_product_attributes(product_pil: Image.Image) -> Dict[str, Any]:
    model = _build_model()
    resp = model.generate_content([PRODUCT_EXTRACT_PROMPT, product_pil])
    print("Gemini response:", resp)
    return _coerce_json((resp.text or "").strip())

def gemini_propose_placement(
    background_pil: Image.Image,
    depth_pil: Image.Image,
    product_pil: Image.Image,
    scene_context_json: Dict[str, Any],
    product_attrs_json: Dict[str, Any],
    user_prompt: str = "",
    object_depth_tiles: List[Tuple[str, Image.Image]] = None,  # ← 新增
) -> Dict[str, Any]:
    model = _build_model()
    scene_text = _pretty(scene_context_json)
    product_text = _pretty(product_attrs_json)

    prompt = PLACEMENT_PROMPT_TEMPLATE.replace(
        "{user_prompt}", user_prompt.strip() or "（無特別補充）"
    )


    parts = [
        "以下是場景結構 JSON（請閱讀）：\n" + scene_text,
        "以下是產品屬性 JSON（請閱讀）：\n" + product_text,
        "請依規格，產生純 JSON 輸出：",
        prompt,
        background_pil,
        depth_pil,
        product_pil,
    ]

    # ← 新增：把每個物件的深度縮圖附上（文字描述 + 圖像）
    if object_depth_tiles:
        parts.append("以下依序提供各偵測物件的『深度縮圖』與其描述 JSON：")
        for meta_text, tile_img in object_depth_tiles:
            parts.extend([meta_text, tile_img])

    resp = model.generate_content(parts)
    return _coerce_json((resp.text or "").strip())


# ===================== 主流程（整合你的 Detectron + MiDaS） =====================
def run_pipeline_with_gemini(
    product_image_path: str,
    background_image_path: str,
    user_prompt: str,
    out_dir: str = "./output",
) -> Dict[str, Any]:
    """
    1) 用你現有的 Detectron/MiDaS 做偵測與深度
    2) 丟給 Gemini：產品屬性 -> 置入建議（回傳 bbox/polygon）
    3) 視覺化結果存檔
    """
    _ensure_dir(out_dir)

    # ---- 初始化你現有的模型 ----
    detectron_model = DetectronModel()
    detectron_model.load_detectron()
    detector = ObjectDetector(detectron_model)

    depth_model = DepthEstimator()
    depth_handler = DepthHandler(depth_model)

    # ---- 讀背景圖（BGR -> RGB/PIL）----
    # 讀影片的第一幀
    # img_bgr = extract_first_frame("data/videos/house_tour.mp4")
    # img_bgr = np.array(img_bgr)  # Add this line to convert PIL.Image to np.ndarray

    # 讀照片
    img_bgr = cv2.imread(background_image_path)
    if img_bgr is None:
        raise FileNotFoundError(background_image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(img_rgb)
    H, W = img_rgb.shape[:2]

    # ---- 偵測物件 ----
    detections = detector.detect_objects_in_image(image_pil)
    visualize_detections(image_pil, detections, output_path=os.path.join(out_dir, "sample_detected.jpg"))

    # ---- 取出 bboxes ----
    bboxes = [ensure_xyxy(det["box"]) for det in detections]

    # ---- 深度圖與各 bbox 平均深度 ----
    depth_map = depth_handler.estimate_depth_map(image_pil)
    bbox_depths = depth_handler.get_bbox_depths(depth_map, bboxes)
    print("bbox depths:", bbox_depths)

    # ---- 視覺化（可選）----
    vis_overlay = overlay_depth_map(img_rgb, depth_map)
    cv2.imwrite(os.path.join(out_dir, "vis_overlay_magma.jpg"), cv2.cvtColor(vis_overlay, cv2.COLOR_RGB2BGR))

    vis_gray = visualize_depth_grayscale(depth_map)
    cv2.imwrite(os.path.join(out_dir, "vis_gray.jpg"), cv2.cvtColor(vis_gray, cv2.COLOR_RGB2BGR))

    vis_segmented = visualize_depth_segmented(depth_map)
    cv2.imwrite(os.path.join(out_dir, "vis_segmented.jpg"), cv2.cvtColor(vis_segmented, cv2.COLOR_RGB2BGR))

    vis_labeled = draw_bbox_depth_labels(img_rgb, bboxes, bbox_depths)
    cv2.imwrite(os.path.join(out_dir, "vis_labeled.jpg"), cv2.cvtColor(vis_labeled, cv2.COLOR_RGB2BGR))

    # ---- 準備要送給 Gemini 的場景 JSON ----
    scene_ctx = build_scene_context_json(W, H, detections, bbox_depths, depth_map)

    # ---- 讀產品照（PIL）----
    product_pil = Image.open(product_image_path).convert("RGB")

    # ---- 轉深度圖為 PIL 灰階 ----
    depth_pil = depth_to_pil(depth_map, target_size=(W, H))

    # ---- 階段 1：讓 Gemini 抽取產品屬性 ----
    product_attrs = gemini_extract_product_attributes(product_pil)
    # 最基本容錯（避免 key 缺失）
    product_attrs.setdefault("object_name", "")
    product_attrs.setdefault("category", "")
    product_attrs.setdefault("brand", "")
    # ---- 建立每個物件的深度縮圖（最多 24 個，避免成本爆炸）----
    object_depth_tiles = make_object_depth_thumbnails(
        depth_map=depth_map,
        detections=detections,
        img_size=(W, H),
        max_tiles=24,
        tile_size=128,
    )

    # ---- 階段 2：讓 Gemini 生成置入建議（含最佳 bbox 與備選）----
    placement = gemini_propose_placement(
        background_pil=image_pil,
        depth_pil=depth_pil,
        product_pil=product_pil,
        scene_context_json=scene_ctx,
        product_attrs_json=product_attrs,
        user_prompt=user_prompt,
        object_depth_tiles=object_depth_tiles
    )

    # ---- 解析回傳，畫到背景圖上 ----
    result_img = img_rgb.copy()
    best = (placement or {}).get("best_point") or {}
    # bbox = best.get("bbox_xyxy")
    # polygon = best.get("polygon")

    # if isinstance(bbox, list) and len(bbox) == 4:
    #     result_img = draw_bbox(result_img, bbox, label="best")
    # if isinstance(polygon, list) and len(polygon) >= 3:
    #     result_img = draw_polygon(result_img, polygon, label="best-mask")
    center = best.get("center")

    if isinstance(center, (list, tuple)) and len(center) == 2:
        result_img = draw_point(result_img, center, label="best-point")
    else:
        print("[WARN] 模型未回傳有效的 best_point.center，略過視覺化。")

    # 畫備選
    for i, alt in enumerate((placement or {}).get("alternatives") or []):
        bbox_a = alt.get("bbox_xyxy")
        poly_a = alt.get("polygon")
        if isinstance(bbox_a, list) and len(bbox_a) == 4:
            result_img = draw_bbox(result_img, bbox_a, label=f"alt{i+1}", color=(0, 200, 0), alpha=0.18)
        if isinstance(poly_a, list) and len(poly_a) >= 3:
            result_img = draw_polygon(result_img, poly_a, label=f"alt{i+1}-mask", color=(0, 200, 200), alpha=0.18)

    # ---- 存檔 ----
    cv2.imwrite(os.path.join(out_dir, "vpp_point.jpg"), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

    # 也把 JSON 結果存起來
    with open(os.path.join(out_dir, "vpp_point.json"), "w", encoding="utf-8") as f:
        json.dump(placement, f, ensure_ascii=False, indent=2)

    print("輸出：")
    print(" - 偵測與深度視覺化：sample_detected.jpg, vis_overlay_magma.jpg, vis_gray.jpg, vis_segmented.jpg, vis_labeled.jpg")
    print(" - 置入結果可視化：vpp_point.jpg")
    print(" - 置入 JSON：vpp_point.json")

    return placement

def get_best_placement_point(
    product_image_path: str,
    background_image_path: str,
    user_prompt: str = "",
    out_dir: str = "./output",
) -> Tuple[float, float]:
    """
    跑完整管線，但只回傳 Gemini 選中的最佳置入中心點 (cx, cy)，相對座標 0~1。
    若模型沒有給有效中心點，丟出 RuntimeError。
    """
    placement = run_pipeline_with_gemini(
        product_image_path=product_image_path,
        background_image_path=background_image_path,
        user_prompt=user_prompt,
        out_dir=out_dir,
    )
    center = (placement or {}).get("best_point", {}).get("center")
    if not (isinstance(center, (list, tuple)) and len(center) == 2):
        raise RuntimeError("Gemini 未回傳有效的 best_point.center")
    cx, cy = float(center[0]), float(center[1])
    # 夾在 [0,1] 之內，避免偶發超界
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    return cx, cy



# ===================== 範例執行 =====================
if __name__ == "__main__":
    # 你可以把這些路徑換成你的專案資料
    product_path = "data/pictures/starbucks.png"         # 要置入的產品照
    background_path = "data/pictures/classroom.jpg"           # 背景照（你原本程式就讀了這張）

    # placement_json = run_pipeline_with_gemini(
    #     product_image_path=product_path,
    #     background_image_path=background_path,
    #     user_prompt="",
    #     out_dir="./output"
    # )
    # print(_pretty(placement_json))
    cx, cy = get_best_placement_point(
        product_image_path = product_path,
        background_image_path = background_path,
        user_prompt="",  # 可填你的客製需求
        out_dir="./output",
    )
    print("Best point (relative):", cx, cy)

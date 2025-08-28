# modules/models/segmentation_model.py
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# 讓它找到 external/sam2
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
SAM2_ROOT = os.path.join(PROJECT_ROOT, "external", "sam2")
if SAM2_ROOT not in sys.path:
    sys.path.append(SAM2_ROOT)

# SAM2 主要匯入
from external.sam2.sam2.build_sam import build_sam2
from external.sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
# AMG 容錯匯入
AMG = None
try:
    from external.sam2.sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator as AMG  # type: ignore
except Exception:
    try:
        from segment_anything import SamAutomaticMaskGenerator as AMG  # type: ignore
    except Exception:
        AMG = None


@dataclass
class SAM2Config:
    checkpoint: str = "checkpoints/sam2.1_hiera_small.pt"
    # 用「config name」最穩：sam2/configs/** 內的相對名稱
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_s"
    device: Optional[str] = None  # "cuda" | "mps" | "cpu" | None(自動選)
    precision: str = "fp32"       # "fp32" | "bf16" (依需求可擴充)


def _auto_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize_model_cfg(cfg: str) -> str:
    """
    Hydra 會把 'sam2.1/sam2.1_hiera_s' 當成 config-name。
    有些人會給 'configs/sam2.1/sam2.1_hiera_s.yaml'，這裡幫你轉成前者。
    """
    cfg = cfg.strip()
    if cfg.endswith(".yaml"):
        # 嘗試抽掉前綴 'configs/' 或 'sam2/configs/' 等
        parts = cfg.replace("\\", "/").split("/")
        # 取最後兩段：sam2.1/sam2.1_hiera_s.yaml
        if len(parts) >= 2:
            cfg = "/".join(parts[-2:])
        if cfg.endswith(".yaml"):
            cfg = cfg[:-5]
    return cfg


def load_sam2_predictor(cfg: SAM2Config) -> Tuple[object, SAM2ImagePredictor]:
    """
    建 SAM2 model 與 predictor。回傳 (sam_model, predictor)
    """
    device = cfg.device or _auto_device()
    model_cfg = _normalize_model_cfg(cfg.model_cfg)

    # SAM2 本身會選擇裝置；部分 fork 支援傳 device，維持簡單先交給內部處理
    sam_model = build_sam2(model_cfg, cfg.checkpoint)
    predictor = SAM2ImagePredictor(sam_model)

    # 對 Apple / bf16 的小優化
    if device == "mps":
        torch.set_float32_matmul_precision("high")
    if cfg.precision.lower() == "bf16" and torch.cuda.is_available():
        torch.set_default_dtype(torch.bfloat16)  # 選擇性

    return sam_model, predictor


def build_amg(sam_model, **kwargs):
    """
    回傳 AMG 實例（若存在），否則回傳 None。
    kwargs 會傳到 AutomaticMaskGenerator 建構子（points_per_side 等）。
    """
    if AMG is None:
        return None
    return AMG(sam_model, **kwargs)

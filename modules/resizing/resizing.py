"""
Size Advisor Pipeline for placement scale decision - Fixed Version
- Uses a small LLM via Ollama (e.g., gemma3:4b) as a *judge* to choose/adjust scale.
- Works with VLM too if the model supports image input (provide `panel_path`).
- Geometry & constraints are enforced locally (verifier) to avoid hallucinations.

Dependencies: pip install pydantic requests numpy opencv-python
"""
from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Annotated

import numpy as np
import requests
from pydantic import BaseModel, Field, ValidationError


# ============================ Data Models (Fixed) ============================ #
class Candidate(BaseModel):
    name: str
    value: float  # px per meter OR a width in px (depends on type)
    confidence: Annotated[float, Field(ge=0, le=1)]  # 修復：使用 Annotated
    type: str = Field(description="'px_per_m' or 'width_px'")

class SceneMeta(BaseModel):
    H: int
    W: int
    fx: Optional[float] = None
    fy: Optional[float] = None

class Placement(BaseModel):
    xy: Annotated[List[int], Field(min_length=2, max_length=2)]  # 修復：使用 List + Field
    plane_normal: Annotated[List[float], Field(min_length=3, max_length=3)]  # 修復
    mean_depth: Optional[float] = None
    edge_safety_px_width: Optional[float] = None

class DetectedObject(BaseModel):
    class_name: str
    bbox: Annotated[List[int], Field(min_length=4, max_length=4)]  # 修復
    score: Annotated[float, Field(ge=0, le=1)]  # 修復

class TargetItem(BaseModel):
    type: str = Field(description="'sticker' | '3d' | etc.")
    real_size_m: Optional[Dict[str, float]] = None  # {"w_m": 0.09, "h_m": 0.09}

class ScenePack(BaseModel):
    scene_meta: SceneMeta
    placement: Placement
    objects: List[DetectedObject]
    candidates: List[Candidate]
    target_item: TargetItem

class SizeAdvice(BaseModel):
    px_per_m: Optional[float] = None
    final_width_px: Optional[float] = None
    sources: List[str] = []
    notes: Optional[str] = None


# =========================== 與你現有 pipeline 整合的介面 =========================== #
def create_scene_from_your_pipeline(
    image_shape: tuple,  # (H, W)
    detectron_results: List[Dict],  # 你的 detectron 結果
    depth_map: np.ndarray,  # 你的 midas 深度圖
    placement_xy: tuple,  # 你選定的最佳座標點
    target_object_type: str = "sticker",  # 要置入的物體類型
    target_real_size: Optional[Dict[str, float]] = None  # {"w_m": 0.09, "h_m": 0.09}
) -> ScenePack:
    """
    將你現有的 pipeline 結果轉換成 ScenePack 格式
    """
    H, W = image_shape
    
    # 1. 轉換 detectron 結果
    objects = []
    for det in detectron_results:
        obj = DetectedObject(
            class_name=det['class'],
            bbox=[int(x) for x in det['bbox']],  # [x1, y1, x2, y2]
            score=float(det['score'])
        )
        objects.append(obj)
    
    # 2. 從深度圖獲取placement信息
    x, y = placement_xy
    mean_depth = float(depth_map[y, x])  # 注意座標順序
    
    # 3. 估算邊緣安全距離
    edge_safety_px_width = min(
        abs(x - 0), abs(W - x),  # 到左右邊距離
        abs(y - 0), abs(H - y)   # 到上下邊距離
    ) * 0.8  # 保守一點
    
    # 4. 生成候選尺度
    candidates = generate_candidates(objects, depth_map, placement_xy, edge_safety_px_width)
    
    # 5. 建立 ScenePack
    scene_pack = ScenePack(
        scene_meta=SceneMeta(H=H, W=W),
        placement=Placement(
            xy=[x, y],
            plane_normal=[0.0, 0.0, 1.0],  # 假設水平平面，你可以改進這部分
            mean_depth=mean_depth,
            edge_safety_px_width=edge_safety_px_width
        ),
        objects=objects,
        candidates=candidates,
        target_item=TargetItem(
            type=target_object_type,
            real_size_m=target_real_size
        )
    )
    
    return scene_pack


def generate_candidates(
    objects: List[DetectedObject], 
    depth_map: np.ndarray, 
    placement_xy: tuple,
    edge_safety_px_width: float
) -> List[Candidate]:
    """
    基於檢測到的物體和深度圖生成候選尺度
    """
    candidates = []
    x, y = placement_xy
    target_depth = depth_map[y, x]
    
    # 人體先驗候選
    for obj in objects:
        if obj.class_name == 'person':
            x1, y1, x2, y2 = obj.bbox
            person_height_px = y2 - y1
            person_depth = depth_map[int((y1+y2)/2), int((x1+x2)/2)]
            
            # 假設人高170cm，計算 px_per_m
            assumed_height_m = 1.70
            px_per_m = person_height_px / assumed_height_m
            
            # 根據深度差異調整
            depth_ratio = person_depth / target_depth
            adjusted_px_per_m = px_per_m * depth_ratio
            
            candidates.append(Candidate(
                name="human_prior_px_per_m",
                value=adjusted_px_per_m,
                confidence=0.7 * obj.score,
                type="px_per_m"
            ))
    
    # 邊緣安全寬度候選
    candidates.append(Candidate(
        name="edge_safety_px_width",
        value=edge_safety_px_width,
        confidence=0.9,
        type="width_px"
    ))
    
    # 可以加入更多啟發式候選...
    
    return candidates


# ========================= Ollama Client (保持不變) ============================= #
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, system: str, user_json_prompt: str, 
                images: Optional[List[str]] = None, format_json: bool = True, 
                temperature: float = 0.0, max_tokens: int = 256) -> str:
        """Calls Ollama /api/generate with optional images (VLM)."""
        url = f"{self.base_url}/api/generate"
        data: Dict = {
            "model": model,
            "system": system,
            "prompt": user_json_prompt,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if format_json:
            data["format"] = "json"
        if images:
            data["images"] = images
        
        try:
            r = requests.post(url, json=data, timeout=120)
            r.raise_for_status()
            # Ollama streams by default; collect the 'response' fields
            text = ""
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                obj = json.loads(line)
                if "response" in obj:
                    text += obj["response"]
            return text
        except Exception as e:
            print(f"Ollama request failed: {e}")
            return "{}"


# ========================= 其他類別保持不變 =========================== #
class SizeVerifier:
    def __init__(self, min_px: int = 8, max_px: int = 4096):
        self.min_px = min_px
        self.max_px = max_px

    def clamp(self, advice: SizeAdvice, scene: ScenePack) -> SizeAdvice:
        edge = scene.placement.edge_safety_px_width
        if advice.final_width_px is not None:
            w = float(advice.final_width_px)
            if edge is not None:
                w = min(w, float(edge))
            w = float(max(self.min_px, min(self.max_px, w)))
            advice.final_width_px = w
        return advice


class SizeAdvisor:
    def __init__(self, model: str = "gemma3:4b", ollama_url: str = "http://localhost:11434"):
        self.client = OllamaClient(ollama_url)
        self.model = model
        self.verifier = SizeVerifier()

    def advise(self, scene_pack: ScenePack, panel_image_b64: Optional[str] = None) -> SizeAdvice:
        # 簡化的系統提示
        system = (
            "你是尺寸決策器。從候選尺度中選擇最適合的，輸出 JSON 格式："
            "{\"px_per_m\": float?, \"final_width_px\": float?, \"sources\": [...], \"notes\": \"...\"}"
        )
        
        prompt = scene_pack.model_dump_json()
        out = self.client.generate(
            model=self.model,
            system=system,
            user_json_prompt=prompt,
            images=[panel_image_b64] if panel_image_b64 else None,
            format_json=True,
            temperature=0.0,
            max_tokens=256,
        )
        
        try:
            data = json.loads(out)
            advice = SizeAdvice(**data)
        except Exception as e:
            # Fallback: basic heuristic if model fails
            advice = self._fallback(scene_pack)
            advice.notes = f"fallback due to parse error: {e}"
        
        # Clamp to hard constraints
        advice = self.verifier.clamp(advice, scene_pack)
        
        # If px_per_m exists and target has real size, compute width if missing
        if (advice.final_width_px is None and advice.px_per_m is not None 
            and scene_pack.target_item.real_size_m):
            w_m = float(scene_pack.target_item.real_size_m.get("w_m", 0))
            if w_m > 0:
                advice.final_width_px = float(advice.px_per_m) * w_m
                advice = self.verifier.clamp(advice, scene_pack)
        return advice

    def _fallback(self, scene_pack: ScenePack) -> SizeAdvice:
        # Simple robust default: use edge safety * 0.4 scaled by depth
        edge = scene_pack.placement.edge_safety_px_width or 128.0
        depth = scene_pack.placement.mean_depth or 2.0
        scale = 1.0 / (1.0 + 0.15 * max(0.0, depth - 2.0))
        width = edge * 0.4 * scale
        return SizeAdvice(
            px_per_m=None, 
            final_width_px=float(width), 
            sources=["fallback"], 
            notes="fallback heuristic"
        )


# ============================== 整合你的 Pipeline ================================ #
def integrate_with_your_pipeline_example():
    """
    示範如何與你現有的 pipeline 整合
    """
    # 假設你有這些來自現有 pipeline 的數據：
    image_shape = (1080, 1920)  # H, W
    
    # 你的 detectron 結果 (需要轉換格式)
    detectron_results = [
        {'class': 'person', 'bbox': [420, 280, 640, 980], 'score': 0.92},
        {'class': 'chair', 'bbox': [100, 400, 300, 800], 'score': 0.85}
    ]
    
    # 你的 midas 深度圖 (模擬)
    depth_map = np.random.rand(1080, 1920) * 5.0 + 1.0  # 1-6米深度
    
    # 你找到的最佳置入座標
    placement_xy = (820, 640)
    
    # 要置入的物體資訊
    target_real_size = {"w_m": 0.09, "h_m": 0.09}  # 9cm x 9cm 貼紙
    
    # 1. 轉換成 ScenePack
    scene_pack = create_scene_from_your_pipeline(
        image_shape=image_shape,
        detectron_results=detectron_results,
        depth_map=depth_map,
        placement_xy=placement_xy,
        target_object_type="sticker",
        target_real_size=target_real_size
    )
    
    # 2. 獲取尺寸建議
    advisor = SizeAdvisor(model="gemma3:4b")
    
    try:
        advice = advisor.advise(scene_pack)
        print("尺寸建議:")
        print(f"  最終寬度: {advice.final_width_px:.1f} px")
        print(f"  px per meter: {advice.px_per_m:.1f}" if advice.px_per_m else "  px per meter: N/A")
        print(f"  來源: {advice.sources}")
        print(f"  備註: {advice.notes}")
        
        return advice.final_width_px
        
    except Exception as e:
        print(f"無法連接 Ollama 或處理失敗: {e}")
        # 使用 fallback
        return scene_pack.placement.edge_safety_px_width * 0.4


if __name__ == "__main__":
    final_size_px = integrate_with_your_pipeline_example()
    print(f"\n最終建議尺寸: {final_size_px:.1f} 像素")
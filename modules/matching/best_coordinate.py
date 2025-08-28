import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.morphology import disk
from typing import Tuple, List, Dict

class OptimalPlacementDetector:
    """
    大幅強化版：最佳置入點檢測器
    - 穩定性與平坦度：使用更健壯的法向量/曲率與鄰域一致性(向量場方差)
    - 可達性：納入遮擋、邊界距離、中心性、形狀薄弱區(細長/尖角)抑制
    - 視覺顯著：頻譜殘差顯著度 + 對比/紋理/邊緣密度
    - 語義：保留可擴展 hook
    - 多準則融合：採用「遮罩內穩健分位數正規化」+ 動態權重回退
    - 候選點：面積約束 + Non-maximum suppression + 深度/空間去重
    - 置信度：局部分數、跨準則一致性與邊界安全係數
    """

    def __init__(
        self,
        stability_weight: float = 0.3,
        flatness_weight: float = 0.25,
        accessibility_weight: float = 0.2,
        visibility_weight: float = 0.15,
        semantic_weight: float = 0.1,
        debug: bool = False,
    ):
        self.weights = {
            "stability": stability_weight,
            "flatness": flatness_weight,
            "accessibility": accessibility_weight,
            "visibility": visibility_weight,
            "semantic": semantic_weight,
        }
        self.debug = debug

    # ============================= Public API ============================= #
    def find_optimal_placement_points(
        self,
        mask: np.ndarray,
        depth_map: np.ndarray,
        image: np.ndarray,
        object_label: str,
        top_k: int = 5,
    ) -> List[Dict]:
        mask = (mask > 0).astype(np.uint8)
        if mask.ndim != 2:
            raise ValueError("mask 必須是二值 2D 陣列")
        if depth_map.shape[:2] != mask.shape:
            raise ValueError("depth_map 與 mask 尺寸不一致")
        if image.shape[:2] != mask.shape:
            raise ValueError("image 與 mask 尺寸不一致")

        object_region = self._extract_object_region(mask, depth_map, image)
        surface = self._multi_scale_surface_analysis(object_region)

        stability_map = self._compute_stability_map(object_region, surface)
        flatness_map = self._compute_flatness_map(object_region, surface)
        accessibility_map = self._compute_accessibility_map(object_region)
        visibility_map = self._compute_visibility_map(object_region)
        semantic_map = self._compute_semantic_suitability(object_region, object_label)

        if self.debug:
            print("✅ DEBUG: mask pixels:", int(mask.sum()))
            d_mask = object_region["depth"][mask > 0]
            if d_mask.size:
                print("✅ DEBUG: depth range:", float(d_mask.min()), "~", float(d_mask.max()))

        composite = self._fuse_criteria_maps(
            stability_map, flatness_map, accessibility_map, visibility_map, semantic_map, mask
        )

        if self.debug:
            valid = composite[mask > 0]
            if valid.size:
                print(
                    "✅ DEBUG: composite max/min/mean:",
                    float(valid.max()),
                    float(valid.min()),
                    float(valid.mean()),
                )

        candidates = self._generate_candidate_points(composite, object_region, top_k)
        validated = self._validate_and_refine_points(candidates, composite, object_region)
        return validated[:top_k]

    # ============================ Core Stages ============================ #
    def _extract_object_region(self, mask: np.ndarray, depth_map: np.ndarray, image: np.ndarray) -> Dict:
        masked_depth = np.where(mask > 0, depth_map, 0)
        masked_image = np.where(mask[..., None] > 0, image, 0)

        # 簡單補洞(inpaint) 以避免梯度/法向量在缺失處爆炸
        missing = ((mask == 0) | (depth_map <= 0) | np.isnan(depth_map)).astype(np.uint8)
        depth_f32 = depth_map.astype(np.float32)
        depth_f32[np.isnan(depth_f32)] = 0
        if missing.any():
            depth_inpaint = cv2.inpaint(depth_f32, missing, 3, cv2.INPAINT_TELEA)
        else:
            depth_inpaint = depth_f32

        y_coords, x_coords = np.where(mask > 0)
        depths = masked_depth[y_coords, x_coords]
        colors = masked_image[y_coords, x_coords]
        return {
            "mask": mask,
            "depth": masked_depth,
            "depth_inpaint": depth_inpaint,
            "image": masked_image,
            "points_3d": np.column_stack([x_coords, y_coords, depths]),
            "colors": colors,
            "x_coords": x_coords,
            "y_coords": y_coords,
        }

    def _multi_scale_surface_analysis(self, obj: Dict) -> Dict:
        depth = obj["depth_inpaint"].astype(np.float32)
        mask = obj["mask"].astype(bool)

        # 一階梯度
        fx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

        normals = self._estimate_surface_normals(depth, mask)
        curv = self._compute_principal_curvatures(depth, mask)

        scales = [1, 2, 4, 6]
        ms = {}
        for s in scales:
            sm = gaussian(depth, sigma=s, preserve_range=True).astype(np.float32)
            sm *= mask
            ms[f"smoothed_{s}"] = sm
            win = 2 * s + 1
            # 局部變化(robust variance via median abs dev approximation)
            med = ndimage.median_filter(sm, size=win)
            mad = ndimage.median_filter(np.abs(sm - med), size=win) + 1e-6
            ms[f"roughness_{s}"] = (mad * mask).astype(np.float32)
        return {"fx": fx, "fy": fy, "normals": normals, "curvatures": curv, "multi_scale": ms}

    def _estimate_surface_normals(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        fx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        # 以圖像坐標近似：n = (-fx, -fy, 1)
        n = np.stack([-fx, -fy, np.ones_like(depth, np.float32)], axis=-1)
        norm = np.linalg.norm(n, axis=2, keepdims=True)
        norm[norm == 0] = 1.0
        n = n / norm
        n *= mask[..., None]
        return n.astype(np.float32)

    def _compute_principal_curvatures(self, depth: np.ndarray, mask: np.ndarray) -> Dict:
        # 二階導數
        fxx = cv2.Sobel(depth, cv2.CV_32F, 2, 0, ksize=3)
        fyy = cv2.Sobel(depth, cv2.CV_32F, 0, 2, ksize=3)
        fxy = cv2.Sobel(cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3), cv2.CV_32F, 0, 1, ksize=3)
        fx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        fy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        denom = (1 + fx * fx + fy * fy)
        denom2 = denom * denom + 1e-8
        denom32 = np.power(denom, 1.5) + 1e-6
        K = (fxx * fyy - fxy * fxy) / denom2
        H = (((1 + fy * fy) * fxx - 2 * fx * fy * fxy + (1 + fx * fx) * fyy) / (2 * denom32))
        return {"gaussian": (K * mask).astype(np.float32), "mean": (H * mask).astype(np.float32)}

    # ========================= Criterion: Stability ======================= #
    def _compute_stability_map(self, obj: Dict, surf: Dict) -> np.ndarray:
        mask = obj["mask"].astype(np.float32)
        normals = surf["normals"]  # (H,W,3)

        # 偏好「水平可放置面」=> 法向 z 分量越大越水平 (以相機座標近似)
        # 這比原始以 y 當重力更合理
        z_align = np.clip(normals[..., 2], 0.0, 1.0)  # 0~1

        # 低曲率更穩
        H = np.abs(surf["curvatures"]["mean"]) * mask
        flat = np.exp(-5.0 * H).astype(np.float32) * mask

        # 支撐面積：以半徑 r 的盤狀核累積有效像素
        support = self._compute_local_support_area(mask)

        # 邊界安全：距離邊緣越遠越穩
        edge_dist = self._edge_distance(mask)

        stability = (0.4 * z_align + 0.3 * flat + 0.2 * support + 0.1 * edge_dist) * mask
        return stability.astype(np.float32)

    def _compute_local_support_area(self, mask: np.ndarray, r: int = 6) -> np.ndarray:
        ker = disk(r).astype(np.float32)
        area = ndimage.convolve(mask.astype(np.float32), ker, mode="constant", cval=0.0)
        return self._robust_norm(area, mask)

    def _edge_distance(self, mask: np.ndarray) -> np.ndarray:
        edt = ndimage.distance_transform_edt(mask > 0)
        if edt.max() <= 0:
            return np.zeros_like(mask, dtype=np.float32)
        return (edt / (edt.max() + 1e-6)).astype(np.float32)

    # ========================= Criterion: Flatness ======================== #
    def _compute_flatness_map(self, obj: Dict, surf: Dict) -> np.ndarray:
        mask = obj["mask"].astype(bool)
        normals = surf["normals"]
        depth = obj["depth_inpaint"].astype(np.float32)

        # 法向量一致性：使用局部向量場方差 (避免 for 迴圈)
        win = 5
        pad = win // 2
        nx, ny, nz = [normals[..., i] for i in range(3)]
        mean_nx = ndimage.uniform_filter(nx, size=win)
        mean_ny = ndimage.uniform_filter(ny, size=win)
        mean_nz = ndimage.uniform_filter(nz, size=win)
        cos_sim = nx * mean_nx + ny * mean_ny + nz * mean_nz
        cos_sim = np.clip(cos_sim, -1, 1) * mask

        # 深度平滑 (拉普拉斯小 => 平滑)
        lap = cv2.Laplacian(depth, cv2.CV_32F)
        depth_smooth = np.exp(-np.abs(lap)) * mask

        # 多尺度粗糙度 (越小越平坦)
        rough_terms = []
        for k, v in surf["multi_scale"].items():
            if k.startswith("roughness_"):
                rough_terms.append(v)
        if rough_terms:
            rough = np.mean(rough_terms, axis=0)
            multi_flat = np.exp(-self._robust_norm(rough, mask)) * mask
        else:
            multi_flat = np.ones_like(depth) * mask

        flatness = (0.5 * self._robust_norm(cos_sim, mask) + 0.3 * self._robust_norm(depth_smooth, mask) + 0.2 * self._robust_norm(multi_flat, mask))
        return flatness.astype(np.float32)

    # ======================== Criterion: Accessibility ==================== #
    def _compute_accessibility_map(self, obj: Dict) -> np.ndarray:
        mask = obj["mask"].astype(np.float32)
        depth = obj["depth_inpaint"].astype(np.float32)

        # 中心性 (離質心近通常更可達)
        cy, cx = ndimage.center_of_mass(mask)
        yy, xx = np.ogrid[:mask.shape[0], :mask.shape[1]]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        if np.any(mask > 0):
            maxd = dist[mask > 0].max()
            center_access = (1 - dist / (maxd + 1e-6)) * mask
        else:
            center_access = mask

        # 遮擋：梯度大 => 可能有遮擋/折線，降低可達
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, 3)
        grad_mag = np.sqrt(gx * gx + gy * gy)
        occlusion = self._robust_norm(grad_mag, mask)
        occlusion = np.clip(occlusion, 0, 1)

        # 幾何寬裕度：遠離窄頸/尖角 (以距離變換 + 細化近似)
        edt = self._edge_distance(mask)
        geom_margin = edt

        accessibility = (0.4 * self._robust_norm(center_access, mask) + 0.4 * (1 - occlusion) + 0.2 * self._robust_norm(geom_margin, mask)) * mask
        return accessibility.astype(np.float32)

    # ========================= Criterion: Visibility ====================== #
    def _compute_visibility_map(self, obj: Dict) -> np.ndarray:
        mask = obj["mask"].astype(np.float32)
        img = obj["image"].astype(np.uint8)
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.astype(np.uint8)

        # 顏色/亮度對比：局部均值差
        blur = cv2.blur(gray, (7, 7))
        contrast = np.abs(gray.astype(np.float32) - blur.astype(np.float32)) * mask

        # 紋理：Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture = np.abs(lap) * mask

        # 邊緣密度
        edges = cv2.Canny(gray, 50, 150).astype(np.float32)
        density = cv2.blur(edges, (7, 7)) * mask

        # 頻譜殘差顯著圖 (簡化版)
        sal = self._spectral_residual_saliency(gray).astype(np.float32) * mask

        visibility = (
            0.25 * self._robust_norm(contrast, mask)
            + 0.25 * self._robust_norm(texture, mask)
            + 0.25 * self._robust_norm(density, mask)
            + 0.25 * self._robust_norm(sal, mask)
        )
        return visibility.astype(np.float32)

    def _spectral_residual_saliency(self, gray: np.ndarray) -> np.ndarray:
        # Hou & Zhang 2007 風格的快速近似
        gray_f = gray.astype(np.float32)
        gray_f = cv2.resize(gray_f, (max(64, gray.shape[1] // 2), max(64, gray.shape[0] // 2)))
        eps = 1e-6
        fft = np.fft.fft2(gray_f)
        log_amp = np.log(np.abs(fft) + eps)
        phase = np.angle(fft)
        avg = cv2.blur(log_amp, (3, 3))
        res = log_amp - avg
        sal = np.abs(np.fft.ifft2(np.exp(res + 1j * phase)))
        sal = np.square(np.abs(sal))
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
        sal = cv2.resize(sal.astype(np.float32), (gray.shape[1], gray.shape[0]))
        return sal

    # ========================= Criterion: Semantic ======================== #
    def _compute_semantic_suitability(self, obj: Dict, label: str) -> np.ndarray:
        label = (label or "").lower()
        rules = {
            "table": self._table_semantic,
            "desk": self._table_semantic,
            "chair": self._chair_semantic,
            "shelf": self._shelf_semantic,
            "floor": self._floor_semantic,
            "wall": self._wall_semantic,
        }
        fn = rules.get(label, self._default_semantic)
        return fn(obj).astype(np.float32)

    def _table_semantic(self, obj: Dict) -> np.ndarray:
        mask = obj["mask"]
        depth = obj["depth_inpaint"]
        if mask.sum() == 0:
            return mask.astype(np.float32)
        dvals = depth[mask > 0]
        th = np.percentile(dvals, 25)
        return ((depth <= th) & (mask > 0)).astype(np.float32)

    def _chair_semantic(self, obj: Dict) -> np.ndarray:
        return obj["mask"].astype(np.float32)

    def _shelf_semantic(self, obj: Dict) -> np.ndarray:
        return obj["mask"].astype(np.float32)

    def _floor_semantic(self, obj: Dict) -> np.ndarray:
        return obj["mask"].astype(np.float32)

    def _wall_semantic(self, obj: Dict) -> np.ndarray:
        return obj["mask"].astype(np.float32)

    def _default_semantic(self, obj: Dict) -> np.ndarray:
        return obj["mask"].astype(np.float32)

    # ========================= Fusion & Scoring =========================== #
    def _fuse_criteria_maps(
        self,
        stability: np.ndarray,
        flatness: np.ndarray,
        accessibility: np.ndarray,
        visibility: np.ndarray,
        semantic: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        maps = {
            "stability": stability,
            "flatness": flatness,
            "accessibility": accessibility,
            "visibility": visibility,
            "semantic": semantic,
        }
        mask_bool = mask > 0

        norm_maps = {}
        for k, m in maps.items():
            norm_maps[k] = self._robust_norm(m.astype(np.float32), mask_bool)

        # 若某圖全為 0，動態回退其權重
        weights = self.weights.copy()
        total_w = 0.0
        for k in list(weights.keys()):
            if np.all(norm_maps[k][mask_bool] == 0):
                weights[k] = 0.0
            total_w += weights[k]
        if total_w == 0:
            total_w = 1.0
        for k in weights:
            weights[k] /= total_w

        comp = (
            weights["stability"] * norm_maps["stability"]
            + weights["flatness"] * norm_maps["flatness"]
            + weights["accessibility"] * norm_maps["accessibility"]
            + weights["visibility"] * norm_maps["visibility"]
            + weights["semantic"] * norm_maps["semantic"]
        ) * mask_bool
        return comp.astype(np.float32)

    def _robust_norm(self, arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """在 mask 內用分位數做穩健 [0,1] 正規化，抑制極端值。"""
        m = arr[mask > 0]
        if m.size == 0:
            return np.zeros_like(arr, dtype=np.float32)
        lo = np.percentile(m, 1)
        hi = np.percentile(m, 99)
        if hi - lo < 1e-6:
            return np.zeros_like(arr, dtype=np.float32)
        out = (arr - lo) / (hi - lo)
        out = np.clip(out, 0, 1)
        out *= (mask > 0)
        return out.astype(np.float32)

    # ====================== Candidates & Validation ======================= #
    def _generate_candidate_points(self, score: np.ndarray, obj: Dict, top_k: int) -> List[Dict]:
        mask = obj["mask"]
        # 僅在 mask 內尋找局部極大；避免靠邊 => 侵蝕 1~2 像素
        safe_mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        labels = safe_mask.astype(np.uint8)
        if labels.sum() == 0:
            labels = mask.astype(np.uint8)

        # 使用 footprint 控制 NMS 範圍
        footprint = disk(9)
        peaks = peak_local_max(
            score,
            min_distance=8,
            threshold_abs=0.15,
            labels=labels,
            exclude_border=False,
            footprint=footprint,
        )

        H, W = score.shape
        cand = []
        for (y, x) in peaks:
            if 0 <= x < W and 0 <= y < H and mask[y, x] > 0:
                cand.append({
                    "position": (int(x), int(y)),
                    "score": float(score[y, x]),
                    "index": len(cand),
                })
        # 先多取一些以利後續去重
        cand.sort(key=lambda c: c["score"], reverse=True)
        return cand[: max(top_k * 5, 20)]

    def _validate_and_refine_points(self, cands: List[Dict], score: np.ndarray, obj: Dict) -> List[Dict]:
        mask = obj["mask"]
        depth = obj["depth"]
        refined = []
        used = np.zeros_like(mask, dtype=np.uint8)
        H, W = mask.shape

        for c in cands:
            x, y = c["position"]
            if not (0 <= x < W and 0 <= y < H) or mask[y, x] == 0:
                continue
            # 局部重心(soft-argmax) 以提升亞像素穩定
            rx, ry = self._soft_refine(x, y, score, mask, r=4)

            # 距離已選點過近則跳過 (去重)
            if used[ry, rx] > 0:
                continue
            cv2.circle(used, (rx, ry), 10, 1, thickness=-1)

            conf = self._placement_confidence((rx, ry), score, obj)
            analysis = self._placement_analysis((rx, ry), obj)
            refined.append({
                "position": (rx, ry),
                "score": float(score[ry, rx]),
                "confidence": float(conf),
                "depth": float(depth[ry, rx]),
                "analysis": analysis,
                "original_candidate": c,
            })
        # 依置信度與分數綜合排序
        refined.sort(key=lambda d: (d["confidence"], d["score"]), reverse=True)
        return refined

    def _soft_refine(self, x: int, y: int, score: np.ndarray, mask: np.ndarray, r: int = 4) -> Tuple[int, int]:
        H, W = mask.shape
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        win = score[y0:y1, x0:x1] * (mask[y0:y1, x0:x1] > 0)
        if win.size == 0 or win.max() <= 0:
            return x, y
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = win / (win.sum() + 1e-6)
        sx = int(np.clip((xx * w).sum(), x0, x1 - 1))
        sy = int(np.clip((yy * w).sum(), y0, y1 - 1))
        return sx, sy

    def _placement_confidence(self, pos: Tuple[int, int], score: np.ndarray, obj: Dict) -> float:
        x, y = pos
        mask = obj["mask"]
        H, W = mask.shape
        r = 6
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        local = score[y0:y1, x0:x1][mask[y0:y1, x0:x1] > 0]
        if local.size == 0:
            return 0.0
        loc_mean = float(local.mean())
        loc_max = float(local.max())
        edge_safe = float(self._edge_distance(mask)[y, x])
        return 0.5 * loc_mean + 0.3 * loc_max + 0.2 * edge_safe

    def _placement_analysis(self, pos: Tuple[int, int], obj: Dict) -> Dict:
        x, y = pos
        depth = float(obj["depth"][y, x])
        return {
            "position_3d": (x, y, depth),
            "local_properties": {"depth": depth, "is_valid": bool(obj["mask"][y, x] > 0)},
            "geometric_analysis": {"surface_type": "unknown", "orientation": "likely_horizontal"},
            "recommendations": {"object_types": ["small_objects", "decorative_items"], "placement_strategy": "stable_horizontal"},
        }

# modules/placement/optimal_placement_detector.py
import numpy as np
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.morphology import disk
from typing import Tuple, List, Dict

class OptimalPlacementDetector:
    """
    學術創新的最佳置入點檢測器
    結合多層次幾何分析、表面法向量估計、穩定性評估和視覺顯著性分析
    """
    
    def __init__(self, 
                 stability_weight: float = 0.3,
                 flatness_weight: float = 0.25, 
                 accessibility_weight: float = 0.2,
                 visibility_weight: float = 0.15,
                 semantic_weight: float = 0.1):
        """
        初始化檢測器參數
        Args:
            stability_weight: 穩定性權重
            flatness_weight: 平坦度權重  
            accessibility_weight: 可達性權重
            visibility_weight: 視覺可見性權重
            semantic_weight: 語義適合性權重
        """
        self.weights = {
            'stability': stability_weight,
            'flatness': flatness_weight,
            'accessibility': accessibility_weight,
            'visibility': visibility_weight,
            'semantic': semantic_weight
        }
        
    def find_optimal_placement_points(self, 
                                    mask: np.ndarray, 
                                    depth_map: np.ndarray,
                                    image: np.ndarray,
                                    object_label: str,
                                    top_k: int = 5) -> List[Dict]:
        """
        主要函數：找到物件內的最佳置入點
        
        Args:
            mask: 物件的二值化遮罩 (H, W)
            depth_map: 深度圖 (H, W)
            image: 原始RGB圖像 (H, W, 3)
            object_label: 物件標籤（用於語義分析）
            top_k: 返回前k個最佳點
            
        Returns:
            List[Dict]: 包含位置、分數和分析結果的字典列表
        """
        
        # 1. 預處理和區域提取
        object_region = self._extract_object_region(mask, depth_map, image)
        
        # 2. 多尺度表面分析
        surface_analysis = self._multi_scale_surface_analysis(object_region)
        
        # 3. 穩定性分析（基於物理約束）
        stability_map = self._compute_stability_map(object_region, surface_analysis)
        
        # 4. 平坦度分析（表面法向量一致性）
        flatness_map = self._compute_flatness_map(object_region, surface_analysis)
        
        # 5. 可達性分析（3D幾何約束）
        accessibility_map = self._compute_accessibility_map(object_region, mask)
        
        # 6. 視覺顯著性分析
        visibility_map = self._compute_visibility_map(object_region, image, mask)
        
        # 7. 語義適合性分析
        semantic_map = self._compute_semantic_suitability(object_region, object_label)
        

        print("✅ DEBUG: mask nonzero pixels:", np.sum(mask))
        print("✅ DEBUG: depth range in masked area:", np.min(depth_map[mask>0]), " ~ ", np.max(depth_map[mask>0])) 
        # 8. 多準則決策融合
        composite_score_map = self._fuse_criteria_maps(
            stability_map, flatness_map, accessibility_map, 
            visibility_map, semantic_map
        )

        print("✅ DEBUG: composite score max:", composite_score_map.max(), 
            "min:", composite_score_map.min(),
            "mean:", np.mean(composite_score_map[composite_score_map > 0]))

        # 9. 候選點生成和優化
        candidate_points = self._generate_candidate_points(
            composite_score_map, object_region, top_k
        )
        
        # 10. 後處理和驗證
        validated_points = self._validate_and_refine_points(
            candidate_points, object_region, mask
        )
        
        return validated_points[:top_k]
    
    def _extract_object_region(self, mask: np.ndarray, depth_map: np.ndarray, 
                              image: np.ndarray) -> Dict:
        """提取物件區域的多模態信息"""
        masked_depth = depth_map * mask
        masked_image = image * mask[:, :, np.newaxis]
        
        # 計算物件的3D點雲
        y_coords, x_coords = np.where(mask > 0)
        depths = masked_depth[y_coords, x_coords]
        colors = masked_image[y_coords, x_coords]
        
        return {
            'mask': mask,
            'depth': masked_depth,
            'image': masked_image,
            'points_3d': np.column_stack([x_coords, y_coords, depths]),
            'colors': colors,
            'x_coords': x_coords,
            'y_coords': y_coords
        }
    
    def _multi_scale_surface_analysis(self, object_region: Dict) -> Dict:
        """多尺度表面幾何分析"""
        depth = object_region['depth']
        mask = object_region['mask']
        
        # 計算梯度和曲率
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        # 表面法向量估計
        normals = self._estimate_surface_normals(depth, mask)
        
        # 主曲率計算
        principal_curvatures = self._compute_principal_curvatures(depth, mask)
        
        # 多尺度平滑分析
        scales = [1, 3, 5, 7]
        multi_scale_features = {}
        
        for scale in scales:
            smoothed = gaussian(depth, sigma=scale)
            smoothed *= mask
            multi_scale_features[f'smoothed_{scale}'] = smoothed
            
            # 計算各尺度下的局部變化
            local_variance = ndimage.generic_filter(
                smoothed, np.var, size=2*scale+1
            ) * mask
            multi_scale_features[f'variance_{scale}'] = local_variance
        
        return {
            'gradients': (grad_x, grad_y),
            'normals': normals,
            'curvatures': principal_curvatures,
            'multi_scale': multi_scale_features
        }
    
    def _estimate_surface_normals(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """估計表面法向量"""
        # 使用深度梯度估計法向量
        grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        # 構建法向量 (注意深度方向)
        normals = np.zeros((depth.shape[0], depth.shape[1], 3))
        normals[:, :, 0] = -grad_x  # x方向
        normals[:, :, 1] = -grad_y  # y方向  
        normals[:, :, 2] = 1        # z方向（深度）
        
        # 正規化
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        norm[norm == 0] = 1  # 避免除以零
        normals = normals / norm
        
        return normals * mask[:, :, np.newaxis]
    
    def _compute_principal_curvatures(self, depth: np.ndarray, mask: np.ndarray) -> Dict:
        """計算主曲率"""
        # 計算二階偏導數
        fxx = cv2.Sobel(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3), cv2.CV_64F, 1, 0, ksize=3)
        fyy = cv2.Sobel(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3), cv2.CV_64F, 0, 1, ksize=3)
        fxy = cv2.Sobel(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3), cv2.CV_64F, 0, 1, ksize=3)
        
        fx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        fy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
        
        # 高斯曲率和平均曲率
        denominator = (1 + fx**2 + fy**2)**2
        denominator[denominator == 0] = 1e-8
        
        gaussian_curvature = (fxx * fyy - fxy**2) / denominator
        mean_curvature = ((1 + fy**2) * fxx - 2 * fx * fy * fxy + (1 + fx**2) * fyy) / (2 * denominator**(3/2))
        
        return {
            'gaussian': gaussian_curvature * mask,
            'mean': mean_curvature * mask
        }
    
    def _compute_stability_map(self, object_region: Dict, surface_analysis: Dict) -> np.ndarray:
        """計算穩定性地圖（基於物理約束）"""
        mask = object_region['mask']
        depth = object_region['depth']
        normals = surface_analysis['normals']
        
        # 1. 重力方向一致性（法向量與重力方向的對齊）
        gravity_vector = np.array([0, 1, 0])  # 假設y軸向下為重力方向
        gravity_alignment = np.abs(np.dot(normals, gravity_vector))
        
        # 2. 局部平坦度（基於曲率）
        mean_curvature = np.abs(surface_analysis['curvatures']['mean'])
        flatness_score = np.exp(-5 * mean_curvature)  # 低曲率 = 高平坦度
        
        # 3. 支撐面積評估
        support_area = self._compute_local_support_area(depth, mask)
        
        # 4. 邊緣距離（避免邊緣位置）
        edge_distance = self._compute_edge_distance(mask)
        
        # 組合穩定性分數
        stability = (
            0.4 * gravity_alignment +
            0.3 * flatness_score + 
            0.2 * support_area +
            0.1 * edge_distance
        ) * mask
        
        return stability
    
    def _compute_flatness_map(self, object_region: Dict, surface_analysis: Dict) -> np.ndarray:
        """計算平坦度地圖"""
        mask = object_region['mask']
        normals = surface_analysis['normals']
        
        # 1. 法向量一致性
        kernel_size = 5
        normal_consistency = np.zeros_like(mask, dtype=float)
        
        for i in range(kernel_size//2, mask.shape[0] - kernel_size//2):
            for j in range(kernel_size//2, mask.shape[1] - kernel_size//2):
                if mask[i, j]:
                    # 取鄰域法向量
                    region_normals = normals[i-kernel_size//2:i+kernel_size//2+1, 
                                           j-kernel_size//2:j+kernel_size//2+1]
                    region_mask = mask[i-kernel_size//2:i+kernel_size//2+1, 
                                     j-kernel_size//2:j+kernel_size//2+1]
                    
                    if np.sum(region_mask) > 0:
                        center_normal = normals[i, j]
                        valid_normals = region_normals[region_mask > 0]
                        
                        if len(valid_normals) > 1:
                            # 計算法向量之間的角度一致性
                            dot_products = np.dot(valid_normals, center_normal)
                            normal_consistency[i, j] = np.mean(np.abs(dot_products))
        
        # 2. 深度變化平滑度
        depth_smoothness = self._compute_depth_smoothness(object_region['depth'], mask)
        
        # 3. 多尺度平坦度
        multi_scale_flatness = np.zeros_like(mask, dtype=float)
        for scale_key in surface_analysis['multi_scale']:
            if 'variance' in scale_key:
                variance_map = surface_analysis['multi_scale'][scale_key]
                multi_scale_flatness += np.exp(-variance_map)
        
        multi_scale_flatness /= len([k for k in surface_analysis['multi_scale'] if 'variance' in k])
        
        flatness = (
            0.5 * normal_consistency +
            0.3 * depth_smoothness +
            0.2 * multi_scale_flatness
        ) * mask
        
        return flatness
    
    def _compute_accessibility_map(self, object_region: Dict, mask: np.ndarray) -> np.ndarray:
        """計算可達性地圖"""
        # 1. 從多個角度的可達性
        accessibility = np.zeros_like(mask, dtype=float)
        
        # 2. 距離物件中心的可達性
        center_distance = self._compute_center_accessibility(mask)
        
        # 3. 遮擋分析
        occlusion_map = self._compute_occlusion_map(object_region)
        
        # 4. 幾何約束分析
        geometric_constraints = self._analyze_geometric_constraints(object_region)
        
        accessibility = (
            0.4 * center_distance +
            0.4 * (1 - occlusion_map) +  # 低遮擋 = 高可達性
            0.2 * geometric_constraints
        ) * mask
        
        return accessibility
    
    def _compute_visibility_map(self, object_region: Dict, image: np.ndarray, 
                               mask: np.ndarray) -> np.ndarray:
        """計算視覺顯著性地圖"""
        masked_image = object_region['image']
        
        # 1. 顏色對比度
        color_contrast = self._compute_color_contrast(masked_image, mask)
        
        # 2. 紋理複雜度
        texture_complexity = self._compute_texture_complexity(masked_image, mask)
        
        # 3. 邊緣密度
        edge_density = self._compute_edge_density(masked_image, mask)
        
        # 4. 視覺中心性
        visual_centrality = self._compute_visual_centrality(mask)
        
        visibility = (
            0.3 * color_contrast +
            0.3 * texture_complexity +
            0.2 * edge_density +
            0.2 * visual_centrality
        ) * mask
        
        return visibility
    
    def _compute_semantic_suitability(self, object_region: Dict, object_label: str) -> np.ndarray:
        """計算語義適合性地圖"""
        mask = object_region['mask']
        
        # 基於物件類型的適合性規則
        semantic_rules = {
            'table': self._table_semantic_analysis,
            'chair': self._chair_semantic_analysis,
            'desk': self._desk_semantic_analysis,
            'shelf': self._shelf_semantic_analysis,
            'floor': self._floor_semantic_analysis,
            'wall': self._wall_semantic_analysis
        }
        
        if object_label.lower() in semantic_rules:
            return semantic_rules[object_label.lower()](object_region)
        else:
            # 默認語義分析
            return self._default_semantic_analysis(object_region)
    
    def _fuse_criteria_maps(self, stability_map: np.ndarray, flatness_map: np.ndarray,
                           accessibility_map: np.ndarray, visibility_map: np.ndarray,
                           semantic_map: np.ndarray) -> np.ndarray:
        """多準則決策融合"""
        # 正規化每個地圖
        maps = [stability_map, flatness_map, accessibility_map, visibility_map, semantic_map]
        normalized_maps = []
        
        for map_data in maps:
            if map_data.max() > 0:
                normalized = (map_data - map_data.min()) / (map_data.max() - map_data.min())
            else:
                normalized = map_data
            normalized_maps.append(normalized)
        
        # 加權融合
        composite_score = (
            self.weights['stability'] * normalized_maps[0] +
            self.weights['flatness'] * normalized_maps[1] +
            self.weights['accessibility'] * normalized_maps[2] +
            self.weights['visibility'] * normalized_maps[3] 
            # self.weights['semantic'] * normalized_maps[4]
        )
        
        return composite_score
    
    def _generate_candidate_points(self, score_map: np.ndarray, object_region: Dict, 
                                top_k: int) -> List[Dict]:
        """生成候選點"""
        mask = object_region['mask']  # ✅ 取得 mask
        
        # ✅ 找到局部最大值：限制只在 mask 裡面
        local_maxima = peak_local_max(
            score_map,
            min_distance=10,
            threshold_abs=0.1,
            labels=mask.astype(np.uint8)  # ✅ 加這行
        )
        
        candidates = []
        h, w = score_map.shape
        for i, (y, x) in enumerate(local_maxima):
            if 0 <= y < h and 0 <= x < w:
                score = score_map[y, x]
                depth = object_region['depth'][y, x]
                candidates.append({
                    'position': (x, y),
                    'score': score,
                    'depth': depth,
                    'index': i
                })

        print("✅ DEBUG: Local maxima count:", len(local_maxima))
        
        # 按分數排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k * 2]

    
    def _validate_and_refine_points(self, candidates: List[Dict], object_region: Dict,
                                   mask: np.ndarray) -> List[Dict]:
        """驗證和精煉候選點"""
        validated = []
        print(f"📌 [validate] Total candidates: {len(candidates)}")

        filtered = 0
        for candidate in candidates:
            x, y = candidate['position']
            if y >= mask.shape[0] or x >= mask.shape[1]:
                print(f"⚠️ Skip: ({x}, {y}) 超出範圍")
                continue
            if mask[y, x] == 0:
                filtered += 1
                continue

        print(f"🚫 被 mask 過濾掉的點數量：{filtered}")

        for candidate in candidates:
            x, y = candidate['position']
            
            # # 驗證點是否在有效區域內
            # if mask[y, x] == 0:
            #     continue
                
            # 局部優化
            refined_pos = self._local_optimization(x, y, object_region)
            
            # 計算置信度
            confidence = self._compute_placement_confidence(refined_pos, object_region)
            
            # 生成詳細分析
            analysis = self._generate_placement_analysis(refined_pos, object_region)
            
            validated.append({
                'position': refined_pos,
                'score': candidate['score'],
                'confidence': confidence,
                'analysis': analysis,
                'original_candidate': candidate
            })
        
        return validated
    
    # 輔助函數實現
    def _compute_local_support_area(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """計算局部支撐面積"""
        kernel = disk(5)
        support_area = ndimage.convolve(mask.astype(float), kernel) * mask
        return support_area / support_area.max() if support_area.max() > 0 else support_area
    
    def _compute_edge_distance(self, mask: np.ndarray) -> np.ndarray:
        """計算到邊緣的距離"""
        return ndimage.distance_transform_edt(mask) / np.max(ndimage.distance_transform_edt(mask))
    
    def _compute_depth_smoothness(self, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """計算深度平滑度"""
        laplacian = cv2.Laplacian(depth.astype(np.float64), cv2.CV_64F)
        smoothness = np.exp(-np.abs(laplacian)) * mask
        return smoothness
    
    def _compute_center_accessibility(self, mask: np.ndarray) -> np.ndarray:
        """計算到中心的可達性"""
        center_y, center_x = ndimage.center_of_mass(mask)
        y_coords, x_coords = np.ogrid[:mask.shape[0], :mask.shape[1]]
        distance_to_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.max(distance_to_center[mask > 0])
        return (1 - distance_to_center / max_distance) * mask if max_distance > 0 else mask.astype(float)
    
    def _compute_occlusion_map(self, object_region: Dict) -> np.ndarray:
        """計算遮擋地圖"""
        # 簡化實現：基於深度梯度
        depth = object_region['depth']
        mask = object_region['mask']
        
        grad_magnitude = np.sqrt(
            cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)**2 +
            cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)**2
        )
        
        occlusion = grad_magnitude / (grad_magnitude.max() + 1e-8) * mask
        return occlusion
    
    def _analyze_geometric_constraints(self, object_region: Dict) -> np.ndarray:
        """分析幾何約束"""
        # 基於曲率和法向量的幾何分析
        mask = object_region['mask']
        return np.ones_like(mask, dtype=float) * mask  # 簡化實現
    
    def _compute_color_contrast(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """計算顏色對比度"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # 使用標準差作為對比度指標
        kernel = np.ones((5, 5), np.float32) / 25
        contrast = cv2.filter2D(gray.astype(float), -1, kernel)
        contrast = np.abs(gray - contrast) * mask
        
        return contrast / (contrast.max() + 1e-8)
    
    def _compute_texture_complexity(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """計算紋理複雜度"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        # 使用Laplacian算子檢測紋理
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture = np.abs(laplacian) * mask
        
        return texture / (texture.max() + 1e-8)
    
    def _compute_edge_density(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """計算邊緣密度"""
        if image.ndim == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
        
        edges = cv2.Canny(gray, 50, 150)
        
        # 計算局部邊緣密度
        kernel = np.ones((7, 7), np.float32)
        edge_density = cv2.filter2D(edges.astype(float), -1, kernel) * mask
        
        return edge_density / (edge_density.max() + 1e-8)
    
    def _compute_visual_centrality(self, mask: np.ndarray) -> np.ndarray:
        """計算視覺中心性"""
        return self._compute_center_accessibility(mask)
    
    # 語義分析函數
    def _table_semantic_analysis(self, object_region: Dict) -> np.ndarray:
        """桌子的語義分析"""
        mask = object_region['mask']
        depth = object_region['depth']
        
        # 桌子優先選擇平坦的桌面區域
        # 通常是深度值較小（離相機較近）的區域
        if np.sum(mask) > 0:
            depth_masked = depth[mask > 0]
            top_surface_threshold = np.percentile(depth_masked, 20)  # 前20%的深度
            top_surface_mask = (depth <= top_surface_threshold) * mask
        else:
            top_surface_mask = mask
            
        return top_surface_mask.astype(float)
    
    def _chair_semantic_analysis(self, object_region: Dict) -> np.ndarray:
        """椅子的語義分析"""
        mask = object_region['mask']
        # 椅子優先選擇座面區域
        return mask.astype(float)
    
    def _desk_semantic_analysis(self, object_region: Dict) -> np.ndarray:
        """桌子的語義分析（同table）"""
        return self._table_semantic_analysis(object_region)
    
    def _shelf_semantic_analysis(self, object_region: Dict) -> np.ndarray:
        """架子的語義分析"""
        mask = object_region['mask']
        # 架子優先選擇水平表面
        return mask.astype(float)
    
    def _floor_semantic_analysis(self, object_region: Dict) -> np.ndarray:
        """地板的語義分析"""
        mask = object_region['mask']
        # 地板選擇相對平坦的區域
        return mask.astype(float)
    
    def _wall_semantic_analysis(self, object_region: Dict) -> np.ndarray:
        """牆面的語義分析"""
        mask = object_region['mask']
        # 牆面分析較複雜，這裡簡化
        return mask.astype(float)
    
    def _default_semantic_analysis(self, object_region: Dict) -> np.ndarray:
        """默認語義分析"""
        mask = object_region['mask']
        return mask.astype(float)
    
    def _local_optimization(self, x: int, y: int, object_region: Dict) -> Tuple[int, int]:
        """局部優化候選點位置"""
        # 在小鄰域內尋找更好的位置
        mask = object_region['mask']
        depth = object_region['depth']
        
        best_x, best_y = x, y
        best_score = 0
        
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < mask.shape[1] and 0 <= new_y < mask.shape[0] and
                    mask[new_y, new_x] > 0):
                    
                    # 簡單的局部評分（可以擴展）
                    local_score = mask[new_y, new_x]
                    if local_score > best_score:
                        best_score = local_score
                        best_x, best_y = new_x, new_y
        
        return best_x, best_y
    
    def _compute_placement_confidence(self, position: Tuple[int, int], 
                                    object_region: Dict) -> float:
        """計算置入點的置信度"""
        x, y = position
        mask = object_region['mask']
        
        if mask[y, x] == 0:
            return 0.0
        
        # 基於鄰域的置信度計算
        neighborhood = mask[max(0, y-5):y+6, max(0, x-5):x+6]
        confidence = np.mean(neighborhood)
        
        return float(confidence)
    
    def _generate_placement_analysis(self, position: Tuple[int, int], 
                                   object_region: Dict) -> Dict:
        """生成置入點的詳細分析"""
        x, y = position
        
        analysis = {
            'position_3d': (x, y, object_region['depth'][y, x]),
            'local_properties': {
                'depth': float(object_region['depth'][y, x]),
                'is_valid': object_region['mask'][y, x] > 0
            },
            'geometric_analysis': {
                'surface_type': 'unknown',  # 可以擴展
                'orientation': 'horizontal'  # 可以基於法向量計算
            },
            'recommendations': {
                'object_types': ['small_objects', 'decorative_items'],
                'placement_strategy': 'stable_horizontal'
            }
        }
        
        return analysis
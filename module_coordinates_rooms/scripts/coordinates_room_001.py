import cv2
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import distance_transform_edt

# Optional skeletonize
try:
    from skimage.morphology import skeletonize
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False
    print("⚠️ skimage not found — using fallback thinning")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_exterior_polygon_from_json(json_path: str) -> Optional[List[Tuple[int, int]]]:
    """Load exterior polygon from JSON file"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if 'polygon' in data and isinstance(data['polygon'], list):
            return [(int(p[0]), int(p[1])) for p in data['polygon']]
        for key in ('objects', 'shapes', 'items'):
            if key in data and isinstance(data[key], list):
                for obj in data[key]:
                    label = obj.get('label', '') or obj.get('type', '')
                    if isinstance(label, str) and 'exterior' in label.lower():
                        poly = obj.get('polygon') or obj.get('points') or obj.get('vertices') or obj.get('coords')
                        if isinstance(poly, list):
                            return [(int(p[0]), int(p[1])) for p in poly]
    
    if isinstance(data, list) and len(data) > 2 and isinstance(data[0], (list, tuple)):
        return [(int(p[0]), int(p[1])) for p in data]
    
    return None


def polygon_area_signed(poly: List[Tuple[int, int]]) -> float:
    """Calculate signed area of polygon"""
    a = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        a += x1*y2 - x2*y1
    return a/2


def snap_polygon_points_to_exterior(approx, segments, snap_thresh=10):
    """Snap polygon points to nearest exterior segment"""
    snapped = []
    for pt in approx:
        x, y = pt[0]
        best_dist = float('inf')
        best_pt = (x, y)
        for x1, y1, x2, y2 in segments:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                dist = np.hypot(x - x1, y - y1)
                proj_x, proj_y = x1, y1
            else:
                t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)))
                proj_x = x1 + t * dx
                proj_y = y1 + t * dy
                dist = np.hypot(x - proj_x, y - proj_y)
            if dist < best_dist:
                best_dist = dist
                best_pt = (int(proj_x), int(proj_y))
        if best_dist <= snap_thresh:
            snapped.append([best_pt])
        else:
            snapped.append([(x, y)])
    return np.array(snapped, dtype=np.int32)


# ============================================================================
# FLOORPLAN MASK GENERATOR
# ============================================================================

class FloorPlanMaskGenerator:
    """Generate room masks from floorplan image"""
    
    def __init__(self, input_folder: str, output_folder: str, output_img_folder: str, output_json_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.output_img_folder = output_img_folder
        self.output_json_folder = output_json_folder
        os.makedirs(output_img_folder, exist_ok=True)
        os.makedirs(output_json_folder, exist_ok=True)
        
        # Input files
        self.input_image_path = os.path.join(input_folder, "drawing_overwrite.png")
        self.exterior_img_path = os.path.join(input_folder, "exterior_wall.png")
        self.exterior_json_path = os.path.join(input_folder, "exterior_wall_shrink.json")
        self.rooms_json_path = os.path.join(input_folder, "rooms_graph.json")
        
        # Color mapping
        self.color_mapping = {
            'living_room': (200, 230, 240),
            'master_room': (0, 165, 255),
            'kitchen': (128, 128, 240),
            'bathroom': (210, 216, 173),
            'balcony': (35, 142, 107),
            'dining_room': (214, 112, 218),
            'storage': (221, 160, 221),
            'common_room': (0, 215, 255),
            'interior_wall': (255, 255, 255)
        }
        
        # State variables
        self.image = None
        self.h = None
        self.w = None
        self.exterior_mask = None
        self.exterior_polygon = None
        self.exterior_segments = []
        self.exterior_edge_zone = None
        self.all_room_masks = {}
        self.final_room_masks = {}
        self.snapped_polygons = {}
        self.interior_wall_mask = None
        self.spacing = 5
        self.json_nodes = []
    
    def load_assets(self):
        """Load input images and exterior polygon"""
        print("📥 Step 1: Loading assets...")
        self.image = cv2.imread(self.input_image_path)
        if self.image is None:
            raise FileNotFoundError(f"Cannot load {self.input_image_path}")
        self.h, self.w = self.image.shape[:2]
        print(f"   Image size: {self.w}x{self.h}")
        
        # Load exterior polygon
        poly = None
        if os.path.exists(self.exterior_json_path):
            try:
                poly = load_exterior_polygon_from_json(self.exterior_json_path)
                if poly:
                    if polygon_area_signed(poly) > 0:
                        poly = poly[::-1]
                    self.exterior_polygon = poly
            except Exception as e:
                print(f"   Warning reading exterior JSON: {e}")
        
        # Fallback to exterior image
        if self.exterior_polygon is None:
            if os.path.exists(self.exterior_img_path):
                ext = cv2.imread(self.exterior_img_path, cv2.IMREAD_GRAYSCALE)
                if ext is None:
                    raise FileNotFoundError(f"Cannot read {self.exterior_img_path}")
                _, extb = cv2.threshold(ext, 127, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(extb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    raise RuntimeError("No contour found in exterior_wall.png")
                c = max(cnts, key=cv2.contourArea)
                poly = c.reshape(-1, 2).tolist()
                self.exterior_polygon = [(int(x), int(y)) for x, y in poly]
                self.exterior_mask = extb
            else:
                raise FileNotFoundError("No exterior polygon JSON and no exterior_wall.png found.")
        else:
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            pts = np.array(self.exterior_polygon, np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
            self.exterior_mask = mask
        
        # Build exterior segments
        segs = []
        pts = self.exterior_polygon
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i+1) % len(pts)]
            segs.append((a[0], a[1], b[0], b[1]))
        self.exterior_segments = segs
        
        # Create edge zone
        edges = cv2.Canny(self.exterior_mask, 50, 150)
        self.exterior_edge_zone = cv2.dilate(
            edges, 
            cv2.getStructuringElement(cv2.MORPH_RECT, (self.spacing, self.spacing)), 
            iterations=1
        )
        # Don't save exterior_mask.png - not in required outputs
    
    def _mask_by_color(self, target_bgr, tol=30):
        """Create mask by color matching"""
        img = self.image.astype(np.int16)
        c = np.array(target_bgr, dtype=np.int16).reshape(1, 1, 3)
        dist = np.linalg.norm(img - c, axis=2)
        return (dist <= tol).astype(np.uint8) * 255
    
    def _clean_mask(self, mask, min_area=50):  # Giảm ngưỡng diện tích
        """Clean mask using morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # Giảm số lần lặp để tránh mất chi tiết
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                out[labels == i] = 255
        return out
    
    def _snap_and_orthogonalize(self, mask, snap_thresh=10):
        """Snap polygon to exterior walls"""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask, None
            
        final_mask = np.zeros_like(mask)
        all_poly_pts = []
        
        for cnt in cnts:
            # Xử lý từng contour riêng lẻ
            area = cv2.contourArea(cnt)
            if area < 100:  # Bỏ qua các contour quá nhỏ
                continue
                
            eps = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if approx.shape[0] < 3:
                continue
                
            snapped = snap_polygon_points_to_exterior(approx, self.exterior_segments, snap_thresh=snap_thresh)
            poly_pts = snapped.reshape(-1, 2).tolist()
            
            # Kiểm tra và điều chỉnh hướng polygon nếu cần
            if polygon_area_signed(poly_pts) > 0:
                snapped = snapped[::-1]
                
            cv2.fillPoly(final_mask, [snapped], 255)
            all_poly_pts.extend(poly_pts)
            
        return final_mask, all_poly_pts
    
    def generate_room_masks(self, tolerance_map=None):
        """Generate masks for all rooms"""
        print("🎨 Step 2: Generating room masks...")
        if tolerance_map is None:
            tolerance_map = {}
        
        cleaned_masks = {}
        snapped_polygons = {}
        
        for room, color in self.color_mapping.items():
            if room == 'interior_wall':
                continue
                
            # 1. Tạo mask ban đầu cho màu
            tol = tolerance_map.get(room, 35)
            room_mask = self._mask_by_color(color, tol=tol)
            
            # 2. Tìm connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_mask, connectivity=8)
            
            # Khởi tạo mask tổng cho loại phòng này
            final_room_mask = np.zeros_like(room_mask)
            room_polygons = []
            
            # 3. Xử lý từng component riêng biệt
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area < 200:  # Ngưỡng diện tích tối thiểu
                    continue
                
                # Tạo mask cho component hiện tại
                component_mask = (labels == label).astype(np.uint8) * 255
                
                # Clean và snap component
                cleaned_component = self._clean_mask(component_mask, min_area=200)
                cleaned_component = cv2.bitwise_and(cleaned_component, self.exterior_mask)
                
                # Snap vào tường và lấy polygon
                snapped_mask, poly_pts = self._snap_and_orthogonalize(cleaned_component, snap_thresh=10)
                if poly_pts:
                    snapped_mask = cv2.bitwise_and(snapped_mask, self.exterior_mask)
                    final_cleaned = self._clean_mask(snapped_mask, min_area=150)
                    
                    # Thêm vào mask tổng
                    final_room_mask = cv2.bitwise_or(final_room_mask, final_cleaned)
                    room_polygons.extend(poly_pts)
            
            # Lưu kết quả cho loại phòng này
            cleaned_masks[room] = final_room_mask
            snapped_polygons[room] = room_polygons
        
        self.cleaned_masks = cleaned_masks
        self.snapped_polygons = snapped_polygons
        
        # Preserve edge contact
        preserve_masks = {}
        for room, cm in cleaned_masks.items():
            contact = cv2.bitwise_and(cm, self.exterior_edge_zone)
            contact = cv2.dilate(contact, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
            preserve_masks[room] = contact
        
        # Apply spacing
        final_masks = {}
        spacing_px = self.spacing
        erode_iters = max(1, spacing_px // 2)
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        for room, cm in cleaned_masks.items():
            eroded = cv2.erode(cm, erode_kernel, iterations=erode_iters)
            restored = cv2.bitwise_or(eroded, preserve_masks[room])
            final = self._clean_mask(restored, min_area=150)
            final_masks[room] = final
        
        self.all_room_masks = final_masks
    
    def build_interior_wall(self):
        """Build interior wall mask"""
        print("🧱 Step 3: Building interior walls...")
        h, w = self.exterior_mask.shape
        union_rooms = np.zeros((h, w), dtype=np.uint8)
        for rm in self.all_room_masks.values():
            union_rooms = cv2.bitwise_or(union_rooms, rm)
        interior_area = cv2.subtract(self.exterior_mask, union_rooms)
        interior_area = self._clean_mask(interior_area, min_area=10)
        
        if SKIMAGE_AVAILABLE:
            bin_img = (interior_area > 0).astype(np.uint8)
            sk = skeletonize(bin_img).astype(np.uint8) * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.spacing, self.spacing))
            interior_wall = cv2.dilate(sk, kernel, iterations=1)
        else:
            dt = cv2.distanceTransform(interior_area, cv2.DIST_L2, 5)
            if dt.max() <= 0:
                interior_wall = np.zeros_like(interior_area)
            else:
                thresh = max(1.0, 0.5 * dt.max())
                center = (dt >= thresh).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.spacing, self.spacing))
                interior_wall = cv2.dilate(center, kernel, iterations=1)
        
        interior_wall = cv2.bitwise_and(interior_wall, interior_area)
        self.interior_wall_mask = interior_wall
        cv2.imwrite(os.path.join(self.output_img_folder, "interior_wall_mask.png"), interior_wall)
    
    def enrich_and_map_json_nodes(self):
        """Map JSON nodes to room centroids"""
        mapping = {}
        json_nodes = []
        
        if os.path.exists(self.rooms_json_path):
            try:
                with open(self.rooms_json_path, 'r', encoding='utf-8') as f:
                    jd = json.load(f)
                if isinstance(jd, dict) and 'rooms' in jd:
                    for r in jd['rooms']:
                        if 'nodes' in r and isinstance(r['nodes'], list):
                            for p in r['nodes']:
                                json_nodes.append((int(p[0]), int(p[1])))
                elif isinstance(jd, list):
                    for p in jd:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            json_nodes.append((int(p[0]), int(p[1])))
            except Exception as e:
                print(f'   Warning: cannot parse rooms JSON: {e}')
        
        self.json_nodes = json_nodes
        
        # Calculate room centroids
        room_centroids = {}
        for name, mask in self.all_room_masks.items():
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                ys, xs = np.where(mask > 0)
                if len(xs) == 0:
                    cx, cy = None, None
                else:
                    cx = int((xs.min() + xs.max()) / 2)
                    cy = int((ys.min() + ys.max()) / 2)
            room_centroids[name] = (cx, cy)
        
        # Map nodes to rooms
        used_idx = set()
        for name, (cx, cy) in room_centroids.items():
            if cx is None:
                mapping[name] = {'centroid': None, 'json_point': None, 'color': self.color_mapping.get(name)}
                continue
            best_d = float('inf')
            best_pt = None
            best_i = None
            for i, p in enumerate(json_nodes):
                d = (p[0] - cx) ** 2 + (p[1] - cy) ** 2
                if d < best_d and i not in used_idx:
                    best_d = d
                    best_pt = p
                    best_i = i
            if best_pt is not None:
                used_idx.add(best_i)
            mapping[name] = {'centroid': (cx, cy), 'json_point': best_pt, 'color': self.color_mapping.get(name)}
        
        return mapping
    
    def watershed_fill_by_centroids(self, mapping):
        """Fill rooms using watershed algorithm"""
        print("💧 Step 4: Watershed fill by centroids...")
        union_rooms = np.zeros((self.h, self.w), dtype=np.uint8)
        for m in self.all_room_masks.values():
            union_rooms = cv2.bitwise_or(union_rooms, m)
        
        edges = cv2.Canny(self.exterior_mask, 50, 150)
        sk = self.interior_wall_mask if self.interior_wall_mask is not None else np.zeros_like(union_rooms)
        skeleton_with_edges = cv2.bitwise_or(sk, edges)
        
        markers = np.zeros((self.h, self.w), dtype=np.int32)
        room_names = list(self.all_room_masks.keys())
        next_label = 1
        
        for name in room_names:
            mask = self.all_room_masks[name]
            # Tìm tất cả các connected components trong mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            
            for label in range(1, num_labels):  # Bỏ qua background (label 0)
                # Lấy centroid cho component này
                cx = int(centroids[label][0])
                cy = int(centroids[label][1])
                
                # Kiểm tra diện tích tối thiểu
                if stats[label, cv2.CC_STAT_AREA] < 50:
                    continue
                    
                # Tạo mask cho component hiện tại
                current_mask = (labels == label).astype(np.uint8) * 255
                
                # Tìm điểm seed phù hợp
                y_coords, x_coords = np.where(current_mask > 0)
                if len(x_coords) == 0:
                    continue
                    
                seed_x = int(np.mean(x_coords))
                seed_y = int(np.mean(y_coords))
                
                # Đặt marker cho component này
                cv2.circle(markers, (seed_x, seed_y), 2, next_label, -1)
                next_label += 1
        
        img_watershed = self.image.copy()
        img_watershed[skeleton_with_edges > 0] = (0, 0, 0)
        bg_mask = (union_rooms == 0)
        img_watershed[bg_mask] = (0, 0, 0)
        
        markers_copy = markers.copy()
        cv2.watershed(img_watershed, markers_copy)
        
        final_masks = {}
        for name in room_names:
            # Tìm tất cả các label liên quan đến phòng này
            mask = np.zeros_like(markers_copy, dtype=np.uint8)
            
            # Lấy các connected components từ mask gốc
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(self.all_room_masks[name])
            
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] < 50:
                    continue
                    
                component_mask = (labels == label).astype(np.uint8)
                # Tìm các marker trong vùng này
                unique_markers = np.unique(markers_copy * component_mask)
                unique_markers = unique_markers[unique_markers != 0]  # Bỏ qua background
                
                # Thêm tất cả các vùng được đánh dấu vào mask cuối cùng
                for marker in unique_markers:
                    mask = cv2.bitwise_or(mask, (markers_copy == marker).astype(np.uint8))
            
            # Chuyển về định dạng 255 và clean
            mask = mask * 255
            mask = cv2.bitwise_and(mask, self.all_room_masks[name])
            if mask.sum() == 0:
                mask = self.all_room_masks[name].copy()
            
            final_masks[name] = self._clean_mask(mask, min_area=50)
        
        self.final_room_masks = final_masks
        return final_masks
    
    def save_overlay_watershed(self):
        """Save watershed overlay image"""
        overlay = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for name, mask in self.final_room_masks.items():
            color = self.color_mapping.get(name, (200, 200, 200))
            overlay[mask > 0] = color
        if self.interior_wall_mask is not None:
            overlay[self.interior_wall_mask > 0] = self.color_mapping.get('interior_wall', (255, 255, 255))
        edges = cv2.Canny(self.exterior_mask, 50, 150)
        overlay[edges > 0] = (0, 0, 0)
        # Don't save overlay_filled_by_watershed.png - not in required outputs
        return overlay


# ============================================================================
# DISTANCE FILL
# ============================================================================

def distance_constrained_fill(output_img_folder: str, overlay_image, distance_threshold=1.5):
    """Apply distance-constrained fill to remove thin wall gaps"""
    print("\n📏 Step 5: Distance-constrained fill...")
    wall_path = os.path.join(output_img_folder, "interior_wall_mask.png")
    output_path = os.path.join(output_img_folder, "layout_final_distancefill.png")
    
    walls = cv2.imread(wall_path, cv2.IMREAD_GRAYSCALE)
    overlay = overlay_image
    
    if walls is None or overlay is None:
        raise FileNotFoundError("Missing wall or overlay image")
    
    h, w = walls.shape[:2]
    wall_mask = (walls > 200).astype(np.uint8)
    free_space = (wall_mask == 0).astype(np.uint8)
    
    dist = distance_transform_edt(free_space)
    
    gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    color_mask = (gray > 10).astype(np.uint8)
    overlay_copy = overlay.copy()
    overlay_copy[color_mask == 0] = 0
    
    color_ids = (overlay_copy[:, :, 0].astype(np.int32)
                 + overlay_copy[:, :, 1].astype(np.int32) * 256
                 + overlay_copy[:, :, 2].astype(np.int32) * 256 * 256)
    unique_ids = np.unique(color_ids[color_ids > 0])
    
    label_map = np.zeros((h, w), np.int32)
    for i, uid in enumerate(unique_ids, 1):
        label_map[color_ids == uid] = i
    
    filled = np.zeros_like(label_map)
    
    for i, uid in enumerate(unique_ids, 1):
        mask_room = (label_map == i)
        if np.sum(mask_room) == 0:
            continue
        
        dist_room = dist * mask_room
        max_idx = np.unravel_index(np.argmax(dist_room), dist_room.shape)
        seed = (int(max_idx[0]), int(max_idx[1]))
        
        stack = [seed]
        visited = np.zeros((h, w), np.uint8)
        visited[seed] = 1
        
        while stack:
            y, x = stack.pop()
            if dist[y, x] < distance_threshold:
                continue
            filled[y, x] = i
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if visited[ny, nx] == 0 and free_space[ny, nx] == 1 and filled[ny, nx] == 0:
                        visited[ny, nx] = 1
                        stack.append((ny, nx))
    
    final = np.zeros_like(overlay)
    for i, uid in enumerate(unique_ids, 1):
        mask = (filled == i)
        if np.any(mask):
            color = overlay[label_map == i][0].tolist()
            final[mask] = color
    
    final[wall_mask > 0] = (255, 255, 255)
    cv2.imwrite(output_path, final)
    print(f"   Saved: {output_path}")
    return output_path


# ============================================================================
# REMOVE BACKGROUND & CROP
# ============================================================================

def remove_background_and_crop(output_img_folder: str, padding=20):
    """Remove background and crop to content"""
    print("\n✂️ Step 6: Remove background & crop...")
    input_path = os.path.join(output_img_folder, "layout_final_distancefill.png")
    output_path = os.path.join(output_img_folder, "layout_final_clean.png")
    
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load {input_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    wall_mask = (gray > 240).astype(np.uint8)
    protected = cv2.dilate(wall_mask, np.ones((5, 5), np.uint8), iterations=2)
    
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    flood_img = img.copy()
    cv2.floodFill(flood_img, ff_mask, (0, 0), (255, 255, 255),
                  loDiff=(3, 3, 3), upDiff=(3, 3, 3))
    
    gray_ff = cv2.cvtColor(flood_img, cv2.COLOR_BGR2GRAY)
    inside_mask = (gray_ff < 250).astype(np.uint8)
    
    result = cv2.bitwise_and(img, img, mask=inside_mask)
    
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    
    gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray_res)
    x, y, w_box, h_box = cv2.boundingRect(coords)
    
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w_box + padding, w)
    y2 = min(y + h_box + padding, h)
    
    cropped = result[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped)
    print(f"   Saved: {output_path}")
    
    return output_path


# ============================================================================
# ROOM COORDINATES EXTRACTOR
# ============================================================================

class RoomCoordinatesExtractor:
    """Extract room coordinates and generate labeled output"""
    
    def __init__(self, output_img_folder: str, output_json_folder: str):
        self.output_img_folder = output_img_folder
        self.output_json_folder = output_json_folder
        self.input_path = os.path.join(output_img_folder, "layout_final_clean.png")
        self.output_img_path = os.path.join(output_img_folder, "floorplan_labeled.png")
        self.output_json_path = os.path.join(output_json_folder, "room_coordinates.json")
        
        self.categories = self._define_categories()
        self.color_map = self._define_color_map()
        self.cat_name_to_id = {c["name"]: c["id"] for c in self.categories}
    
    def _define_categories(self) -> List[Dict]:
        """Define room categories"""
        return [
            {"id": 1, "name": "Living"},
            {"id": 2, "name": "Kitchen"},
            {"id": 3, "name": "Storage"},
            {"id": 4, "name": "Common"},
            {"id": 5, "name": "Dining"},
            {"id": 6, "name": "Master"},
            {"id": 7, "name": "Balcony"},
            {"id": 8, "name": "Bath"},
            {"id": 9, "name": "Frontdoor"},
            {"id": 10, "name": "Exterior wall"},
            {"id": 11, "name": "Interior wall"},
        ]
    
    def _define_color_map(self) -> Dict[str, list]:
        """Define color mapping for room detection"""
        return {
            "Living": [200, 230, 240],  # BGR format - màu vàng nhạt/kem
            "Master": [0, 165, 255],
            "Kitchen": [128, 128, 240],
            "Bath": [210, 216, 173],
            "Balcony": [35, 142, 107],
            "Dining": [214, 112, 218],
            "Storage": [221, 160, 221],
            "Common": [0, 215, 255],
            "Exterior wall": [0, 0, 0],
            "Frontdoor": [25, 225, 255],
            "Interior wall": [255, 255, 255],
        }
    
    def _init_coco(self, img: np.ndarray) -> Dict:
        """Initialize COCO format output"""
        h, w = img.shape[:2]
        return {
            "info": {"description": "floorplan-rooms"},
            "images": [{"id": 1, "width": w, "height": h, "file_name": os.path.basename(self.input_path)}],
            "annotations": [],
            "categories": self.categories,
        }
    
    def _process_rooms(self, img: np.ndarray, annotated: np.ndarray, coco: Dict) -> Dict:
        """Process rooms and extract coordinates"""
        ann_id = 0
        for room, bgr in self.color_map.items():
            if room in ["Interior wall", "Exterior wall"]:
                continue
            
            # Create mask for room color with tolerance
            lower = np.clip(np.array(bgr) - 25, 0, 255)
            upper = np.clip(np.array(bgr) + 25, 0, 255)
            room_mask = cv2.inRange(img, lower, upper)
            
            # Tìm các connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_mask, connectivity=8)
            
            for label in range(1, num_labels):  # Bỏ qua background (label = 0)
                # Kiểm tra diện tích tối thiểu
                area = stats[label, cv2.CC_STAT_AREA]
                if area < 100:  # Ngưỡng diện tích tối thiểu cho mỗi phòng
                    continue
                
                # Tạo mask cho component hiện tại
                component_mask = (labels == label).astype(np.uint8) * 255
                
                # Làm mịn mask
                kernel = np.ones((3, 3), np.uint8)
                component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
                
                # Tìm contour cho component này
                contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                    
                cnt = max(contours, key=cv2.contourArea)
                
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                area = float(cv2.contourArea(cnt))
                
                # All rooms use polygon (no bbox)
                segmentation = approx.flatten().astype(float).tolist()
                x, y, w, h = cv2.boundingRect(approx)
                bbox = [float(x), float(y), float(w), float(h)]
                cv2.polylines(annotated, [approx], True, (0, 0, 255), 2)
                
                ann = {
                    "id": ann_id,
                    "iscrowd": 0,
                    "image_id": 1,
                    "category_id": self.cat_name_to_id[room],
                    "segmentation": [segmentation],
                    "bbox": bbox,
                    "area": area
                }
                coco["annotations"].append(ann)
                ann_id += 1
                
                cv2.putText(annotated, room, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return coco
    
    def run(self):
        """Run coordinate extraction"""
        print("\n📍 Step 7: Extracting room coordinates...")
        img = cv2.imread(self.input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read input image at {self.input_path}")
        
        annotated = img.copy()
        coco_output = self._init_coco(img)
        coco_output = self._process_rooms(img, annotated, coco_output)
        
        cv2.imwrite(self.output_img_path, annotated)
        with open(self.output_json_path, "w") as f:
            json.dump(coco_output, f, indent=4)
        
        print(f"   Saved labeled image: {self.output_img_path}")
        print(f"   Saved coordinates JSON: {self.output_json_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class CoordinatesRoomPipeline:
    """Complete pipeline from floorplan image to room coordinates"""
    
    def __init__(self, module_folder: str = None):
        if module_folder is None:
            # Auto-detect module folder (parent of scripts folder)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            module_folder = os.path.dirname(script_dir)
        
        self.module_folder = module_folder
        self.input_folder = os.path.join(module_folder, "input")
        self.output_folder = os.path.join(module_folder, "output")
        self.output_img_folder = os.path.join(module_folder, "output", "images")
        self.output_json_folder = os.path.join(module_folder, "output", "json")
        
        # Create directories
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_img_folder, exist_ok=True)
        os.makedirs(self.output_json_folder, exist_ok=True)
    
    def validate_inputs(self):
        """Validate that all required input files exist"""
        required_files = [
            "drawing_overwrite.png",
            "exterior_wall_shrink.json"
        ]
        
        optional_files = [
            "exterior_wall.png",
            "rooms_graph.json"
        ]
        
        print("🔍 Validating input files...")
        for file in required_files:
            file_path = os.path.join(self.input_folder, file)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Required file not found: {file_path}")
            print(f"   ✅ {file}")
        
        for file in optional_files:
            file_path = os.path.join(self.input_folder, file)
            if os.path.exists(file_path):
                print(f"   ✅ {file} (optional)")
            else:
                print(f"   ⚠️  {file} (optional, not found)")
    
    def run(self, distance_threshold=1.5, crop_padding=20):
        """Run complete pipeline"""
        print("=" * 70)
        print("🏠 COORDINATES ROOM PIPELINE - COMPLETE")
        print("=" * 70)
        print(f"📂 Module folder: {self.module_folder}")
        print(f"📥 Input folder: {self.input_folder}")
        print(f"📤 Output images folder: {self.output_img_folder}")
        print(f"📤 Output json folder: {self.output_json_folder}")
        print()
        
        # Validate inputs
        self.validate_inputs()
        print()
        
        # Step 1-4: Generate masks and watershed fill
        mask_gen = FloorPlanMaskGenerator(self.input_folder, self.output_folder, self.output_img_folder, self.output_json_folder)
        mask_gen.load_assets()
        mask_gen.generate_room_masks(tolerance_map=None)
        mask_gen.build_interior_wall()
        mapping = mask_gen.enrich_and_map_json_nodes()
        mask_gen.watershed_fill_by_centroids(mapping)
        overlay_image = mask_gen.save_overlay_watershed()
        
        # Step 5: Distance fill
        distance_constrained_fill(self.output_img_folder, overlay_image, distance_threshold=distance_threshold)
        
        # Step 6: Remove background & crop
        remove_background_and_crop(self.output_img_folder, padding=crop_padding)
        
        # Step 7: Extract room coordinates
        extractor = RoomCoordinatesExtractor(self.output_img_folder, self.output_json_folder)
        extractor.run()
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print(f"📂 Image outputs: {self.output_img_folder}")
        print("   • interior_wall_mask.png")
        print("   • layout_final_distancefill.png")
        print("   • layout_final_clean.png")
        print("   • floorplan_labeled.png")
        print(f"📂 JSON outputs: {self.output_json_folder}")
        print("   • room_coordinates.json")
        print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    # Auto-detect module folder (parent of scripts folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_folder = os.path.dirname(script_dir)
    
    # Create and run pipeline
    pipeline = CoordinatesRoomPipeline(module_folder=module_folder)
    pipeline.run(distance_threshold=1.5, crop_padding=20)


if __name__ == '__main__':
    main()

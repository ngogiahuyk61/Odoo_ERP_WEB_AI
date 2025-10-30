import cv2
import numpy as np
import os
import json
from shapely.geometry import Polygon
import math
# -------------------------------
# Cấu hình thư mục
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "module_floorplan_valid", "output")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSON_DIR = os.path.join(OUTPUT_DIR, "json")

INPUT_IMG_DIR = os.path.join(BASE_DIR, "..", "..", "module_floorplan_valid", "output", "images")
INPUT_JSON_DIR = os.path.join(BASE_DIR, "..", "..", "module_floorplan_valid", "output", "json")

TARGET_BASE_NAME = os.getenv("TARGET_BASE_NAME", "278")

INPUT_IMG_PATH = os.path.join(INPUT_IMG_DIR, f"drawing_with_exterior_{TARGET_BASE_NAME}.png")
INPUT_EXTERIOR_JSON = os.path.join(INPUT_JSON_DIR, f"exterior_wall_{TARGET_BASE_NAME}.json")
INPUT_SHRINK_JSON = os.path.join(INPUT_JSON_DIR, f"exterior_wall_shrink_{TARGET_BASE_NAME}.json")

if not os.path.exists(INPUT_IMG_PATH):
    raise FileNotFoundError(f"Error: Cannot find PNG image file: {INPUT_IMG_PATH}")
if not os.path.exists(INPUT_EXTERIOR_JSON):
    raise FileNotFoundError(f"Error: Cannot find exterior JSON file: {INPUT_EXTERIOR_JSON}")
if not os.path.exists(INPUT_SHRINK_JSON):
    raise FileNotFoundError(f"Error: Cannot find shrink JSON file: {INPUT_SHRINK_JSON}")

print(f"Input image: {INPUT_IMG_PATH}")
print(f"Input exterior JSON: {INPUT_EXTERIOR_JSON}")
print(f"Input shrink JSON: {INPUT_SHRINK_JSON}")

output_img_path = os.path.join(OUTPUT_IMG_DIR, f"drawing_with_black_walls_{TARGET_BASE_NAME}.png")
output_json_path = os.path.join(OUTPUT_JSON_DIR, f"exterior_wall_shrink_{TARGET_BASE_NAME}.json")
# -------------------------------
img = cv2.imread(INPUT_IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Cannot read image at: {INPUT_IMG_PATH}")

lower_black = np.array([0, 0, 0])
upper_black = np.array([50, 50, 50])
mask_black = cv2.inRange(img, lower_black, upper_black)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_black = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel, iterations=1)

img_clean = img.copy()
img_clean[mask_black > 0] = (255, 255, 255)

# -------------------------------
# 2. Load JSON exterior_wall polygon (gốc và đã thu nhỏ)
# -------------------------------
with open(INPUT_EXTERIOR_JSON, "r", encoding="utf-8") as f:
    data_exterior = json.load(f)

with open(INPUT_SHRINK_JSON, "r", encoding="utf-8") as f:
    data_shrink = json.load(f)

exterior_coords = data_exterior.get("exterior_wall", [])
shrink_coords = data_shrink.get("exterior_wall_shrink", [])

if not exterior_coords:
    raise ValueError("Error: Cannot find 'exterior_wall' data in exterior JSON!")
if not shrink_coords:
    raise ValueError("Error: Cannot find 'exterior_wall_shrink' data in shrink JSON!")

# -------------------------------
# 3. Hàm chỉnh góc vuông 90°
# -------------------------------
def snap_to_90_degrees(coords, angle_threshold=10):
    """
    Chỉnh các góc của polygon về 90° (vuông góc)
    
    Args:
        coords: List tọa độ [[x1,y1], [x2,y2], ...]
        angle_threshold: Ngưỡng góc (độ) để coi là gần vuông góc
    
    Returns:
        List tọa độ đã được chỉnh về 90°
    """
    if len(coords) < 3:
        return coords
    
    # Loại bỏ điểm cuối nếu trùng điểm đầu
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    
    n = len(coords)
    new_coords = []
    
    for i in range(n):
        p_prev = coords[(i - 1) % n]
        p_curr = coords[i]
        p_next = coords[(i + 1) % n]
        
        # Vector từ prev -> curr và curr -> next
        v1 = np.array([p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]], dtype=float)
        v2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]], dtype=float)
        
        # Tính góc giữa 2 vector
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 == 0 or len_v2 == 0:
            new_coords.append(p_curr)
            continue
        
        cos_angle = np.dot(v1, v2) / (len_v1 * len_v2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_deg = abs(math.degrees(math.acos(cos_angle)))
        
        # Nếu góc gần 90° (±angle_threshold), chỉnh lại
        if abs(angle_deg - 90) <= angle_threshold:
            # Chọn hướng dominant (ngang hoặc dọc) cho mỗi cạnh
            if abs(v1[0]) > abs(v1[1]):  # Cạnh trước chủ yếu ngang
                # Giữ nguyên x, chỉnh y về cùng hàng với prev
                new_x = p_curr[0]
                new_y = p_prev[1]
            else:  # Cạnh trước chủ yếu dọc
                # Giữ nguyên y, chỉnh x về cùng cột với prev
                new_x = p_prev[0]
                new_y = p_curr[1]
            
            new_coords.append([int(new_x), int(new_y)])
        else:
            new_coords.append(p_curr)
    
    # Thêm điểm đầu vào cuối để đóng polygon
    new_coords.append(new_coords[0])
    
    return new_coords

def force_orthogonal_polygon(coords):
    """
    Ép polygon thành hình chữ nhật hoặc hình có các cạnh vuông góc 100%
    
    Thuật toán:
    1. Tìm các cạnh dominant (ngang/dọc)
    2. Chỉnh tất cả các điểm về grid vuông góc
    """
    if len(coords) < 3:
        return coords
    
    # Loại bỏ điểm cuối nếu trùng điểm đầu
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    
    # Phân loại các cạnh thành ngang hoặc dọc
    new_coords = []
    n = len(coords)
    
    for i in range(n):
        p_curr = coords[i]
        p_next = coords[(i + 1) % n]
        
        dx = p_next[0] - p_curr[0]
        dy = p_next[1] - p_curr[1]
        
        # Thêm điểm hiện tại
        new_coords.append(p_curr)
        
        # Nếu cạnh không ngang và không dọc, thêm điểm trung gian
        if abs(dx) > 5 and abs(dy) > 5:  # Cạnh xiên
            if abs(dx) > abs(dy):
                # Ưu tiên ngang -> thêm điểm trung gian dọc
                new_coords.append([p_next[0], p_curr[1]])
            else:
                # Ưu tiên dọc -> thêm điểm trung gian ngang
                new_coords.append([p_curr[0], p_next[1]])
    
    # Làm sạch các điểm trùng lặp
    cleaned_coords = []
    for i, p in enumerate(new_coords):
        if i == 0 or p != cleaned_coords[-1]:
            cleaned_coords.append(p)
    
    # Đóng polygon
    cleaned_coords.append(cleaned_coords[0])
    
    return cleaned_coords

# -------------------------------
# 4. Sử dụng tọa độ đã thu nhỏ từ file shrink
# -------------------------------
shrunk_coords_all = shrink_coords  # Đã có sẵn từ file shrink

# Kiểm tra tính hợp lệ của các polygon
valid_shrunk_coords = []
for coords in shrunk_coords_all:
    try:
        poly = Polygon(coords)
        if poly.is_valid and not poly.is_empty:
            valid_shrunk_coords.append(coords)
    except:
        continue

shrunk_coords_all = valid_shrunk_coords

# -------------------------------
# 5. Ẩn vùng NGOÀI polygon thu nhỏ, giữ nguyên vùng TRONG
# -------------------------------
img_final = img_clean.copy()  # Bắt đầu từ ảnh đã làm sạch

# Tạo mask cho vùng cần ẩn (NGOÀI shrink polygon)
shrink_polygons = []

for coords in shrunk_coords_all:
    try:
        poly = Polygon(coords)
        if poly.is_valid and not poly.is_empty:
            shrink_polygons.append(poly)
    except:
        continue

# Tạo ảnh trắng để vẽ mask ẩn (chỉ ẩn vùng NGOÀI)
mask_hide = np.zeros(img.shape[:2], dtype=np.uint8)

# Vẽ vùng shrink màu đen (để bảo vệ vùng trong)
for poly in shrink_polygons:
    coords = list(poly.exterior.coords)
    cnt = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.drawContours(mask_hide, [cnt], -1, 255, thickness=cv2.FILLED)

# Đảo ngược mask (ẩn vùng NGOÀI = trắng, giữ vùng TRONG = đen)
mask_hide = 255 - mask_hide

# Áp dụng mask để ẩn vùng ngoài bằng màu trắng
img_final[mask_hide > 0] = (255, 255, 255)

# Vẽ đường viền đen 5px quanh polygon thu nhỏ
for coords in shrunk_coords_all:
    cnt = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.drawContours(img_final, [cnt], -1, (0, 0, 0), thickness=5)

# -------------------------------
# 6. Xuất kết quả
# -------------------------------
cv2.imwrite(output_img_path, img_final)
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump({"exterior_wall_shrink": shrunk_coords_all}, f, indent=2, ensure_ascii=False)
print("\nSuccessfully exported image with area OUTSIDE shrunk polygon hidden in white, area INSIDE preserved:", output_img_path)
print(f"Number of polygons: {len(shrunk_coords_all)}")
for i, coords in enumerate(shrunk_coords_all):
    print(f"   Polygon {i+1}: {len(coords)} points")
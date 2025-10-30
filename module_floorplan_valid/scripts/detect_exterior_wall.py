import cv2
import numpy as np
import os
import json
import shutil
from shapely.geometry import Polygon

# -------------------------------
# Cấu hình thư mục
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input
MASK_DIR = os.path.join(BASE_DIR, "..", "..", "model", "source", "mask")
IMG_DIR = os.path.join(BASE_DIR, "..", "..", "model", "source", "result")

# Output (đặt trong module_floorplan_valid)
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "module_floorplan_valid", "output")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSON_DIR = os.path.join(OUTPUT_DIR, "json")

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

TARGET_BASE_NAME = os.getenv("TARGET_BASE_NAME", "278")
UPSCALED_SIZE = os.getenv("UPSCALED_SIZE", "1024")
MASK_SUFFIX = f"_{UPSCALED_SIZE}" if UPSCALED_SIZE else ""

# -------------------------------
# Tự động tìm file input PNG
# -------------------------------
mask_input_path = os.path.join(MASK_DIR, f"{TARGET_BASE_NAME}{MASK_SUFFIX}.png")
drawing_input_path = os.path.join(IMG_DIR, f"{TARGET_BASE_NAME}.png")

if not os.path.exists(mask_input_path):
    raise FileNotFoundError(f"Error: Cannot find mask file: {mask_input_path}")
if not os.path.exists(drawing_input_path):
    raise FileNotFoundError(f"Error: Cannot find drawing file: {drawing_input_path}")

print(f"Processing mask: {mask_input_path}")
print(f"Processing drawing: {drawing_input_path}")

# -------------------------------
# File output
# -------------------------------
output_img_path = os.path.join(OUTPUT_IMG_DIR, f"exterior_wall_{TARGET_BASE_NAME}.png")
output_overlay_path = os.path.join(OUTPUT_IMG_DIR, f"drawing_with_exterior_{TARGET_BASE_NAME}.png")
output_json_path = os.path.join(OUTPUT_JSON_DIR, f"exterior_wall_{TARGET_BASE_NAME}.json")

# -------------------------------
# Load images
# -------------------------------
mask_img = cv2.imread(mask_input_path)
drawing_img = cv2.imread(drawing_input_path)

if mask_img is None:
    raise FileNotFoundError(f"Cannot read mask file: {mask_input_path}")
if drawing_img is None:
    raise FileNotFoundError(f"Cannot read drawing file: {drawing_input_path}")

# -------------------------------
# Xử lý ảnh mask -> phát hiện tường ngoài
# -------------------------------
gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
mask_black = cv2.inRange(gray, 0, 30)  # vùng màu đen = tường ngoài
contours, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# -------------------------------
# Vẽ ảnh tường ngoài (nền trắng)
# -------------------------------
exterior = np.ones_like(gray) * 255
cv2.drawContours(exterior, contours, -1, (0), thickness=5)
cv2.imwrite(output_img_path, exterior)

# -------------------------------
# Vẽ đè tường ngoài lên bản vẽ gốc
# -------------------------------
drawing_overlay = drawing_img.copy()
cv2.drawContours(drawing_overlay, contours, -1, (0, 0, 255), thickness=3)  # vẽ màu đỏ
cv2.imwrite(output_overlay_path, drawing_overlay)

# -------------------------------
# Xuất JSON toạ độ contour
# -------------------------------
exterior_coords = []
for cnt in contours:
    coords = cnt.squeeze().tolist()  # từ (N,1,2) -> (N,2)
    if isinstance(coords[0], list):
        exterior_coords.append(coords)
    else:
        exterior_coords.append([coords])

# 👇 Thêm phần tính CENTER của ảnh
height, width = drawing_img.shape[:2]
center_x = width // 2
center_y = height // 2
center_point = [center_x, center_y]

# 👇 Ghi dữ liệu JSON có thêm center
json_data = {
    "exterior_wall": exterior_coords,
    "center": center_point
}

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=2, ensure_ascii=False)

# -------------------------------
# Kết quả
# -------------------------------
def _copy_exterior_wall_image(source_path):
    """Copy exterior_wall_278.png to exterior_wall.png"""
    try:
        # Đường dẫn file đích - copy vào module_detect_rooms/inputs/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(script_dir, "..", "..", "module_detect_rooms", "inputs", "exterior_wall.png")
        
        # Tạo thư mục đích nếu chưa có
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy file
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
            print(f"✅ Copied {os.path.basename(source_path)} to {target_path}")
        else:
            print(f"⚠️ Source file not found: {source_path}")
            
    except Exception as e:
        print(f"⚠️ Error copying exterior wall image: {str(e)}")

print(f"\nSuccessfully exported exterior wall image: {output_img_path}")
print(f"Successfully exported overlay image: {output_overlay_path}")
print(f"Successfully exported coordinates JSON: {output_json_path}")
print(f"Image center (x, y): {center_point}")

# Copy exterior_wall_278.png to exterior_wall.png
_copy_exterior_wall_image(output_img_path)

# -------------------------------
# THU NHỎ POLYGON TƯỜNG NGOÀI (TỰ ĐỘNG)
# -------------------------------
def shrink_exterior_wall_polygon(exterior_coords, buffer_size=-10):
    """
    Thu nhỏ polygon tường ngoài inward bằng buffer_size

    Args:
        exterior_coords: List tọa độ polygon từ exterior_wall.json
        buffer_size: Số pixel thu nhỏ (âm = inward)

    Returns:
        shrunk_coords: List tọa độ đã thu nhỏ
    """
    shrunk_coords_all = []

    for polygon_coords in exterior_coords:
        try:
            # Đảm bảo polygon_coords là list của các điểm
            if isinstance(polygon_coords, list) and len(polygon_coords) > 0:
                # Đóng polygon nếu chưa đóng
                if polygon_coords[0] != polygon_coords[-1]:
                    polygon_coords = polygon_coords + [polygon_coords[0]]

                # Thu nhỏ từng cạnh của polygon
                shrunk_coords = []
                n = len(polygon_coords) - 1  # trừ điểm cuối trùng với điểm đầu

                for i in range(n):
                    p1 = polygon_coords[i]
                    p2 = polygon_coords[i + 1]

                    # Vector từ p1 đến p2
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]

                    # Độ dài cạnh
                    length = (dx**2 + dy**2) ** 0.5

                    if length == 0:
                        shrunk_coords.append(p1)
                        continue

                    # Vector đơn vị vuông góc với cạnh
                    # Để thu nhỏ inward, cần tìm hướng vuông góc vào trong polygon
                    perp_dx = -dy / length
                    perp_dy = dx / length

                    # Kiểm tra hướng inward (đối với polygon clockwise, inward là bên phải)
                    # Nhân với buffer_size (âm để thu nhỏ inward)
                    shrink_x = p1[0] + perp_dx * buffer_size
                    shrink_y = p1[1] + perp_dy * buffer_size

                    shrunk_coords.append([int(round(shrink_x)), int(round(shrink_y))])

                # Đóng polygon
                shrunk_coords.append(shrunk_coords[0])
                shrunk_coords_all.append(shrunk_coords)

        except Exception as e:
            print(f"Warning: Error shrinking polygon: {e}")
            continue

    return shrunk_coords_all

# Đường dẫn file exterior_wall_shrink.json
shrink_json_path = os.path.join(OUTPUT_JSON_DIR, f"exterior_wall_shrink_{TARGET_BASE_NAME}.json")

print(f"\nShrinking exterior wall polygon...")
shrunk_coords_all = []

for polygon_coords in exterior_coords:
    try:
        poly = Polygon(polygon_coords)
        if not poly.is_valid or poly.is_empty:
            continue

        shrunk_poly = poly.buffer(-10)
        if shrunk_poly.is_empty:
            continue

        geometries = shrunk_poly.geoms if shrunk_poly.geom_type == "MultiPolygon" else [shrunk_poly]
        for geom in geometries:
            coords = [[int(round(x)), int(round(y))] for x, y in geom.exterior.coords]
            if coords:
                shrunk_coords_all.append(coords)
    except Exception as e:
        print(f"⚠️ Lỗi thu nhỏ polygon: {e}")
        continue

if shrunk_coords_all:
    # Xuất file exterior_wall_shrink.json
    shrink_json_data = {
        "exterior_wall_shrink": shrunk_coords_all
    }

    with open(shrink_json_path, "w", encoding="utf-8") as f:
        json.dump(shrink_json_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Đã xuất exterior_wall_shrink.json: {shrink_json_path}")
    print(f"📐 Đã thu nhỏ {len(shrunk_coords_all)} polygon(s)")
    for i, coords in enumerate(shrunk_coords_all):
        print(f"   Polygon {i+1}: {len(coords)} điểm")
else:
    print("❌ Không thể tạo exterior_wall_shrink.json")

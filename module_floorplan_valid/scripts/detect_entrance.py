import cv2
import numpy as np
import json
import os

# -------------------------------
# Cấu hình đường dẫn
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Sử dụng đường dẫn tương đối từ thư mục gốc
INPUT_IMG_DIR = os.path.join(BASE_DIR, "..", "..", "module_floorplan_valid", "output", "images")
INPUT_JSON_DIR = os.path.join(BASE_DIR, "..", "..", "module_floorplan_valid", "output", "json")

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "..", "module_floorplan_valid", "output")
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSON_DIR = os.path.join(OUTPUT_DIR, "json")

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

TARGET_BASE_NAME = os.getenv("TARGET_BASE_NAME", "278")

# -------------------------------
# Đọc file ảnh và JSON
# -------------------------------
img_path = os.path.join(INPUT_IMG_DIR, f"drawing_with_exterior_{TARGET_BASE_NAME}.png")
json_outer_path = os.path.join(INPUT_JSON_DIR, f"exterior_wall_{TARGET_BASE_NAME}.json")
json_inner_path = os.path.join(INPUT_JSON_DIR, f"exterior_wall_shrink_{TARGET_BASE_NAME}.json")

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot find image: {img_path}")

with open(json_outer_path, "r") as f:
    data_outer = json.load(f)
with open(json_inner_path, "r") as f:
    data_inner = json.load(f)

poly_outer = np.array(data_outer["exterior_wall"], np.int32)
poly_inner = np.array(data_inner["exterior_wall_shrink"], np.int32)

# -------------------------------
# Tạo mask vùng giữa 2 polygon
# -------------------------------
mask_outer = np.zeros(img.shape[:2], dtype=np.uint8)
mask_inner = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask_outer, [poly_outer], 255)
cv2.fillPoly(mask_inner, [poly_inner], 255)
mask_between = cv2.subtract(mask_outer, mask_inner)

# -------------------------------
# Chuyển sang HSV và tìm màu
# -------------------------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 200, 200], dtype=np.uint8)
upper = np.array([50, 255, 255], dtype=np.uint8)
mask_color = cv2.inRange(img, lower, upper)

# Giao giữa vùng giữa và vùng màu
mask_entrance = cv2.bitwise_and(mask_color, mask_between)

# -------------------------------
# Tìm điểm entrance
# -------------------------------
ys, xs = np.where(mask_entrance > 0)
if len(xs) > 0:
    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    entrance_point = [cx, cy]
    print(f"Successfully detected entrance at: {entrance_point}")
    cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
else:
    entrance_point = None
    print("Warning: No suitable color detected in the area between walls.")

# -------------------------------
# Lưu kết quả
# -------------------------------
output_img_path = os.path.join(OUTPUT_IMG_DIR, f"entrance_detected_{TARGET_BASE_NAME}.png")
cv2.imwrite(output_img_path, img)

output_json_path = os.path.join(OUTPUT_JSON_DIR, f"entrance_{TARGET_BASE_NAME}.json")
with open(output_json_path, "w") as f:
    json.dump({"entrance": entrance_point}, f, indent=2)

print(f"Saved result: {output_img_path}")
print(f"Saved JSON: {output_json_path}")

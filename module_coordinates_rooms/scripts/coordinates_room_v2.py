"""
Phiên bản cập nhật của coordinates_room.py
Sử dụng thông tin từ room_coordinates_278.json để cải thiện việc tô màu phòng
"""

import cv2
import numpy as np
import json
import os
import shutil
from typing import List, Dict, Tuple
from room_area_calculator import RoomAreaCalculator

class RoomCoordinatesExtractor:
    """Extract room coordinates and generate labeled output"""
    
    def __init__(self, output_img_folder: str, output_json_folder: str, total_area_m2: float = 100.0):
        self.output_img_folder = output_img_folder
        self.output_json_folder = output_json_folder
        self.input_path = os.path.join(output_img_folder, "layout_final_clean.png")
        # Change output paths for different visualizations
        self.output_img_path = os.path.join(output_img_folder, "room_visualization.png")
        self.output_img_path_no_living = os.path.join(output_img_folder, "room_visualization_no_living.png")
        self.output_img_path_with_areas = os.path.join(output_img_folder, "room_visualization_with_areas.png")
        self.output_json_path = os.path.join(output_json_folder, "room_coordinates.json")
        self.output_areas_json_path = os.path.join(output_json_folder, "room_areas.json")
        
        # Initialize area calculator
        self.total_area_m2 = total_area_m2
        self.area_calculator = RoomAreaCalculator(total_area_m2=total_area_m2)
        
        # Thêm đường dẫn đến file room info - sử dụng đường dẫn tương đối từ script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_name = os.getenv("TARGET_BASE_NAME", "278")
        self.room_info_path = os.path.join(
            script_dir,
            "..",
            "..",
            "module_floorplan_valid",
            "output",
            "json",
            f"room_coordinates_{self.base_name}.json",
        )
        
        self.categories = self._define_categories()
        self.color_map = self._define_color_map()
        self.cat_name_to_id = {c["name"]: c["id"] for c in self.categories}
        
        # Load room info
        self.room_info = self._load_room_info()
    
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
            "Living": [200, 230, 240],  # BGR format
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
    
    def _load_room_info(self) -> Dict:
        """Load room information from room_coordinates_<base>.json"""
        print(f"\nFile path: {self.room_info_path}")
        print(f"File exists: {os.path.exists(self.room_info_path)}")
        
        if not os.path.exists(self.room_info_path):
            print(f"Warning: Cannot find room_coordinates_{self.base_name}.json")
            return {}
            
        try:
            with open(self.room_info_path, 'r', encoding='utf-8') as f:
                print("File opened successfully")
                room_info = json.load(f)
                print(f"Number of rooms loaded: {len(room_info)}")
                print(f"Successfully loaded room information from room_coordinates_{self.base_name}.json")
                return room_info
        except Exception as e:
            print(f"Error reading room_coordinates_{self.base_name}.json:")
            print(f"   - Error: {str(e)}")
            print(f"   - Type: {type(e)}")
            return {}
    
    def _init_coco(self, img: np.ndarray) -> Dict:
        """Initialize COCO format output"""
        h, w = 1024, 1024  # Fixed size output
        return {
            "info": {"description": "floorplan-rooms"},
            "images": [{"id": 1, "width": w, "height": h, "file_name": os.path.basename(self.input_path)}],
            "annotations": [],
            "categories": self.categories,
        }
    
    def _process_rooms(self, img: np.ndarray, annotated: np.ndarray, coco: Dict, 
                      exclude_living: bool = False, show_labels: bool = True) -> Dict:
        """Process rooms and extract coordinates"""
        ann_id = 0
        h, w = img.shape[:2]
        
        # Xử lý theo thông tin từ room_coordinates_<base>.json nếu có
        if self.room_info:
            print(f"   Using room information from room_coordinates_{self.base_name}.json")
            for room_type, info in self.room_info.items():
                if room_type in ["ExteriorWall", "InteriorWall", "Entrance"]:
                    continue
                    
                # Skip Living Room if exclude_living is True
                if exclude_living and room_type == "Living Room":
                    continue
                    
                # Map tên phòng sang tên trong color_map
                # Map room names
                name_map = {
                    "Bathroom": "Bath",
                    "Master Room": "Master",
                    "Living Room": "Living"
                }
                room_name = name_map.get(room_type, room_type.split()[0])
                bgr = self.color_map.get(room_name)
                if not bgr:
                    print(f"   Warning: Cannot find color for room: {room_type}")
                    continue
                
                # Xử lý từng phòng
                for room_data in info.get("coordinates", []):
                    center = room_data.get("center", [0, 0])
                    bbox = room_data.get("bounding_box", [0, 0, 0, 0])
                    area = room_data.get("area_pixels", 0)
                    
                    if area < 100:
                        continue
                    
                    # Scale coordinates to match 1024x1024
                    orig_h, orig_w = 1000, 1000  # Original dimensions in JSON
                    scale_x = 1024 / orig_w
                    scale_y = 1024 / orig_h
                    
                    # Scale bounding box and center
                    bbox_scaled = [
                        int(bbox[0] * scale_x),  # x
                        int(bbox[1] * scale_y),  # y
                        int(bbox[2] * scale_x),  # width
                        int(bbox[3] * scale_y)   # height
                    ]
                    x, y, w_box, h_box = bbox_scaled
                    cx = int(center[0] * scale_x)
                    cy = int(center[1] * scale_y)
                    
                    # Create and fill mask
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), 255, -1)
                    flood_mask = np.zeros((h+2, w+2), dtype=np.uint8)
                    
                    # Ensure center is within bounds and flood fill
                    if 0 <= cx < w and 0 <= cy < h:
                        cv2.floodFill(mask, flood_mask, (cx, cy), 255)
                    else:
                        print(f"   Warning: Center point ({cx}, {cy}) out of bounds for {room_type}")
                    
                    # Clean up mask
                    kernel = np.ones((3,3), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    
                    # Tìm contour
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                        
                    cnt = max(contours, key=cv2.contourArea)
                    
                    # Create polygon with fewer points
                    epsilon = 0.02 * cv2.arcLength(cnt, True)  # Increased epsilon for simpler polygon
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    
                    # Smooth the polygon
                    approx = cv2.boxPoints(cv2.minAreaRect(approx)).astype(np.int32)
                    area = float(cv2.contourArea(approx))
                    
                    # Create annotation
                    segmentation = approx.flatten().astype(float).tolist()
                    x, y, w_box, h_box = cv2.boundingRect(approx)
                    bbox = [float(x), float(y), float(w_box), float(h_box)]
                    
                    # Vẽ và ghi nhãn
                    cv2.fillPoly(annotated, [approx], bgr)
                    cv2.polylines(annotated, [approx], True, (0,0,0), 1)
                    
                    # Thêm vào COCO format
                    ann = {
                        "id": ann_id,
                        "iscrowd": 0,
                        "image_id": 1,
                        "category_id": self.cat_name_to_id[room_name],
                        "segmentation": [segmentation],
                        "bbox": bbox,
                        "area": area
                    }
                    coco["annotations"].append(ann)
                    ann_id += 1
                    
                    # Thêm nhãn nếu show_labels = True
                    if show_labels:
                        cv2.putText(annotated, room_name, (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        # Fallback to color detection if no room info
        else:
            print("   Warning: No room information, using color detection")
            for room, bgr in self.color_map.items():
                if room in ["Interior wall", "Exterior wall"]:
                    continue
                    
                # Create mask for room color with tolerance
                lower = np.clip(np.array(bgr) - 25, 0, 255)
                upper = np.clip(np.array(bgr) + 25, 0, 255)
                room_mask = cv2.inRange(img, lower, upper)
                
                # Find and process connected components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    room_mask, connectivity=8
                )
                
                for label in range(1, num_labels):
                    area = stats[label, cv2.CC_STAT_AREA]
                    if area < 100:
                        continue
                        
                    # Create mask for this component
                    component_mask = (labels == label).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                        
                    cnt = max(contours, key=cv2.contourArea)
                    
                    # Create polygon
                    epsilon = 0.01 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    area = float(cv2.contourArea(approx))
                    
                    # Create annotation
                    segmentation = approx.flatten().astype(float).tolist()
                    x, y, w_box, h_box = cv2.boundingRect(approx)
                    bbox = [float(x), float(y), float(w_box), float(h_box)]
                    
                    # Draw and label
                    cv2.fillPoly(annotated, [approx], bgr)
                    cv2.polylines(annotated, [approx], True, (0,0,0), 1)
                    
                    # Add to COCO format
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
                    
                    # Add label if show_labels is True
                    if show_labels:
                        cv2.putText(annotated, room, (x, y - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        
        return coco
    
    def _process_and_save(self, img: np.ndarray, exclude_living: bool = False, show_labels: bool = True) -> Dict:
        """Process rooms and save visualization"""
        # Always create 1024x1024 output
        h, w = 1024, 1024
        annotated = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Resize input image to 1024x1024 if needed
        if img.shape[:2] != (1024, 1024):
            img = cv2.resize(img, (1024, 1024))
        
        coco_output = self._init_coco(img)
        coco_output = self._process_rooms(img, annotated, coco_output, 
                                        exclude_living=exclude_living,
                                        show_labels=show_labels)
        
        output_path = self.output_img_path_no_living if exclude_living else self.output_img_path
        cv2.imwrite(output_path, annotated)
        print(f"   ✅ Saved visualization to: {output_path}")
        
        return coco_output
        
    def _calculate_and_add_areas(self):
        """Calculate room areas and add labels to visualization"""
        try:
            print("\n   📏 Calculating room areas...")
            
            # Step 1: Calculate exterior wall pixels
            self.area_calculator.calculate_exterior_wall_pixels(self.input_path)
            
            # Step 2: Calculate all room areas
            room_areas = self.area_calculator.calculate_all_room_areas(self.output_json_path)
            
            # Step 3: Add area labels to image
            print("\n   📊 Adding area labels to visualization...")
            self.area_calculator.add_area_labels_to_image(
                self.output_img_path, 
                room_areas, 
                self.output_img_path_with_areas
            )
            
            # Step 4: Save room areas JSON
            self.area_calculator.save_room_areas_json(room_areas, self.output_areas_json_path)
            
        except Exception as e:
            print(f"   ⚠️ Error calculating areas: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _copy_output_image(self):
        """Copy room_visualization_no_living.png to img.png"""
        try:
            # Đường dẫn file nguồn
            source_path = self.output_img_path_no_living
            
            # Đường dẫn file đích - copy vào module_detect_rooms/inputs/
            script_dir = os.path.dirname(os.path.abspath(__file__))
            target_path = os.path.join(script_dir, "..", "..", "module_detect_rooms", "inputs", "img.png")
            
            # Tạo thư mục đích nếu chưa có
            target_dir = os.path.dirname(target_path)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy file
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                print(f"   ✅ Copied {os.path.basename(source_path)} to {target_path}")
            else:
                print(f"   ⚠️ Source file not found: {source_path}")
                
        except Exception as e:
            print(f"   ⚠️ Error copying image: {str(e)}")

    def run(self):
        """Run coordinate extraction"""
        print("\n📍 Step 7: Extracting room coordinates...")
        
        # Check if input file exists
        if not os.path.exists(self.input_path):
            print(f"Error: File does not exist: {self.input_path}")
            print("Please run coordinates_room_001.py first to create layout_final_clean.png")
            return
            
        img = cv2.imread(self.input_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read input image at {self.input_path}")
        
        # Generate both visualizations
        print("\n   💡 Generating visualization with all rooms and labels...")
        coco_output = self._process_and_save(img, exclude_living=False, show_labels=True)
        
        print("\n   💡 Generating visualization without Living room and labels...")
        self._process_and_save(img, exclude_living=True, show_labels=False)
        
        # Save COCO format data
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(coco_output, f, indent=4)
        print(f"   ✅ Saved coordinates JSON: {self.output_json_path}")
        
        # Calculate and add area labels
        self._calculate_and_add_areas()
        
        # Copy room_visualization_no_living.png to img.png
        self._copy_output_image()

def main():
    """Main entry point"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.dirname(script_dir)
    
    output_img_dir = os.path.join(module_dir, "output", "images")
    output_json_dir = os.path.join(module_dir, "output", "json")
    
    # Get total area from environment variable or use default
    total_area = float(os.getenv("TOTAL_AREA_M2", "100.0"))
    
    extractor = RoomCoordinatesExtractor(output_img_dir, output_json_dir, total_area_m2=total_area)
    extractor.run()

if __name__ == "__main__":
    main()
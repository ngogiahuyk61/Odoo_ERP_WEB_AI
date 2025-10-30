"""
Module tính toán diện tích phòng dựa trên tổng diện tích và pixel
Calculates room areas based on total area and pixel count
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, Tuple, List


class RoomAreaCalculator:
    """Calculate room areas based on total area input"""
    
    def __init__(self, total_area_m2: float = 100.0):
        """
        Initialize calculator with total area
        
        Args:
            total_area_m2: Total area in square meters (default: 100.0)
        """
        self.total_area_m2 = total_area_m2
        self.pixel_to_m2_ratio = 0.0
        self.exterior_wall_pixels = 0
        
    def calculate_exterior_wall_pixels(self, image_path: str) -> int:
        """
        Calculate the number of pixels inside exterior walls
        
        Args:
            image_path: Path to the floorplan image
            
        Returns:
            Number of pixels inside exterior walls
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find non-white areas (walls are black/dark, rooms are colored)
        # White pixels (255) are background, everything else is inside the floorplan
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        
        # Count non-zero pixels (everything inside exterior walls)
        interior_pixels = cv2.countNonZero(binary)
        
        self.exterior_wall_pixels = interior_pixels
        
        # Calculate pixel to m2 ratio
        if interior_pixels > 0:
            self.pixel_to_m2_ratio = self.total_area_m2 / interior_pixels
        
        print(f"   📏 Interior pixels: {interior_pixels}")
        print(f"   📏 Total area: {self.total_area_m2} m²")
        print(f"   📏 Pixel to m² ratio: {self.pixel_to_m2_ratio:.6f}")
        
        return interior_pixels
    
    def calculate_room_area_from_pixels(self, pixel_count: float) -> float:
        """
        Calculate room area in m² from pixel count
        
        Args:
            pixel_count: Number of pixels in the room
            
        Returns:
            Area in square meters
        """
        if self.pixel_to_m2_ratio == 0:
            raise ValueError("Pixel to m² ratio not calculated. Run calculate_exterior_wall_pixels first.")
        
        return pixel_count * self.pixel_to_m2_ratio
    
    def calculate_all_room_areas(self, room_coordinates_json: str) -> Dict[str, Dict]:
        """
        Calculate areas for all rooms from room coordinates JSON
        
        Args:
            room_coordinates_json: Path to room_coordinates.json file
            
        Returns:
            Dictionary with room information including areas
        """
        if not os.path.exists(room_coordinates_json):
            raise FileNotFoundError(f"Room coordinates file not found: {room_coordinates_json}")
        
        with open(room_coordinates_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create category mapping
        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        
        # Calculate areas for each room
        room_areas = {}
        total_calculated_area = 0.0
        
        print("\n   📊 Room Areas:")
        print("   " + "="*50)
        
        for ann in data.get('annotations', []):
            room_id = ann['id']
            category_id = ann['category_id']
            room_name = categories.get(category_id, f"Room_{category_id}")
            pixel_area = ann['area']
            
            # Skip walls
            if room_name in ['Exterior wall', 'Interior wall']:
                continue
            
            # Calculate area in m²
            area_m2 = self.calculate_room_area_from_pixels(pixel_area)
            total_calculated_area += area_m2
            
            # Store room information
            room_key = f"{room_name}_{room_id}"
            room_areas[room_key] = {
                'id': room_id,
                'name': room_name,
                'category_id': category_id,
                'pixel_area': pixel_area,
                'area_m2': round(area_m2, 2),
                'bbox': ann['bbox'],
                'segmentation': ann['segmentation']
            }
            
            print(f"   • {room_name:15} : {area_m2:6.2f} m² ({pixel_area:8.0f} pixels)")
        
        print("   " + "="*50)
        print(f"   • {'Total':15} : {total_calculated_area:6.2f} m²")
        print(f"   • {'Input Total':15} : {self.total_area_m2:6.2f} m²")
        print(f"   • {'Difference':15} : {abs(total_calculated_area - self.total_area_m2):6.2f} m²")
        
        return room_areas
    
    def add_area_labels_to_image(self, 
                                  image_path: str, 
                                  room_areas: Dict[str, Dict],
                                  output_path: str) -> None:
        """
        Add area labels to room visualization image
        
        Args:
            image_path: Path to the input image
            room_areas: Dictionary with room area information
            output_path: Path to save the output image
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        # Add area labels to each room
        for room_key, room_info in room_areas.items():
            bbox = room_info['bbox']
            x, y, w, h = [int(v) for v in bbox]
            
            room_name = room_info['name']
            area_m2 = room_info['area_m2']
            
            # Calculate text position (center-top of bounding box)
            text_x = x + w // 2
            text_y = y + 15  # Slightly below the top
            
            # Add room name
            cv2.putText(img, room_name, 
                       (text_x - 30, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Add area label below room name
            area_text = f"{area_m2} m²"
            cv2.putText(img, area_text, 
                       (text_x - 25, text_y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.45, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Save output image
        cv2.imwrite(output_path, img)
        print(f"\n   ✅ Saved image with area labels: {output_path}")
    
    def save_room_areas_json(self, room_areas: Dict[str, Dict], output_path: str) -> None:
        """
        Save room areas to JSON file
        
        Args:
            room_areas: Dictionary with room area information
            output_path: Path to save the JSON file
        """
        output_data = {
            'total_area_m2': self.total_area_m2,
            'pixel_to_m2_ratio': self.pixel_to_m2_ratio,
            'exterior_wall_pixels': self.exterior_wall_pixels,
            'rooms': room_areas
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        
        print(f"   ✅ Saved room areas JSON: {output_path}")


def main():
    """Main entry point for testing"""
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    module_dir = os.path.dirname(script_dir)
    
    # Input/output paths
    floorplan_image = os.path.join(module_dir, "output", "images", "layout_final_clean.png")
    room_coordinates_json = os.path.join(module_dir, "output", "json", "room_coordinates.json")
    visualization_image = os.path.join(module_dir, "output", "images", "room_visualization.png")
    
    output_image = os.path.join(module_dir, "output", "images", "room_visualization_with_areas.png")
    output_json = os.path.join(module_dir, "output", "json", "room_areas.json")
    
    # Default total area (can be changed via environment variable or parameter)
    total_area = float(os.getenv("TOTAL_AREA_M2", "100.0"))
    
    print("\n" + "="*60)
    print("🏠 ROOM AREA CALCULATOR")
    print("="*60)
    
    # Initialize calculator
    calculator = RoomAreaCalculator(total_area_m2=total_area)
    
    # Step 1: Calculate exterior wall pixels
    print("\n📍 Step 1: Calculating exterior wall pixels...")
    calculator.calculate_exterior_wall_pixels(floorplan_image)
    
    # Step 2: Calculate all room areas
    print("\n📍 Step 2: Calculating room areas...")
    room_areas = calculator.calculate_all_room_areas(room_coordinates_json)
    
    # Step 3: Add area labels to image
    print("\n📍 Step 3: Adding area labels to visualization...")
    calculator.add_area_labels_to_image(visualization_image, room_areas, output_image)
    
    # Step 4: Save room areas JSON
    print("\n📍 Step 4: Saving room areas JSON...")
    calculator.save_room_areas_json(room_areas, output_json)
    
    print("\n" + "="*60)
    print("✅ COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()

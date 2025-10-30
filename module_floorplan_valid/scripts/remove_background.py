import argparse
import os
from pathlib import Path
import shutil

from rembg import remove


# Sử dụng đường dẫn tương đối từ thư mục gốc
BASE_DIR = Path(__file__).parent
TARGET_BASE_NAME = os.getenv("TARGET_BASE_NAME", "278")
INPUT_PATH = BASE_DIR / ".." / ".." / "module_floorplan_valid" / "output" / "images" / f"drawing_with_black_walls_{TARGET_BASE_NAME}.png"
OUTPUT_PATH = BASE_DIR / ".." / ".." / "module_coordinates_rooms" / "input" / "drawing_overwrite.png"
EXTERIOR_IMG_SOURCE = BASE_DIR / ".." / ".." / "module_floorplan_valid" / "output" / "images" / f"exterior_wall_{TARGET_BASE_NAME}.png"
EXTERIOR_IMG_TARGET = BASE_DIR / ".." / ".." / "module_coordinates_rooms" / "input" / "exterior_wall.png"
EXTERIOR_JSON_SOURCE = BASE_DIR / ".." / ".." / "module_floorplan_valid" / "output" / "json" / f"exterior_wall_shrink_{TARGET_BASE_NAME}.json"
EXTERIOR_JSON_TARGET = BASE_DIR / ".." / ".." / "module_coordinates_rooms" / "input" / "exterior_wall_shrink.json"


def remove_background(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("rb") as source:
        input_bytes = source.read()
    result = remove(input_bytes)
    with output_path.open("wb") as target:
        target.write(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    remove_background(args.input, args.output)

    EXTERIOR_IMG_TARGET.parent.mkdir(parents=True, exist_ok=True)
    EXTERIOR_JSON_TARGET.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(EXTERIOR_IMG_SOURCE, EXTERIOR_IMG_TARGET)
    shutil.copy2(EXTERIOR_JSON_SOURCE, EXTERIOR_JSON_TARGET)


if __name__ == "__main__":
    main()

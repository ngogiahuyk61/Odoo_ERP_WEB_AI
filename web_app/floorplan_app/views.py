import json
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


MODEL_DIR = Path(settings.BASE_DIR).parent / "ai_processing" / "model"
SOURCE_DIR = MODEL_DIR / "source"
TEXT_INPUT_DIR = SOURCE_DIR / "text_input"
QUEUE_DIR = SOURCE_DIR / "queue"
NEW_TEXT_DIR = SOURCE_DIR / "new_text"
RESULT_DIR = SOURCE_DIR / "result"
QUEUE_PROCESSED_DIR = QUEUE_DIR / "processed"
QUEUE_FAILED_DIR = QUEUE_DIR / "failed"
STATIC_DIR = Path(settings.BASE_DIR) / "floorplan_app" / "static" / "floorplan_app" / "img"
STATICFILES_DIR = Path(settings.BASE_DIR) / "staticfiles" / "floorplan_app" / "img"
HERO_IMAGE_REL = "floorplan_app/img/proposal1.png"
GALLERY_IMAGE_REL = "floorplan_app/img/floorplan1.png"
COLLECT_MASK_INPUT_DIR = Path(settings.BASE_DIR).parent / "ai_processing" / "module_collect_mask" / "input"
SITE_DATA_PATH = COLLECT_MASK_INPUT_DIR / "site_data.json"

for directory in (TEXT_INPUT_DIR, QUEUE_DIR, NEW_TEXT_DIR, RESULT_DIR, QUEUE_PROCESSED_DIR, QUEUE_FAILED_DIR, COLLECT_MASK_INPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_total_area_from_text_file(relative_path: str | None) -> float | None:
    if not relative_path:
        return None
    text_path = MODEL_DIR / Path(relative_path)
    if not text_path.exists():
        return None
    try:
        payload = json.loads(text_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _coerce_float(payload.get("total_area_m2"))


def _latest_text_request_file() -> Path | None:
    candidates = sorted(
        TEXT_INPUT_DIR.glob("floorplan-request-*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_latest_request_summary() -> dict:
    latest_file = _latest_text_request_file()
    if not latest_file:
        return {"text": None, "total_area_m2": None}
    try:
        payload = json.loads(latest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"text": None, "total_area_m2": None}
    return {
        "text": payload.get("text"),
        "total_area_m2": _coerce_float(payload.get("total_area_m2")),
    }


# Trang chính: render ra index.html
def index(request):
    summary = _load_latest_request_summary()
    context = {
        "latest_description": summary.get("text"),
        "latest_area_plan": summary.get("total_area_m2"),
    }
    return render(request, "floorplan_app/index.html", context)


@csrf_exempt
def save_text_request(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)

    text = payload.get("text")
    if not isinstance(text, str) or not text.strip():
        return JsonResponse({"error": "`text` must be a non-empty string"}, status=400)
    
    # Get total area (default to 100.0 if not provided)
    total_area_m2 = payload.get("total_area_m2", 100.0)
    try:
        total_area_m2 = float(total_area_m2)
        if total_area_m2 <= 0:
            total_area_m2 = 100.0
    except (ValueError, TypeError):
        total_area_m2 = 100.0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"floorplan-request-{timestamp}.json"
    filepath = TEXT_INPUT_DIR / filename
    filepath.write_text(json.dumps({"text": text, "total_area_m2": total_area_m2}, ensure_ascii=False, indent=2), encoding="utf-8")

    job_id = timestamp
    new_text_rel = f"source/new_text/{job_id}.json"
    result_rel = f"source/result/{job_id}.png"

    enqueue_payload = {
        "job_id": job_id,
        "text_file": str(filepath.relative_to(MODEL_DIR)),
        "description": text.strip(),
        "total_area_m2": total_area_m2,
        "queued_at": timestamp,
        "new_text_file": new_text_rel,
        "result_image": result_rel,
    }
    queue_filename = f"queue-{job_id}.json"
    queue_path = QUEUE_DIR / queue_filename
    queue_path.write_text(json.dumps(enqueue_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return JsonResponse({
        "success": True,
        "filename": filename,
        "queue_file": queue_filename,
        "job_id": job_id,
        "new_text_json": new_text_rel,
        "result_image": result_rel,
        "hero_image": f"/{HERO_IMAGE_REL}",
        "gallery_image": f"/{GALLERY_IMAGE_REL}",
    })


@csrf_exempt
def save_land_settings(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({"error": "Invalid JSON payload"}, status=400)

    try:
        length = int(payload.get("length"))
        width = int(payload.get("width"))
    except (TypeError, ValueError):
        return JsonResponse({"error": "`length` and `width` must be integers"}, status=400)

    if length <= 0 or width <= 0:
        return JsonResponse({"error": "`length` and `width` must be positive"}, status=400)

    grid = payload.get("grid")
    if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
        return JsonResponse({"error": "`grid` must be a 2D array"}, status=400)

    if any(len(row) != width for row in grid) or len(grid) != length:
        return JsonResponse({"error": "`grid` dimensions must match `length` and `width`"}, status=400)

    try:
        SITE_DATA_PATH.write_text(
            json.dumps({"length": length, "width": width, "grid": grid}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse({"success": True, "path": str(SITE_DATA_PATH)})


def job_status(request, job_id: str):
    job_id = (job_id or "").strip()
    if not job_id:
        return JsonResponse({"error": "Invalid job_id"}, status=400)

    queue_filename = f"queue-{job_id}.json"
    queue_path = QUEUE_DIR / queue_filename
    processed_path = QUEUE_PROCESSED_DIR / queue_filename
    failed_path = QUEUE_FAILED_DIR / queue_filename

    if processed_path.exists():
        payload = json.loads(processed_path.read_text(encoding="utf-8"))

        # Get modification times for both images to use as version
        hero_image_path = STATIC_DIR / Path(HERO_IMAGE_REL).name
        gallery_image_path = STATIC_DIR / Path(GALLERY_IMAGE_REL).name
        
        hero_version = None
        gallery_version = None
        
        if hero_image_path.exists():
            hero_version = int(hero_image_path.stat().st_mtime)
        
        if gallery_image_path.exists():
            gallery_version = int(gallery_image_path.stat().st_mtime)
        
        # Use the latest version for cache busting
        version = max(hero_version or 0, gallery_version or 0)
        
        hero_url = f"{settings.STATIC_URL}{HERO_IMAGE_REL}"
        gallery_url = f"{settings.STATIC_URL}{GALLERY_IMAGE_REL}"

        total_area = _load_total_area_from_text_file(payload.get("text_file"))
        if total_area is None:
            total_area = _coerce_float(payload.get("total_area_m2"))

        return JsonResponse(
            {
                "status": "completed",
                "job_id": job_id,
                "queue_file": queue_filename,
                "new_text_json": payload.get("new_text_file"),
                "result_image": payload.get("result_image"),
                "hero_image_url": hero_url,
                "gallery_image_url": gallery_url,
                "version": version,
                "hero_version": hero_version,
                "gallery_version": gallery_version,
                "total_area_m2": total_area,
            }
        )

    if failed_path.exists():
        log_path = failed_path.with_suffix(".log")
        error_message = log_path.read_text(encoding="utf-8") if log_path.exists() else "Unknown error"
        payload = json.loads(failed_path.read_text(encoding="utf-8")) if failed_path.exists() else {}
        total_area = _load_total_area_from_text_file(payload.get("text_file")) if isinstance(payload, dict) else None
        if total_area is None and isinstance(payload, dict):
            total_area = _coerce_float(payload.get("total_area_m2"))
        return JsonResponse({"status": "failed", "job_id": job_id, "error": error_message, "total_area_m2": total_area})

    if queue_path.exists():
        payload = json.loads(queue_path.read_text(encoding="utf-8"))
        total_area = _load_total_area_from_text_file(payload.get("text_file"))
        if total_area is None:
            total_area = _coerce_float(payload.get("total_area_m2"))
        return JsonResponse({"status": "pending", "job_id": job_id, "total_area_m2": total_area})

@csrf_exempt
def single_job_request(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        # TODO: Implement the actual single job processing logic
        # This appears to be for processing a specific mask file (14.png)
        # and generating floorplan outputs. For now, returning success.

        # Simulate processing delay
        import time
        time.sleep(1)

        return JsonResponse({
            "success": True,
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

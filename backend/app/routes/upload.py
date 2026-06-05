import os
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter(prefix="/api")

# Directories
UPLOADS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "uploads")
)
FLOW_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "flow")
)

os.makedirs(UPLOADS_DIR, exist_ok=True)

class UploadResponse(BaseModel):
  job_id: str
  filename: str
  status: str
  detected_object: str
  scale_estimate: float # mm per pixel
  scale_status: str
  closest_preset: str
  preview_url: str

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Accepts image/video, runs keyframe extraction, marker scale detection, silhouette contour extraction, and maps to the closest preset case."""
    # Generate unique Job ID
    job_id = str(uuid.uuid4())
    
    # Save the file temporarily
    ext = os.path.splitext(file.filename)[1].lower() if file.filename else ".jpg"
    temp_filename = f"upload_{job_id}{ext}"
    temp_path = os.path.join(UPLOADS_DIR, temp_filename)
    
    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # 1. Keyframe extraction (if video)
    frame = None
    if ext in [".mp4", ".mov", ".avi", ".mkv"]:
        try:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Grab frame at 50% through the video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise Exception("Failed to read video frame")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Error extracting video keyframe: {str(e)}")
    else:
        # Load image directly
        try:
            frame = cv2.imread(temp_path)
            if frame is None:
                raise Exception("Invalid image file format")
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise HTTPException(status_code=400, detail=f"Error reading image: {str(e)}")

    # Resize frame to standard height=600 for faster CV processing
    h, w = frame.shape[:2]
    target_h = 600
    target_w = int(w * (target_h / h))
    frame = cv2.resize(frame, (target_w, target_h))

    # Save extracted base keyframe for reference
    keyframe_path = os.path.join(UPLOADS_DIR, f"frame_{job_id}.jpg")
    cv2.imwrite(keyframe_path, frame)

    # 2. Calibration Card (scale) and Object Silhouette processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding (Otsu + Canny edges)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    scale_estimate = 1.0  # default 1.0 mm per pixel
    scale_status = "Default Scale (No marker detected)"
    marker_contour = None

    # Try to find rectangular credit-card shape for calibration
    # Standard credit card aspect ratio is ~1.58
    for c in contours:
        area = cv2.contourArea(c)
        if area < 1000: # ignore small noise
            continue
        
        # Approximate contour to polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # If it has 4 vertices, it might be our card
        if len(approx) == 4:
            (x, y, w_box, h_box) = cv2.boundingRect(approx)
            aspect_ratio = float(w_box) / float(h_box) if h_box > 0 else 0
            
            # Aspect ratio range for card (horizontal or vertical rotation)
            if 1.3 <= aspect_ratio <= 1.8 or 0.55 <= aspect_ratio <= 0.75:
                marker_contour = c
                # Card width is standard 85.6 mm
                card_pixels = max(w_box, h_box)
                scale_estimate = 85.6 / card_pixels
                scale_status = f"Calibration Card Detected ({card_pixels}px = 85.6mm)"
                break

    # 3. Find the main object contour
    # Largest contour that is not the marker contour
    object_contour = None
    max_area = 0
    for c in contours:
        if marker_contour is not None and np.array_equal(c, marker_contour):
            continue
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            object_contour = c

    # Fallback if no object contour is found: use center box
    if object_contour is None:
        object_contour = np.array([
            [[int(target_w*0.3), int(target_h*0.3)]],
            [[int(target_w*0.7), int(target_h*0.3)]],
            [[int(target_w*0.7), int(target_h*0.7)]],
            [[int(target_w*0.3), int(target_h*0.7)]]
        ])

    # Draw previews
    preview_frame = frame.copy()
    if marker_contour is not None:
        cv2.drawContours(preview_frame, [marker_contour], -1, (0, 242, 254), 3) # Neon Cyan for card
    cv2.drawContours(preview_frame, [object_contour], -1, (124, 58, 237), 3) # Neon Purple for object

    preview_path = os.path.join(UPLOADS_DIR, f"preview_{job_id}.jpg")
    cv2.imwrite(preview_path, preview_frame)

    # 4. Generate 2D simulation binary grid of size (64, 128)
    # Create empty black canvas
    binary_grid = np.zeros((64, 128), dtype=np.uint8)
    
    # We want to fit/scale the object contour inside the grid center.
    # Translate and scale contour to fit inside grid x index [30, 90], y index [16, 48]
    (ox, oy, ow, oh) = cv2.boundingRect(object_contour)
    
    # Scale factors to fit within 60x32 center region of 128x64 grid
    max_target_w = 60
    max_target_h = 32
    scale_factor = min(max_target_w / ow, max_target_h / oh)
    
    scaled_w = int(ow * scale_factor)
    scaled_h = int(oh * scale_factor)
    
    # Shift contour points to center of grid
    cx_grid = 64
    cy_grid = 32
    start_x = cx_grid - scaled_w // 2
    start_y = cy_grid - scaled_h // 2
    
    # Extract contour points relative to bounding box, scale them, and shift to grid center
    scaled_pts = []
    for pt in object_contour:
        pt_x = int((pt[0][0] - ox) * scale_factor + start_x)
        pt_y = int((pt[0][1] - oy) * scale_factor + start_y)
        scaled_pts.append([[pt_x, pt_y]])
        
    scaled_pts_arr = np.array(scaled_pts, dtype=np.int32)
    
    # Draw filled polygon on the LBM grid
    cv2.fillPoly(binary_grid, [scaled_pts_arr], 255)
    
    # Save the custom boundary mask `.npy` file
    mask_npy_path = os.path.join(UPLOADS_DIR, f"mask_{job_id}.npy")
    np.save(mask_npy_path, (binary_grid > 0).astype(bool))

    # 5. Template matching (IoU comparison against precomputed masks)
    best_preset = "sports_car_v1"
    max_iou = 0.0
    
    preset_files = ["sports_car_v1", "drone_v1", "airfoil_v1"]
    upload_mask = (binary_grid > 0).astype(bool)
    
    for preset in preset_files:
        preset_mask_path = os.path.join(FLOW_ASSETS_DIR, f"{preset}_mask.npy")
        if os.path.exists(preset_mask_path):
            preset_mask = np.load(preset_mask_path)
            
            # Compute intersection and union
            intersection = np.logical_and(upload_mask, preset_mask)
            union = np.logical_or(upload_mask, preset_mask)
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            
            if iou > max_iou:
                max_iou = iou
                best_preset = preset

    # Map preset case names for display
    detected_object = "Car profile detected"
    if best_preset == "drone_v1":
        detected_object = "Quadcopter profile detected"
    elif best_preset == "airfoil_v1":
        detected_object = "Symmetric Wing profile detected"

    # Clean up uploaded raw source file to save storage
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return UploadResponse(
        job_id=job_id,
        filename=file.filename if file.filename else "upload.jpg",
        status="completed",
        detected_object=detected_object,
        scale_estimate=scale_estimate,
        scale_status=scale_status,
        closest_preset=best_preset,
        preview_url=f"/api/upload/preview/{job_id}"
    )

@router.get("/upload/preview/{job_id}")
async def get_preview_image(job_id: str):
    """Exposes preview contour image to frontend client."""
    file_path = os.path.join(UPLOADS_DIR, f"preview_{job_id}.jpg")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Preview contour not found")
    return FileResponse(file_path, media_type="image/jpeg")

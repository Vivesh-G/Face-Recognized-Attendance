"""
main.py  –  FastAPI Attendance System with Part-fViT
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import cv2
import base64
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from PIL import Image

from LAFS.face_engine import FaceEngine
from attendance_manager import (
    mark_present, get_attendance, update_attendance,
    ensure_today, get_csv_path
)

DATASET_DIR = "dataset"
EMBEDDINGS_DIR = "embeddings"
MODEL_PATH = "lafs_webface_finetune_withaugmentation.pth"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

engine = FaceEngine(model_path=MODEL_PATH, threshold=0.5)


# ──────────────────────────── Models ──────────────────────────────── #
class AttendanceUpdate(BaseModel):
    """Request model for attendance status update."""
    name: str
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load database on startup."""
    engine.load_db(EMBEDDINGS_DIR)
    # Always ensure attendance CSV is created
    people_meta = {k: v["meta"] for k, v in engine.db.items()} if engine.db else {}
    ensure_today(people_meta)
    print(f"[INFO] Loaded {len(engine.db)} people, attendance CSV ready")
    yield


app = FastAPI(title="Part-fViT Attendance System", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ──────────────────────────── Camera State ──────────────────────────── #
_cam_lock = threading.Lock()
_latest_frame = None  # annotated BGR frame
_latest_results: list[dict] = []
_camera_active = False
_marked_today: set[str] = set()


def _camera_worker():
    """Background camera processing thread."""
    global _latest_frame, _latest_results, _camera_active, _marked_today

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        _camera_active = False
        return

    frame_n = 0
    current_results: list[dict] = []

    while _camera_active:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_n += 1

        # Run recognition every 3rd frame for efficiency
        if frame_n % 3 == 0:
            current_results = engine.process_frame(frame)
            _latest_results = current_results

            # Auto-mark attendance
            people_meta = {k: v["meta"] for k, v in engine.db.items()}
            for r in current_results:
                if r["name"] != "UNK" and r["name"] not in _marked_today:
                    if mark_present(r["name"], r["sim"], people_meta):
                        _marked_today.add(r["name"])
                        print(f"[ATTEND] Marked present: {r['name']}")

        # Draw boxes on every frame (smooth video)
        display = frame.copy()
        for r in current_results:
            x1, y1, x2, y2 = r["box"]
            is_known = r["name"] != "UNK"
            color = (0, 220, 80) if is_known else (0, 60, 220)
            label = f"{r['name']} {r['sim']*100:.1f}%"
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(display, (x1, y1 - 24), (x1 + len(label)*11, y1), color, -1)
            cv2.putText(display, label, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        with _cam_lock:
            _latest_frame = display

    cap.release()
    print("[CAM] Released")


def _gen_mjpeg():
    """Generator for MJPEG streaming."""
    while True:
        with _cam_lock:
            frame = _latest_frame
        if frame is not None:
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        else:
            time.sleep(0.1)


# ──────────────────────────── Routes ──────────────────────────────── #

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve HTML homepage."""
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/api/camera/start")
async def start_camera():
    """Start camera feed."""
    global _camera_active, _marked_today
    if not _camera_active:
        _camera_active = True
        _marked_today = set()
        threading.Thread(target=_camera_worker, daemon=True).start()
    return {"status": "Camera started"}


@app.post("/api/camera/stop")
async def stop_camera():
    """Stop camera feed."""
    global _camera_active
    _camera_active = False
    return {"status": "Camera stopped"}


@app.get("/api/camera/stream")
async def stream_camera():
    """Stream camera MJPEG."""
    return StreamingResponse(_gen_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/detections")
async def get_detections():
    """Get latest detected faces."""
    with _cam_lock:
        results = _latest_results.copy()
    return {"detections": results}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload and process a single image."""
    try:
        data = await file.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        results = engine.process_frame(img)
        return {
            "filename": file.filename,
            "detections": results,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/enroll")
async def enroll_face(name: str, cls: str = "", dept: str = "", file: UploadFile = File(...)):
    """Enroll a person from an image."""
    try:
        # Save temp file
        temp_path = os.path.join(DATASET_DIR, f"temp_{file.filename}")
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Enroll
        success = engine.enroll(name, temp_path, meta={"class": cls, "dept": dept})
        if success:
            engine.save_embedding(name, EMBEDDINGS_DIR)
            # Update attendance CSV with new person
            people_meta = {k: v["meta"] for k, v in engine.db.items()}
            ensure_today(people_meta)
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return {"status": "Enrolled", "name": name}
        else:
            return JSONResponse({"error": "No face detected"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/rebuild-db")
async def rebuild_db():
    """Rebuild database from dataset directory with metadata extraction."""
    try:
        # Clear engine memory
        engine.db.clear()
        
        # Clear existing embeddings on disk
        if os.path.exists(EMBEDDINGS_DIR):
            for f in os.listdir(EMBEDDINGS_DIR):
                if f.endswith('.npy') or f.endswith('.pkl'):
                    os.remove(os.path.join(EMBEDDINGS_DIR, f))

        # Scan dataset
        for person_dir in os.listdir(DATASET_DIR):
            person_path = os.path.join(DATASET_DIR, person_dir)
            if not os.path.isdir(person_path):
                continue

            # Process all images for this person
            enrolled = False
            for fname in sorted(os.listdir(person_path)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, fname)
                    
                    # Metadata extraction from filename: "Name_Num__Class__Dept.jpg"
                    # Split by "__" to get [Name_Num, Class, Dept.jpg]
                    parts = fname.split("__")
                    cls = ""
                    dept = ""
                    if len(parts) >= 3:
                        cls = parts[1]
                        dept = parts[2].rsplit(".", 1)[0]
                    
                    # Enroll (using folder name as the primary identity key)
                    success = engine.enroll(person_dir, img_path, meta={"class": cls, "dept": dept})
                    if success:
                        engine.save_embedding(person_dir, EMBEDDINGS_DIR)
                        enrolled = True
                        # For now, we only need one good embedding per person
                        break

        # Re-initialize attendance CSV
        people_meta = {k: v["meta"] for k, v in engine.db.items()}
        
        # Force recreate today's CSV by deleting it if it exists
        path = get_csv_path()
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"[WARN] Could not remove old CSV: {e}")
        
        ensure_today(people_meta)

        return {"status": f"Successfully rebuilt database with {len(engine.db)} identities"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/attendance")
async def get_today_attendance():
    """Get today's attendance records."""
    try:
        records = get_attendance()
        present = sum(1 for r in records if r["Status"] == "Present")
        absent = sum(1 for r in records if r["Status"] == "Absent")
        return {
            "total": len(records),
            "present": present,
            "absent": absent,
            "records": records,
        }
    except Exception as e:
        print(f"[ERROR] Failed to get attendance: {e}")
        return {
            "total": 0,
            "present": 0,
            "absent": 0,
            "records": [],
        }


@app.post("/api/attendance/update")
async def update_attend(request: AttendanceUpdate):
    """Update attendance status for a person."""
    try:
        people_meta = {k: v["meta"] for k, v in engine.db.items()}
        success = update_attendance(request.name, request.status, people_meta)
        if success:
            return {"status": "Updated"}
        else:
            return JSONResponse({"error": "Person not found"}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/attendance/export")
async def export_attendance():
    """Download today's attendance as CSV."""
    csv_path = get_csv_path()
    if os.path.exists(csv_path):
        return FileResponse(csv_path, filename=os.path.basename(csv_path))
    return JSONResponse({"error": "No attendance data"}, status_code=404)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
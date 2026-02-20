"""
FastAPI Backend — Real-time surveillance dashboard server.
WebSocket video streaming, REST API for alerts/persons/evidence.
"""
import os
import sys
import json
import time
import asyncio
import base64
import cv2
import numpy as np
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import config

app = FastAPI(title="AI Behavior Detection System", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ============================================================
# Shared State (set by main.py when starting)
# ============================================================
class SystemState:
    """Shared state between main pipeline and dashboard."""
    def __init__(self):
        self.is_running = False
        self.current_frame = None          # Latest annotated frame (JPEG bytes)
        self.current_scores = {}           # track_id -> BehaviorScore
        self.alert_system = None           # Reference to AlertSystem
        self.recognizer = None             # Reference to PersonRecognizer
        self.evidence_recorder = None      # Reference to EvidenceRecorder
        self.tracker = None                # Reference to PersonTracker
        self.zone_manager = None           # Reference to ZoneManager
        self.fps = 0.0
        self.frame_count = 0
        self.persons_count = 0
        self.alerts_queue = asyncio.Queue() if asyncio else None
        

state = SystemState()

# Active WebSocket connections
active_connections: List[WebSocket] = []


# ============================================================
# WebSocket — Real-time Video Stream
# ============================================================
@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    """Stream annotated video frames via WebSocket."""
    await websocket.accept()
    active_connections.append(websocket)
    print(f"[Dashboard] Video client connected ({len(active_connections)} total)")
    
    try:
        while True:
            if state.current_frame is not None:
                # Send frame as base64 JPEG
                frame_data = base64.b64encode(state.current_frame).decode('utf-8')
                await websocket.send_json({
                    "type": "frame",
                    "data": frame_data,
                    "fps": round(state.fps, 1),
                    "persons": state.persons_count,
                    "frame_count": state.frame_count
                })
            
            await asyncio.sleep(1 / 30)  # ~30 FPS target
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"[Dashboard] Video client disconnected ({len(active_connections)} total)")
    except Exception as e:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket):
    """Stream real-time alerts via WebSocket."""
    await websocket.accept()
    
    try:
        while True:
            # Send current behavior scores
            scores_data = {}
            for tid, score in state.current_scores.items():
                scores_data[str(tid)] = {
                    "track_id": score.track_id,
                    "total_score": score.total_score,
                    "alert_level": score.alert_level,
                    "reasons": score.reasons,
                    "person_name": score.person_name,
                    "action_score": score.action_score,
                    "trajectory_score": score.trajectory_score,
                    "pose_score": score.pose_score,
                    "zone_score": score.zone_score
                }
            
            await websocket.send_json({
                "type": "scores",
                "data": scores_data,
                "timestamp": time.time()
            })
            
            await asyncio.sleep(0.5)  # 2 updates/sec
    except WebSocketDisconnect:
        pass


# ============================================================
# REST API — Alerts
# ============================================================
@app.get("/api/alerts")
async def get_alerts(count: int = 50):
    """Get recent alerts."""
    if state.alert_system is None:
        return {"alerts": []}
    
    alerts = state.alert_system.get_recent_alerts(count)
    return {"alerts": [a.to_dict() for a in alerts]}


@app.get("/api/alerts/stats")
async def get_alert_stats():
    """Get alert statistics."""
    if state.alert_system is None:
        return {"stats": {}}
    return {"stats": state.alert_system.get_alert_stats()}


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    if state.alert_system and state.alert_system.acknowledge_alert(alert_id):
        return {"status": "acknowledged"}
    raise HTTPException(status_code=404, detail="Alert not found")


# ============================================================
# REST API — Persons
# ============================================================
@app.get("/api/persons")
async def get_persons():
    """Get all known persons."""
    if state.recognizer is None:
        return {"persons": []}
    
    persons = state.recognizer.get_all_persons()
    result = []
    for name, identity in persons.items():
        result.append({
            "name": name,
            "images": len(identity.image_paths),
            "times_recognized": identity.times_recognized,
            "is_flagged": identity.is_flagged,
            "flag_reason": identity.flag_reason,
            "last_seen": identity.last_seen
        })
    return {"persons": result}


@app.post("/api/persons/register")
async def register_person(name: str = Form(...), file: UploadFile = File(...)):
    """Register a new person with a face photo."""
    if state.recognizer is None:
        raise HTTPException(status_code=503, detail="Recognizer not initialized")
    
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    success = state.recognizer.register_face(name, img)
    if success:
        return {"status": "registered", "name": name}
    raise HTTPException(status_code=500, detail="Registration failed")


@app.post("/api/persons/{name}/flag")
async def flag_person(name: str, reason: str = "Suspicious behavior"):
    """Flag a person as suspicious."""
    if state.recognizer:
        state.recognizer.flag_person(name, reason)
        return {"status": "flagged", "name": name}
    raise HTTPException(status_code=503, detail="Recognizer not initialized")


# ============================================================
# REST API — Evidence
# ============================================================
@app.get("/api/evidence")
async def get_evidence(count: int = 20):
    """Get recent evidence clips."""
    if state.evidence_recorder is None:
        return {"clips": []}
    
    clips = state.evidence_recorder.get_clips(count)
    return {"clips": [c.to_dict() for c in clips]}


@app.get("/api/evidence/{clip_id}/video")
async def get_evidence_video(clip_id: str):
    """Stream an evidence video file."""
    if state.evidence_recorder is None:
        raise HTTPException(status_code=503)
    
    for clip in state.evidence_recorder.get_clips(100):
        if clip.clip_id == clip_id:
            if os.path.exists(clip.video_path):
                return FileResponse(clip.video_path, media_type="video/mp4")
    
    raise HTTPException(status_code=404, detail="Clip not found")


@app.get("/api/evidence/{clip_id}/thumbnail")
async def get_evidence_thumbnail(clip_id: str):
    """Get evidence clip thumbnail."""
    if state.evidence_recorder is None:
        raise HTTPException(status_code=503)
    
    for clip in state.evidence_recorder.get_clips(100):
        if clip.clip_id == clip_id:
            if os.path.exists(clip.thumbnail_path):
                return FileResponse(clip.thumbnail_path, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="Thumbnail not found")


# ============================================================
# REST API — System
# ============================================================
@app.get("/api/system/status")
async def system_status():
    """Get system status."""
    return {
        "is_running": state.is_running,
        "fps": round(state.fps, 1),
        "frame_count": state.frame_count,
        "persons_tracked": state.persons_count,
        "active_connections": len(active_connections)
    }


@app.get("/api/zones")
async def get_zones():
    """Get configured zones."""
    if state.zone_manager is None:
        return {"zones": []}
    
    zones = []
    for zone in state.zone_manager.zones:
        zones.append({
            "name": zone.name,
            "type": zone.zone_type,
            "points": zone.points,
            "color": zone.color
        })
    return {"zones": zones}


# ============================================================
# Dashboard HTML
# ============================================================
@app.get("/")
async def dashboard():
    """Serve the main dashboard page."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return HTMLResponse("<h1>Dashboard files not found</h1>")


# ============================================================
# Server Launch
# ============================================================
def start_dashboard(system_state: "SystemState" = None):
    """Start the dashboard server (called from main.py)."""
    global state
    if system_state:
        state = system_state
    
    uvicorn.run(
        app,
        host=config.dashboard.host,
        port=config.dashboard.port,
        log_level="warning"
    )


if __name__ == "__main__":
    print("[Dashboard] Starting standalone dashboard...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

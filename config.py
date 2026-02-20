"""
Central Configuration for AI Behavior Detection System
All tuneable parameters in one place.
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

# ============================================================
# VIDEO SOURCE
# ============================================================
@dataclass
class VideoConfig:
    source: str = "0"                    # "0" = webcam, RTSP URL, or file path
    width: int = 1280
    height: int = 720
    fps: int = 30
    
# ============================================================
# PERSON DETECTION (YOLOv8)
# ============================================================
@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"       # yolov8n (fast) / yolov8s (balanced) / yolov8m (accurate)
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    person_class_id: int = 0             # COCO class 0 = person
    device: str = "auto"                 # "auto", "cpu", "cuda", "mps"
    img_size: int = 640

# ============================================================
# PERSON TRACKING (DeepSORT)
# ============================================================
@dataclass
class TrackingConfig:
    max_age: int = 70                    # Frames before track is deleted
    n_init: int = 3                      # Detections before track is confirmed
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.3
    nn_budget: int = 100                 # Max appearance features per track
    trajectory_length: int = 50          # Store last N positions

# ============================================================
# FACE RECOGNITION (DeepFace)
# ============================================================
@dataclass
class RecognitionConfig:
    model_name: str = "ArcFace"          # ArcFace, Facenet, VGG-Face
    detector_backend: str = "opencv"     # opencv, retinaface, mtcnn
    distance_metric: str = "cosine"
    threshold: float = 0.55              # Match threshold (lower = stricter)
    known_faces_dir: str = "known_faces"
    recognition_interval: int = 15       # Run recognition every N frames (performance)
    min_face_size: int = 40              # Minimum face pixel size to attempt recognition

# ============================================================
# POSE ESTIMATION (MediaPipe)
# ============================================================
@dataclass
class PoseConfig:
    model_complexity: int = 1            # 0=lite, 1=full, 2=heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False
    num_landmarks: int = 33

# ============================================================
# ACTION CLASSIFICATION (LSTM)
# ============================================================
@dataclass
class ActionConfig:
    sequence_length: int = 30            # Frames in sliding window
    num_features: int = 132              # 33 landmarks * 4 (x,y,z,visibility)
    hidden_size: int = 128
    num_layers: int = 2
    num_classes: int = 9
    action_labels: List[str] = field(default_factory=lambda: [
        "standing",
        "walking",
        "bending",
        "reaching_up",
        "reaching_down",
        "carrying_object",
        "hiding_object",
        "running",
        "loitering"
    ])
    model_path: str = "models/action_lstm.pth"
    confidence_threshold: float = 0.6

# ============================================================
# SUSPICIOUS BEHAVIOR DETECTION
# ============================================================
@dataclass
class BehaviorConfig:
    # --- Signal Weights (must sum to 1.0) ---
    weight_action: float = 0.35
    weight_trajectory: float = 0.25
    weight_pose: float = 0.20
    weight_zone: float = 0.15
    weight_time: float = 0.05
    
    # --- Thresholds ---
    alert_threshold_low: float = 40.0      # Yellow alert
    alert_threshold_medium: float = 65.0   # Orange alert
    alert_threshold_high: float = 85.0     # Red alert
    
    # --- Temporal Smoothing ---
    smoothing_window: int = 15             # Frames to average over
    persistence_required: int = 10         # Frames suspicion must persist
    decay_rate: float = 0.95               # Score decay per frame when normal
    
    # --- Suspicious Actions ---
    suspicious_actions: List[str] = field(default_factory=lambda: [
        "hiding_object",
        "carrying_object",
        "bending",
        "reaching_down"
    ])
    
    # --- Loitering ---
    loitering_time_threshold: float = 30.0   # Seconds in same area = loitering
    loitering_radius: float = 50.0           # Pixel radius for "same area"

# ============================================================
# ZONE MANAGEMENT
# ============================================================
@dataclass
class ZoneConfig:
    # Zones defined as list of (name, type, polygon_points)
    # polygon_points are normalized [0,1] coordinates
    zones: List[dict] = field(default_factory=lambda: [
        {
            "name": "Main Work Area",
            "type": "normal",
            "points": [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]
        },
        {
            "name": "Storage Room",
            "type": "restricted",
            "points": [(0.0, 0.0), (0.15, 0.0), (0.15, 0.3), (0.0, 0.3)]
        },
        {
            "name": "Exit Area",
            "type": "exit",
            "points": [(0.85, 0.7), (1.0, 0.7), (1.0, 1.0), (0.85, 1.0)]
        }
    ])

# ============================================================
# ALERT SYSTEM
# ============================================================
@dataclass
class AlertConfig:
    cooldown_seconds: float = 30.0       # Min time between alerts for same person
    sound_enabled: bool = True
    email_enabled: bool = False
    email_to: str = ""
    email_from: str = ""
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    log_file: str = "alerts.log"

# ============================================================
# EVIDENCE RECORDING
# ============================================================
@dataclass
class EvidenceConfig:
    pre_buffer_seconds: float = 10.0     # Seconds before alert to save
    post_buffer_seconds: float = 10.0    # Seconds after alert to save
    output_dir: str = "evidence"
    codec: str = "mp4v"
    max_clips: int = 1000                # Max stored clips before cleanup

# ============================================================
# DASHBOARD
# ============================================================
@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    stream_quality: int = 70             # JPEG quality (1-100)
    max_connections: int = 10

# ============================================================
# PREPROCESSING
# ============================================================
@dataclass
class PreprocessConfig:
    enable_clahe: bool = True            # Adaptive histogram equalization
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    enable_gamma: bool = True
    gamma_value: float = 1.0             # Auto-adjusted at runtime
    target_brightness: int = 127

# ============================================================
# MASTER CONFIG
# ============================================================
@dataclass
class SystemConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    zones: ZoneConfig = field(default_factory=ZoneConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


# Global config instance
config = SystemConfig()

"""
Pose Estimator — MediaPipe-based skeleton extraction.
Compatible with MediaPipe 0.9.x (solutions API) AND 0.10.x+.
Extracts 33 body landmarks per person for action classification.
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import math

from config import PoseConfig


# ============================================================
# MediaPipe Compatibility Layer
# MediaPipe 0.10+ removed mp.solutions — detect which API is available
# ============================================================
_mp_pose = None
_mp_drawing = None
_USE_LEGACY = False

try:
    import mediapipe as mp

    # Try legacy solutions API first (0.9.x)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        _mp_pose = mp.solutions.pose
        _mp_drawing = mp.solutions.drawing_utils
        _USE_LEGACY = True
        print("[PoseEstimator] Using MediaPipe solutions API (legacy)")
    else:
        # MediaPipe 0.10+ — use the Tasks API
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        _mp_pose = mp_vision          # store module
        _USE_LEGACY = False
        print("[PoseEstimator] Using MediaPipe Tasks API (0.10+)")

except ImportError as e:
    print(f"[PoseEstimator] WARNING: MediaPipe not available: {e}")


# ============================================================
# MediaPipe landmark indices (same in both APIs)
# ============================================================
LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# Key joint indices
IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}


# ============================================================
# PoseResult Dataclass
# ============================================================
@dataclass
class PoseResult:
    """Pose estimation result for one person."""
    track_id: int
    landmarks: np.ndarray            # (33, 3) — x, y, visibility (pixel coords)
    normalized_landmarks: np.ndarray # (33, 3) — x, y, visibility (0-1 within bbox)
    bbox: Tuple[int, int, int, int]  # Source bounding box

    # ---- Geometry helpers ----
    def landmark(self, name: str) -> np.ndarray:
        return self.landmarks[IDX[name]]

    def angle(self, a: str, b: str, c: str) -> float:
        """Calculate angle at joint b (degrees)."""
        A = self.landmarks[IDX[a]][:2]
        B = self.landmarks[IDX[b]][:2]
        C = self.landmarks[IDX[c]][:2]
        ba = A - B
        bc = C - B
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

    def is_bending(self, threshold: float = 130.0) -> bool:
        """True if person is significantly bending forward."""
        try:
            hip_angle = self.angle('left_shoulder', 'left_hip', 'left_knee')
            return hip_angle < threshold
        except Exception:
            return False

    def is_hand_near_torso(self, threshold: float = 0.25) -> bool:
        """True if hand is suspiciously close to torso (concealment gesture)."""
        try:
            lw = self.normalized_landmarks[IDX['left_wrist']]
            rw = self.normalized_landmarks[IDX['right_wrist']]
            lh = self.normalized_landmarks[IDX['left_hip']]
            rh = self.normalized_landmarks[IDX['right_hip']]
            torso_center = (lh + rh) / 2.0
            dist_l = np.linalg.norm(lw[:2] - torso_center[:2])
            dist_r = np.linalg.norm(rw[:2] - torso_center[:2])
            return min(dist_l, dist_r) < threshold
        except Exception:
            return False

    @property
    def hands_below_waist(self) -> bool:
        """True if either hand is below the waist line."""
        try:
            lw_y = self.normalized_landmarks[IDX['left_wrist'], 1]
            rw_y = self.normalized_landmarks[IDX['right_wrist'], 1]
            lh_y = self.normalized_landmarks[IDX['left_hip'], 1]
            rh_y = self.normalized_landmarks[IDX['right_hip'], 1]
            waist_y = (lh_y + rh_y) / 2.0
            # In normalized coords, higher y = lower in image
            return lw_y > waist_y + 0.05 or rw_y > waist_y + 0.05
        except Exception:
            return False

    @property
    def hand_to_bag_gesture(self) -> bool:
        """True if hand is near hip/bag area while bending — concealment gesture."""
        try:
            return self.is_bending() and self.is_hand_near_torso(threshold=0.2)
        except Exception:
            return False

    def get_angle(self, idx_a: int, idx_b: int, idx_c: int) -> float:
        """Calculate angle at joint idx_b using landmark indices."""
        A = self.landmarks[idx_a][:2]
        B = self.landmarks[idx_b][:2]
        C = self.landmarks[idx_c][:2]
        ba = A - B
        bc = C - B
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

    def to_feature_vector(self) -> np.ndarray:
        """Flatten normalized landmarks to 1-D feature vector (99 values)."""
        return self.normalized_landmarks.flatten()


# ============================================================
# PoseEstimator
# ============================================================
class PoseEstimator:
    """
    Extracts human pose landmarks from cropped person bounding boxes.
    Supports MediaPipe 0.9.x (solutions) and 0.10.x+ (Tasks API).
    Falls back to a dummy result if MediaPipe is unavailable.
    """

    def __init__(self, cfg: PoseConfig = None):
        self.cfg = cfg or PoseConfig()
        self._pose_sessions = {}   # track_id -> mediapipe Pose instance
        self._available = _mp_pose is not None

        if not self._available:
            print("[PoseEstimator] WARNING: Running without pose estimation")
            return

        if _USE_LEGACY:
            # Legacy: create a single shared Pose instance
            self.pose = _mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.cfg.model_complexity,
                smooth_landmarks=True,
                min_detection_confidence=self.cfg.min_detection_confidence,
                min_tracking_confidence=self.cfg.min_tracking_confidence,
            )
            print(f"[PoseEstimator] MediaPipe Pose loaded "
                  f"(complexity={self.cfg.model_complexity})")
        else:
            # New Tasks API
            self._init_tasks_api()

    def _init_tasks_api(self):
        """Initialise the Tasks API pose landmarker (0.10+)."""
        import urllib.request
        import os

        model_path = os.path.join("models", "pose_landmarker_lite.task")
        if not os.path.exists(model_path):
            os.makedirs("models", exist_ok=True)
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/1/"
                   "pose_landmarker_lite.task")
            print(f"[PoseEstimator] Downloading pose model...")
            urllib.request.urlretrieve(url, model_path)

        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )
        self.pose = mp_vision.PoseLandmarker.create_from_options(options)
        print("[PoseEstimator] Tasks API PoseLandmarker loaded")

    # ----------------------------------------------------------------
    def estimate(self, frame: np.ndarray,
                 bbox: Tuple[int, int, int, int],
                 track_id: int = 0) -> Optional[PoseResult]:
        """
        Run pose estimation on the person crop defined by bbox.

        Returns PoseResult or None if no pose detected.
        """
        if not self._available:
            return None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        # Clamp bbox to frame
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None

        crop = frame[y1:y2, x1:x2]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        ch, cw = crop.shape[:2]

        try:
            if _USE_LEGACY:
                return self._estimate_legacy(
                    rgb_crop, bbox, track_id, x1, y1, cw, ch
                )
            else:
                return self._estimate_tasks(
                    rgb_crop, bbox, track_id, x1, y1, cw, ch
                )
        except Exception as e:
            return None

    def _estimate_legacy(self, rgb_crop, bbox, track_id, ox, oy, cw, ch):
        """Use mp.solutions.pose (MediaPipe ≤ 0.9.x)."""
        results = self.pose.process(rgb_crop)
        if not results.pose_landmarks:
            return None

        landmarks = np.zeros((33, 3))
        norm_landmarks = np.zeros((33, 3))

        for i, lm in enumerate(results.pose_landmarks.landmark):
            # Pixel coords in full frame
            px = int(lm.x * cw) + ox
            py = int(lm.y * ch) + oy
            landmarks[i] = [px, py, lm.visibility]
            norm_landmarks[i] = [lm.x, lm.y, lm.visibility]

        return PoseResult(
            track_id=track_id,
            landmarks=landmarks,
            normalized_landmarks=norm_landmarks,
            bbox=bbox,
        )

    def _estimate_tasks(self, rgb_crop, bbox, track_id, ox, oy, cw, ch):
        """Use MediaPipe Tasks API (MediaPipe ≥ 0.10.x)."""
        import mediapipe as mp
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_crop
        )
        detection_result = self.pose.detect(mp_image)
        if not detection_result.pose_landmarks:
            return None

        pose_lms = detection_result.pose_landmarks[0]
        landmarks = np.zeros((33, 3))
        norm_landmarks = np.zeros((33, 3))

        for i, lm in enumerate(pose_lms):
            px = int(lm.x * cw) + ox
            py = int(lm.y * ch) + oy
            vis = getattr(lm, 'visibility', 1.0) or 1.0
            landmarks[i] = [px, py, vis]
            norm_landmarks[i] = [lm.x, lm.y, vis]

        return PoseResult(
            track_id=track_id,
            landmarks=landmarks,
            normalized_landmarks=norm_landmarks,
            bbox=bbox,
        )

    # ----------------------------------------------------------------
    def draw_skeleton(self, frame: np.ndarray,
                      pose: PoseResult,
                      color: Tuple = (0, 255, 0)) -> np.ndarray:
        """Draw pose skeleton on frame."""
        if pose is None:
            return frame

        # Connections to draw
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
        ]

        lms = pose.landmarks
        for a, b in connections:
            ia, ib = IDX[a], IDX[b]
            if lms[ia][2] > 0.3 and lms[ib][2] > 0.3:
                pt1 = (int(lms[ia][0]), int(lms[ia][1]))
                pt2 = (int(lms[ib][0]), int(lms[ib][1]))
                cv2.line(frame, pt1, pt2, color, 2)

        # Draw joints
        for i in range(33):
            if lms[i][2] > 0.3:
                cv2.circle(frame, (int(lms[i][0]), int(lms[i][1])),
                           3, color, -1)

        return frame

    def release(self):
        """Release resources."""
        if self._available and hasattr(self, 'pose'):
            try:
                self.pose.close()
            except Exception:
                pass

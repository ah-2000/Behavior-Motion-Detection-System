"""
Multi-Person Tracker — DeepSORT with trajectory tracking.
Assigns persistent IDs, tracks movement paths, handles occlusions.
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None

from config import TrackingConfig
from modules.detector import Detection


@dataclass
class TrackedPerson:
    """Represents a tracked individual with history."""
    track_id: int
    bbox: Tuple[int, int, int, int]          # (x1, y1, x2, y2)
    confidence: float
    is_confirmed: bool
    age: int                                  # Frames since first detection
    person_name: str = "Unknown"              # Set by recognizer
    trajectory: deque = field(default_factory=lambda: deque(maxlen=50))
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    def speed(self) -> float:
        """Calculate pixel speed from last 2 trajectory points."""
        if len(self.trajectory) < 2:
            return 0.0
        p1 = self.trajectory[-2]
        p2 = self.trajectory[-1]
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def direction(self) -> Optional[Tuple[float, float]]:
        """Get movement direction vector (normalized)."""
        if len(self.trajectory) < 2:
            return None
        p1 = np.array(self.trajectory[-2])
        p2 = np.array(self.trajectory[-1])
        diff = p2 - p1
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            return (0.0, 0.0)
        return tuple(diff / norm)
    
    def displacement_from_start(self) -> float:
        """Total displacement from first position."""
        if len(self.trajectory) < 2:
            return 0.0
        start = np.array(self.trajectory[0])
        current = np.array(self.trajectory[-1])
        return float(np.linalg.norm(current - start))


class PersonTracker:
    """DeepSORT-based multi-person tracker with trajectory analysis."""
    
    def __init__(self, cfg: TrackingConfig = None):
        self.cfg = cfg or TrackingConfig()
        
        if DeepSort is None:
            raise ImportError(
                "deep_sort_realtime not installed. Run: pip install deep-sort-realtime"
            )
        
        self.tracker = DeepSort(
            max_age=self.cfg.max_age,
            n_init=self.cfg.n_init,
            max_iou_distance=self.cfg.max_iou_distance,
            max_cosine_distance=self.cfg.max_cosine_distance,
            nn_budget=self.cfg.nn_budget,
            embedder=None,         # Disable built-in embedder (avoids pkg_resources error)
            embedder_gpu=False,
        )
        
        # Persistent storage: track_id -> TrackedPerson
        self._persons: Dict[int, TrackedPerson] = {}
        self._trajectory_maxlen = self.cfg.trajectory_length
        
        print(f"[Tracker] DeepSORT initialized (max_age={self.cfg.max_age})")
    
    def _compute_embedding(self, frame: np.ndarray,
                            bbox: Tuple[int, int, int, int],
                            fw: int, fh: int) -> np.ndarray:
        """Compute a lightweight color-histogram embedding for a person crop."""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return np.zeros(128, dtype=np.float32)
        
        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (32, 64))
        
        # 3-channel histogram (16 bins each = 48 values)
        hist = []
        for ch in range(3):
            h = cv2.calcHist([crop_resized], [ch], None, [16], [0, 256])
            hist.append(h.flatten())
        embedding = np.concatenate(hist).astype(np.float32)
        
        # Pad to 128 dims (DeepSORT default expects 128)
        padded = np.zeros(128, dtype=np.float32)
        padded[:len(embedding)] = embedding
        norm = np.linalg.norm(padded) + 1e-6
        return padded / norm

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[TrackedPerson]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects from the detector
            frame: Current frame (for appearance features)
            
        Returns:
            List of TrackedPerson objects with updated positions
        """
        # Convert detections to DeepSORT format with embeddings.
        # When embedder=None, DeepSORT requires embeddings via the `embeds` kwarg.
        # Detections stay as 3-tuples: ([left,top,w,h], confidence, class)
        fh, fw = frame.shape[:2]
        raw_detections = []
        embeds = []
        for det in detections:
            ltwh = det.to_ltwh()
            raw_detections.append(
                (list(ltwh), det.confidence, "person")
            )
            embeds.append(self._compute_embedding(frame, det.bbox, fw, fh))
        
        # Run DeepSORT update
        tracks = self.tracker.update_tracks(raw_detections, embeds=embeds, frame=frame)
        
        # Process tracks
        active_persons = []
        active_ids = set()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            active_ids.add(track_id)
            
            # Get bounding box
            ltrb = track.to_ltrb()
            bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))
            
            # Update or create TrackedPerson
            if track_id in self._persons:
                person = self._persons[track_id]
                person.bbox = bbox
                person.confidence = track.det_conf if track.det_conf else 0.0
                person.is_confirmed = True
                person.age += 1
            else:
                person = TrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=track.det_conf if track.det_conf else 0.0,
                    is_confirmed=True,
                    age=1,
                    trajectory=deque(maxlen=self._trajectory_maxlen)
                )
                self._persons[track_id] = person
            
            # Update trajectory
            person.trajectory.append(person.center)
            
            active_persons.append(person)
        
        # Cleanup stale tracks
        stale_ids = [tid for tid in self._persons if tid not in active_ids]
        for tid in stale_ids:
            if self._persons[tid].age > self.cfg.max_age * 2:
                del self._persons[tid]
        
        return active_persons
    
    def get_person(self, track_id: int) -> Optional[TrackedPerson]:
        """Get a specific tracked person by ID."""
        return self._persons.get(track_id)
    
    def get_all_persons(self) -> Dict[int, TrackedPerson]:
        """Get all tracked persons (including inactive)."""
        return self._persons.copy()
    
    def set_person_name(self, track_id: int, name: str):
        """Associate a name with a tracked person (from face recognition)."""
        if track_id in self._persons:
            self._persons[track_id].person_name = name

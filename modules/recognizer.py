"""
Person Recognizer — DeepFace-based face recognition.
Maintains a face database, encodes faces, matches against known individuals.
"""
import os
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import threading
import time

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

from config import RecognitionConfig


@dataclass
class PersonIdentity:
    """Stored identity of a known person."""
    name: str
    face_encodings: List[np.ndarray] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    times_recognized: int = 0
    is_flagged: bool = False
    flag_reason: str = ""


class PersonRecognizer:
    """
    DeepFace-based person recognition system.
    Runs recognition at intervals to balance accuracy vs performance.
    """
    
    def __init__(self, cfg: RecognitionConfig = None):
        self.cfg = cfg or RecognitionConfig()
        
        if DeepFace is None:
            raise ImportError(
                "deepface not installed. Run: pip install deepface"
            )
        
        # Known faces database: name -> PersonIdentity
        self._database: Dict[str, PersonIdentity] = {}
        
        # Track ID -> recognized name cache
        self._track_cache: Dict[int, str] = {}
        
        # Frame counter for recognition interval
        self._frame_count = 0
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Ensure known faces directory exists
        os.makedirs(self.cfg.known_faces_dir, exist_ok=True)
        
        # Load existing face database
        self._load_database()
        
        print(f"[Recognizer] DeepFace initialized ({self.cfg.model_name}), "
              f"{len(self._database)} known persons")
    
    def _load_database(self):
        """Load face images from known_faces/ directory structure."""
        if not os.path.exists(self.cfg.known_faces_dir):
            return
        
        for person_name in os.listdir(self.cfg.known_faces_dir):
            person_dir = os.path.join(self.cfg.known_faces_dir, person_name)
            if not os.path.isdir(person_dir):
                continue
            
            identity = PersonIdentity(
                name=person_name,
                first_seen=time.time()
            )
            
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(person_dir, img_file)
                    identity.image_paths.append(img_path)
            
            if identity.image_paths:
                self._database[person_name] = identity
                print(f"  Loaded {person_name}: {len(identity.image_paths)} images")
    
    def recognize(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                  track_id: int) -> str:
        """
        Recognize a person from their face within a bounding box.
        
        Args:
            frame: Full frame
            bbox: Person bounding box (x1, y1, x2, y2)
            track_id: Tracker ID for caching results
            
        Returns:
            Person name or "Unknown"
        """
        self._frame_count += 1
        
        # Return cached result if not time for re-recognition
        if (self._frame_count % self.cfg.recognition_interval != 0
                and track_id in self._track_cache):
            return self._track_cache[track_id]
        
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Ensure bbox is within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Crop the upper portion for face (top 40% of person bbox)
        face_y2 = y1 + int((y2 - y1) * 0.4)
        face_region = frame[y1:face_y2, x1:x2]
        
        if face_region.size == 0:
            return self._track_cache.get(track_id, "Unknown")
        
        # Check minimum face size
        if face_region.shape[0] < self.cfg.min_face_size or \
           face_region.shape[1] < self.cfg.min_face_size:
            return self._track_cache.get(track_id, "Unknown")
        
        # Try to recognize against database
        name = self._match_face(face_region)
        
        # Cache the result
        with self._lock:
            self._track_cache[track_id] = name
        
        # Update identity stats
        if name != "Unknown" and name in self._database:
            identity = self._database[name]
            identity.last_seen = time.time()
            identity.times_recognized += 1
        
        return name
    
    def _match_face(self, face_img: np.ndarray) -> str:
        """Match a face image against the known database."""
        if not self._database:
            return "Unknown"
        
        try:
            # Search against all known faces 
            best_match = "Unknown"
            best_score = float('inf')
            
            for person_name, identity in self._database.items():
                for img_path in identity.image_paths:
                    try:
                        result = DeepFace.verify(
                            img1_path=face_img,
                            img2_path=img_path,
                            model_name=self.cfg.model_name,
                            detector_backend=self.cfg.detector_backend,
                            distance_metric=self.cfg.distance_metric,
                            enforce_detection=False
                        )
                        
                        if result["verified"] and result["distance"] < best_score:
                            best_score = result["distance"]
                            best_match = person_name
                    except Exception:
                        continue
            
            return best_match
            
        except Exception as e:
            return "Unknown"
    
    def register_face(self, name: str, face_image: np.ndarray) -> bool:
        """
        Register a new face to the database.
        
        Args:
            name: Person's name
            face_image: Face image (BGR numpy array)
            
        Returns:
            True if registration successful
        """
        try:
            # Create person directory
            person_dir = os.path.join(self.cfg.known_faces_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            
            # Save the face image
            existing = len([f for f in os.listdir(person_dir) if f.endswith('.jpg')])
            img_path = os.path.join(person_dir, f"{name}_{existing + 1}.jpg")
            cv2.imwrite(img_path, face_image)
            
            # Update database
            with self._lock:
                if name not in self._database:
                    self._database[name] = PersonIdentity(
                        name=name,
                        first_seen=time.time()
                    )
                self._database[name].image_paths.append(img_path)
            
            print(f"[Recognizer] Registered face for: {name}")
            return True
            
        except Exception as e:
            print(f"[Recognizer] Registration failed: {e}")
            return False
    
    def flag_person(self, name: str, reason: str = "Suspicious behavior"):
        """Flag a person as suspicious."""
        with self._lock:
            if name in self._database:
                self._database[name].is_flagged = True
                self._database[name].flag_reason = reason
                print(f"[Recognizer] FLAGGED: {name} — {reason}")
    
    def is_flagged(self, name: str) -> bool:
        """Check if a person is flagged."""
        if name in self._database:
            return self._database[name].is_flagged
        return False
    
    def get_all_persons(self) -> Dict[str, PersonIdentity]:
        """Get the full person database."""
        return self._database.copy()
    
    def clear_cache(self):
        """Clear the track-to-name cache."""
        with self._lock:
            self._track_cache.clear()

"""
Evidence Recorder — Pre-buffer video clip extraction on alerts.
Saves annotated video evidence with metadata for review.
"""
import os
import cv2
import json
import time
import numpy as np
import threading
from typing import Optional, Dict, Deque
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime

from config import EvidenceConfig


@dataclass
class EvidenceClip:
    """Metadata for a recorded evidence clip."""
    clip_id: str
    alert_id: str
    track_id: int
    person_name: str
    video_path: str
    thumbnail_path: str
    start_time: float
    end_time: float
    duration: float
    behavior_score: float
    reasons: list
    
    def to_dict(self):
        return asdict(self)


class EvidenceRecorder:
    """
    Records video evidence on suspicious behavior alerts.
    Maintains a rolling pre-buffer so clips include footage BEFORE the alert.
    """
    
    def __init__(self, cfg: EvidenceConfig = None, fps: int = 30):
        self.cfg = cfg or EvidenceConfig()
        self.fps = fps
        
        # Rolling pre-buffer (circular buffer of annotated frames)
        buffer_size = int(self.cfg.pre_buffer_seconds * fps)
        self._frame_buffer: Deque[np.ndarray] = deque(maxlen=max(buffer_size, 30))
        
        # Active recordings: alert_id -> recording state
        self._active_recordings: Dict[str, dict] = {}
        
        # Completed clips
        self._clips: list = []
        self._clip_counter = 0
        
        # Ensure output directory
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        print(f"[EvidenceRecorder] Pre-buffer: {self.cfg.pre_buffer_seconds}s, "
              f"Post-buffer: {self.cfg.post_buffer_seconds}s")
    
    def feed_frame(self, frame: np.ndarray):
        """Add a frame to the rolling pre-buffer. Call every frame."""
        self._frame_buffer.append(frame.copy())
        
        # Update any active recordings
        with self._lock:
            completed = []
            for alert_id, rec in self._active_recordings.items():
                rec["frames"].append(frame.copy())
                rec["frame_count"] += 1
                
                # Check if post-buffer is complete
                elapsed = time.time() - rec["alert_time"]
                if elapsed >= self.cfg.post_buffer_seconds:
                    completed.append(alert_id)
            
            for alert_id in completed:
                self._finalize_recording(alert_id)
    
    def start_recording(self, alert_id: str, track_id: int,
                        person_name: str, behavior_score: float,
                        reasons: list):
        """
        Start recording evidence for an alert.
        Includes pre-buffered frames + new frames until post-buffer expires.
        """
        with self._lock:
            if alert_id in self._active_recordings:
                return  # Already recording for this alert
            
            # Capture pre-buffer frames
            pre_frames = list(self._frame_buffer)
            
            self._active_recordings[alert_id] = {
                "alert_id": alert_id,
                "track_id": track_id,
                "person_name": person_name,
                "behavior_score": behavior_score,
                "reasons": reasons,
                "frames": pre_frames,
                "frame_count": len(pre_frames),
                "start_time": time.time() - self.cfg.pre_buffer_seconds,
                "alert_time": time.time()
            }
            
            print(f"[EvidenceRecorder] Recording started for {alert_id} "
                  f"(pre-buffer: {len(pre_frames)} frames)")
    
    def _finalize_recording(self, alert_id: str):
        """Save recorded frames to a video file."""
        rec = self._active_recordings.pop(alert_id, None)
        if rec is None or not rec["frames"]:
            return
        
        self._clip_counter += 1
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evidence_{self._clip_counter:04d}_{timestamp_str}.mp4"
        video_path = os.path.join(self.cfg.output_dir, filename)
        
        # Get frame dimensions
        sample_frame = rec["frames"][0]
        h, w = sample_frame.shape[:2]
        
        # Write video
        fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec)
        writer = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
        
        for frame in rec["frames"]:
            if frame.shape[:2] == (h, w):
                writer.write(frame)
        
        writer.release()
        
        # Save thumbnail (frame at alert time)
        thumb_path = video_path.replace(".mp4", "_thumb.jpg")
        alert_frame_idx = min(
            int(self.cfg.pre_buffer_seconds * self.fps),
            len(rec["frames"]) - 1
        )
        cv2.imwrite(thumb_path, rec["frames"][alert_frame_idx])
        
        # Create clip metadata
        clip = EvidenceClip(
            clip_id=f"CLIP-{self._clip_counter:04d}",
            alert_id=alert_id,
            track_id=rec["track_id"],
            person_name=rec["person_name"],
            video_path=video_path,
            thumbnail_path=thumb_path,
            start_time=rec["start_time"],
            end_time=time.time(),
            duration=time.time() - rec["start_time"],
            behavior_score=rec["behavior_score"],
            reasons=rec["reasons"]
        )
        
        self._clips.append(clip)
        
        # Save metadata JSON alongside video
        meta_path = video_path.replace(".mp4", "_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(clip.to_dict(), f, indent=2)
        
        print(f"[EvidenceRecorder] Saved: {video_path} "
              f"({len(rec['frames'])} frames, {clip.duration:.1f}s)")
        
        # Cleanup old clips if over limit
        self._cleanup_old_clips()
        
        return clip
    
    def _cleanup_old_clips(self):
        """Remove oldest clips if over the maximum limit."""
        while len(self._clips) > self.cfg.max_clips:
            old_clip = self._clips.pop(0)
            try:
                if os.path.exists(old_clip.video_path):
                    os.remove(old_clip.video_path)
                if os.path.exists(old_clip.thumbnail_path):
                    os.remove(old_clip.thumbnail_path)
                meta_path = old_clip.video_path.replace(".mp4", "_meta.json")
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            except Exception:
                pass
    
    def get_clips(self, count: int = 20) -> list:
        """Get recent evidence clips."""
        return self._clips[-count:]
    
    def get_clip_for_alert(self, alert_id: str) -> Optional[EvidenceClip]:
        """Get evidence clip associated with an alert."""
        for clip in self._clips:
            if clip.alert_id == alert_id:
                return clip
        return None
    
    def is_recording(self) -> bool:
        """Check if any recording is in progress."""
        return len(self._active_recordings) > 0

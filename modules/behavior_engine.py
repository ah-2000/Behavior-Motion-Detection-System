"""
Suspicious Behavior Detection Engine — Multi-signal consensus scoring.
Combines action, trajectory, pose, zone, and time signals to produce
a suspicion score with temporal smoothing and false-positive reduction.
"""
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field

from config import BehaviorConfig
from modules.tracker import TrackedPerson
from modules.action_classifier import ActionPrediction
from modules.pose_estimator import PoseResult
from modules.zone_manager import ZoneEvent


@dataclass
class BehaviorScore:
    """Composite suspicion score for a tracked person."""
    track_id: int
    total_score: float                        # 0–100
    action_score: float = 0.0
    trajectory_score: float = 0.0
    pose_score: float = 0.0
    zone_score: float = 0.0
    time_score: float = 0.0
    alert_level: str = "none"                 # "none", "low", "medium", "high"
    reasons: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    person_name: str = "Unknown"
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class BehaviorEngine:
    """
    Multi-signal suspicious behavior detector.
    
    Combines 5 signal types with configurable weights:
    1. Action Classification (35%) — What the person is doing
    2. Trajectory Anomaly (25%) — Where/how they're moving
    3. Pose Anomaly (20%) — Unnatural body positions
    4. Zone Violations (15%) — Being in restricted areas
    5. Time Correlation (5%) — Actions at unusual times
    
    Uses temporal smoothing and multi-signal consensus to reduce false positives.
    """
    
    def __init__(self, cfg: BehaviorConfig = None):
        self.cfg = cfg or BehaviorConfig()
        
        # Per-person score history for temporal smoothing
        self._score_history: Dict[int, deque] = {}
        
        # Per-person persistence counter (consecutive high-score frames)
        self._persistence: Dict[int, int] = {}
        
        # Per-person last alert time (for cooldown)
        self._last_alert: Dict[int, float] = {}
        
        # Per-person loitering tracker: track_id -> (start_position, start_time)
        self._loiter_start: Dict[int, Tuple[np.ndarray, float]] = {}
        
        print(f"[BehaviorEngine] Initialized — "
              f"Thresholds: low={self.cfg.alert_threshold_low}, "
              f"med={self.cfg.alert_threshold_medium}, "
              f"high={self.cfg.alert_threshold_high}")
    
    def analyze(self,
                person: TrackedPerson,
                action: Optional[ActionPrediction] = None,
                pose: Optional[PoseResult] = None,
                zone_events: Optional[List[ZoneEvent]] = None) -> BehaviorScore:
        """
        Analyze a person's behavior and produce a composite suspicion score.
        
        Args:
            person: TrackedPerson with trajectory
            action: Action classification result
            pose: Pose estimation result
            zone_events: Recent zone events for this person
            
        Returns:
            BehaviorScore with 0–100 composite score
        """
        track_id = person.track_id
        reasons = []
        
        # --- Signal 1: Action Classification Score ---
        action_score = self._score_action(action, reasons)
        
        # --- Signal 2: Trajectory Anomaly Score ---
        trajectory_score = self._score_trajectory(person, reasons)
        
        # --- Signal 3: Pose Anomaly Score ---
        pose_score = self._score_pose(pose, reasons)
        
        # --- Signal 4: Zone Violation Score ---
        zone_score = self._score_zones(zone_events, reasons)
        
        # --- Signal 5: Time Correlation Score ---
        time_score = self._score_time(reasons)
        
        # --- Weighted Composite Score ---
        raw_score = (
            action_score * self.cfg.weight_action +
            trajectory_score * self.cfg.weight_trajectory +
            pose_score * self.cfg.weight_pose +
            zone_score * self.cfg.weight_zone +
            time_score * self.cfg.weight_time
        )
        
        # --- Multi-signal consensus bonus ---
        # If multiple signals are high, boost the score
        high_signals = sum(1 for s in [action_score, trajectory_score, 
                                        pose_score, zone_score] if s > 50)
        if high_signals >= 3:
            raw_score = min(100, raw_score * 1.3)
            reasons.append("Multi-signal consensus (3+ signals)")
        elif high_signals >= 2:
            raw_score = min(100, raw_score * 1.1)
        
        # --- Temporal Smoothing ---
        smoothed_score = self._smooth_score(track_id, raw_score)
        
        # --- Persistence Check ---
        alert_level = self._determine_alert_level(track_id, smoothed_score)
        
        return BehaviorScore(
            track_id=track_id,
            total_score=round(smoothed_score, 1),
            action_score=round(action_score, 1),
            trajectory_score=round(trajectory_score, 1),
            pose_score=round(pose_score, 1),
            zone_score=round(zone_score, 1),
            time_score=round(time_score, 1),
            alert_level=alert_level,
            reasons=reasons,
            person_name=person.person_name
        )
    
    # ====================================================================
    # Signal Scoring Functions
    # ====================================================================
    
    def _score_action(self, action: Optional[ActionPrediction],
                      reasons: List[str]) -> float:
        """Score based on classified action (0–100)."""
        if action is None:
            return 0.0
        
        score = 0.0
        
        # High suspicion actions
        if action.action == "hiding_object":
            score = 90.0 * action.confidence
            reasons.append(f"Hiding object detected ({action.confidence:.0%})")
        elif action.action == "carrying_object":
            score = 50.0 * action.confidence
            reasons.append(f"Carrying object ({action.confidence:.0%})")
        elif action.action == "bending":
            score = 30.0 * action.confidence
        elif action.action == "reaching_down":
            score = 35.0 * action.confidence
            reasons.append(f"Reaching down ({action.confidence:.0%})")
        elif action.action == "running":
            score = 45.0 * action.confidence
            reasons.append(f"Running detected ({action.confidence:.0%})")
        elif action.action == "loitering":
            score = 40.0 * action.confidence
            reasons.append(f"Loitering ({action.confidence:.0%})")
        else:
            score = 5.0  # Normal activity
        
        return min(100, score)
    
    def _score_trajectory(self, person: TrackedPerson,
                          reasons: List[str]) -> float:
        """Score based on movement trajectory anomalies (0–100)."""
        score = 0.0
        trajectory = list(person.trajectory)
        
        if len(trajectory) < 5:
            return 0.0
        
        # --- Loitering detection (staying in same area) ---
        current_pos = np.array(trajectory[-1])
        track_id = person.track_id
        current_time = time.time()
        
        if track_id not in self._loiter_start:
            self._loiter_start[track_id] = (current_pos, current_time)
        else:
            start_pos, start_time = self._loiter_start[track_id]
            distance = np.linalg.norm(current_pos - start_pos)
            elapsed = current_time - start_time
            
            if distance < self.cfg.loitering_radius:
                if elapsed > self.cfg.loitering_time_threshold:
                    loiter_score = min(80, 30 + (elapsed - self.cfg.loitering_time_threshold) * 2)
                    score = max(score, loiter_score)
                    reasons.append(f"Loitering for {elapsed:.0f}s")
            else:
                # Person moved significantly — reset loiter tracker
                self._loiter_start[track_id] = (current_pos, current_time)
        
        # --- Erratic movement (high direction changes) ---
        if len(trajectory) >= 10:
            directions = []
            for i in range(1, len(trajectory)):
                dx = trajectory[i][0] - trajectory[i-1][0]
                dy = trajectory[i][1] - trajectory[i-1][1]
                if abs(dx) > 1 or abs(dy) > 1:
                    directions.append(np.arctan2(dy, dx))
            
            if len(directions) >= 5:
                direction_changes = 0
                for i in range(1, len(directions)):
                    angle_diff = abs(directions[i] - directions[i-1])
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    if angle_diff > np.pi / 3:  # >60 degree change
                        direction_changes += 1
                
                change_ratio = direction_changes / len(directions)
                if change_ratio > 0.5:
                    erratic_score = min(60, change_ratio * 100)
                    score = max(score, erratic_score)
                    reasons.append(f"Erratic movement (change ratio: {change_ratio:.2f})")
        
        # --- Suspicious speed changes (sudden stop/start) ---
        speed = person.speed()
        if speed > 50:  # Pixel speed threshold
            score = max(score, 35)
            reasons.append(f"Fast movement (speed: {speed:.1f})")
        
        return min(100, score)
    
    def _score_pose(self, pose: Optional[PoseResult],
                    reasons: List[str]) -> float:
        """Score based on pose anomalies (0–100)."""
        if pose is None:
            return 0.0
        
        score = 0.0
        
        # Concealing gesture (hand-to-bag with bending)
        if pose.hand_to_bag_gesture:
            score = max(score, 80.0)
            reasons.append("Concealing gesture detected")
        
        # Bending with hands low (potential hiding)
        if pose.is_bending() and pose.hands_below_waist:
            score = max(score, 55.0)
            reasons.append("Bending with hands below waist")
        
        # Just bending
        elif pose.is_bending():
            score = max(score, 25.0)
        
        # Low visibility landmarks (possible occlusion or hiding)
        visible_landmarks = np.sum(pose.landmarks[:, 2] > 0.5)
        if visible_landmarks < 15:  # Less than half visible
            score = max(score, 30.0)
            reasons.append(f"Partially occluded ({visible_landmarks}/33 landmarks)")
        
        return min(100, score)
    
    def _score_zones(self, zone_events: Optional[List[ZoneEvent]],
                     reasons: List[str]) -> float:
        """Score based on zone violations (0–100)."""
        if not zone_events:
            return 0.0
        
        score = 0.0
        
        for event in zone_events:
            if event.zone_type == "restricted":
                if event.event_type == "enter":
                    score = max(score, 70.0)
                    reasons.append(f"Entered restricted zone: {event.zone_name}")
                elif event.event_type == "linger":
                    linger_score = min(95, 70 + event.duration * 2)
                    score = max(score, linger_score)
                    reasons.append(f"Lingering in restricted zone: {event.zone_name} "
                                 f"({event.duration:.0f}s)")
            
            elif event.zone_type == "inventory":
                if event.event_type == "linger" and event.duration > 10:
                    score = max(score, 50.0)
                    reasons.append(f"Extended time at inventory: {event.zone_name}")
            
            elif event.zone_type == "exit":
                if event.event_type == "enter":
                    score = max(score, 20.0)  # Normal to go to exit
        
        return min(100, score)
    
    def _score_time(self, reasons: List[str]) -> float:
        """Score based on time of day (0–100)."""
        current_hour = time.localtime().tm_hour
        
        # Suspicious time windows (adjust for your business hours)
        if current_hour < 6 or current_hour > 22:
            reasons.append(f"Activity outside business hours ({current_hour}:00)")
            return 80.0
        elif current_hour < 7 or current_hour > 21:
            return 30.0
        
        return 0.0
    
    # ====================================================================
    # Temporal Smoothing & Alert Logic
    # ====================================================================
    
    def _smooth_score(self, track_id: int, raw_score: float) -> float:
        """Apply temporal smoothing to reduce noise/flicker."""
        if track_id not in self._score_history:
            self._score_history[track_id] = deque(
                maxlen=self.cfg.smoothing_window
            )
        
        history = self._score_history[track_id]
        history.append(raw_score)
        
        if len(history) < 3:
            return raw_score
        
        # Weighted moving average (recent frames weighted more)
        weights = np.linspace(0.5, 1.0, len(history))
        smoothed = np.average(list(history), weights=weights)
        
        # Apply decay if score is dropping
        if smoothed < raw_score:
            smoothed = raw_score  # Rising scores take effect immediately
        
        return smoothed
    
    def _determine_alert_level(self, track_id: int,
                                score: float) -> str:
        """
        Determine alert level with persistence requirement.
        Score must persist above threshold for N frames to trigger.
        """
        # Track consecutive high-score frames
        if track_id not in self._persistence:
            self._persistence[track_id] = 0
        
        if score >= self.cfg.alert_threshold_low:
            self._persistence[track_id] += 1
        else:
            # Decay persistence slowly
            self._persistence[track_id] = max(
                0, self._persistence[track_id] - 2
            )
        
        # Require persistence before alerting
        persistent_frames = self._persistence[track_id]
        if persistent_frames < self.cfg.persistence_required:
            return "none"
        
        # Determine level
        if score >= self.cfg.alert_threshold_high:
            return "high"
        elif score >= self.cfg.alert_threshold_medium:
            return "medium"
        elif score >= self.cfg.alert_threshold_low:
            return "low"
        
        return "none"
    
    def get_score_history(self, track_id: int) -> List[float]:
        """Get recent score history for a person."""
        if track_id in self._score_history:
            return list(self._score_history[track_id])
        return []
    
    def reset_person(self, track_id: int):
        """Reset all state for a specific person."""
        self._score_history.pop(track_id, None)
        self._persistence.pop(track_id, None)
        self._last_alert.pop(track_id, None)
        self._loiter_start.pop(track_id, None)

"""
AI-Powered Behavior & Motion Detection System
=============================================
Main processing pipeline — captures video, runs detection, tracking,
recognition, pose estimation, action classification, and behavior
analysis in real-time. Streams results to the web dashboard.

Usage:
    python main.py                          # Webcam (default)
    python main.py --source video.mp4       # Video file
    python main.py --source rtsp://...      # RTSP stream
    python main.py --no-dashboard           # Run without web UI
"""
import os
import sys
import cv2
import time
import json
import base64
import asyncio
import argparse
import threading
import numpy as np
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from config import config

# ============================================================
# Import modules
# ============================================================
from modules.preprocessor import FramePreprocessor
from modules.detector import PersonDetector
from modules.tracker import PersonTracker
from modules.recognizer import PersonRecognizer
from modules.pose_estimator import PoseEstimator
from modules.action_classifier import ActionClassifier
from modules.behavior_engine import BehaviorEngine
from modules.zone_manager import ZoneManager
from modules.alert_system import AlertSystem
from modules.evidence_recorder import EvidenceRecorder


class BehaviorDetectionSystem:
    """
    Main system orchestrator.
    Manages the full pipeline: capture → detect → track → recognize →
    pose → action → behavior → alert → evidence → dashboard stream.
    """
    
    def __init__(self, source=None, enable_dashboard=True):
        self.source = source or config.video.source
        self.enable_dashboard = enable_dashboard
        
        print("=" * 60)
        print("  AI-Powered Behavior & Motion Detection System")
        print("=" * 60)
        print(f"  Source: {self.source}")
        print(f"  Dashboard: {'Enabled' if enable_dashboard else 'Disabled'}")
        print("=" * 60)
        
        # ---- Initialize Modules ----
        print("\n[Init] Loading modules...")
        
        self.preprocessor = FramePreprocessor(config.preprocess)
        self.detector = PersonDetector(config.detection)
        self.tracker = PersonTracker(config.tracking)
        self.recognizer = PersonRecognizer(config.recognition)
        self.pose_estimator = PoseEstimator(config.pose)
        self.action_classifier = ActionClassifier(config.action)
        self.behavior_engine = BehaviorEngine(config.behavior)
        self.zone_manager = ZoneManager(config.zones)
        self.alert_system = AlertSystem(config.alert)
        self.evidence_recorder = EvidenceRecorder(config.evidence, fps=config.video.fps)
        
        # ---- Dashboard State ----
        self.dashboard_state = None
        if enable_dashboard:
            from dashboard.app import state as dashboard_state, start_dashboard
            self.dashboard_state = dashboard_state
            self.dashboard_state.alert_system = self.alert_system
            self.dashboard_state.recognizer = self.recognizer
            self.dashboard_state.evidence_recorder = self.evidence_recorder
            self.dashboard_state.tracker = self.tracker
            self.dashboard_state.zone_manager = self.zone_manager
            self._start_dashboard = start_dashboard
        
        # ---- Runtime State ----
        self.is_running = False
        self.frame_count = 0
        self.fps = 0.0
        self._fps_start_time = time.time()
        self._fps_frame_count = 0
        
        # Ensure directories exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("known_faces", exist_ok=True)
        os.makedirs("evidence", exist_ok=True)
        
        print("\n[Init] All modules loaded successfully!\n")
    
    def run(self):
        """Start the main processing loop."""
        # Start dashboard in background thread
        if self.enable_dashboard:
            dashboard_thread = threading.Thread(
                target=self._start_dashboard,
                args=(self.dashboard_state,),
                daemon=True
            )
            dashboard_thread.start()
            print(f"[Dashboard] Started at http://localhost:{config.dashboard.port}")
        
        # Open video source
        source = self.source
        if source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video source: {self.source}")
            print("  Try: python main.py --source 0        (webcam)")
            print("  Try: python main.py --source video.mp4 (file)")
            return
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.video.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video.height)
        
        print(f"[System] Video source opened: {int(cap.get(3))}x{int(cap.get(4))}")
        print(f"[System] Press 'q' to quit, 'r' to register face, 'z' to toggle zones\n")
        
        self.is_running = True
        if self.dashboard_state:
            self.dashboard_state.is_running = True
        
        show_zones = True
        show_skeleton = True
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    # Try to loop video files
                    if isinstance(source, str) and os.path.isfile(source):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    print("[System] Video stream ended")
                    break
                
                self.frame_count += 1
                current_time = time.time()
                
                # ========================================
                # PIPELINE STEP 1: Preprocess
                # ========================================
                processed_frame = self.preprocessor.process(frame)
                
                # ========================================
                # PIPELINE STEP 2: Detect Persons
                # ========================================
                detections = self.detector.detect(processed_frame)
                
                # ========================================
                # PIPELINE STEP 3: Track Persons
                # ========================================
                tracked_persons = self.tracker.update(detections, processed_frame)
                
                # ========================================
                # PIPELINE STEP 4-7: Per-person analysis
                # ========================================
                annotated_frame = processed_frame.copy()
                
                # Draw zones
                if show_zones:
                    annotated_frame = self.zone_manager.draw_zones(annotated_frame)
                
                behavior_scores = {}
                
                for person in tracked_persons:
                    tid = person.track_id
                    bbox = person.bbox
                    
                    # ---- Step 4: Face Recognition ----
                    person_name = self.recognizer.recognize(
                        processed_frame, bbox, tid
                    )
                    self.tracker.set_person_name(tid, person_name)
                    
                    # ---- Step 5: Pose Estimation ----
                    pose_result = self.pose_estimator.estimate(
                        processed_frame, bbox, tid
                    )
                    
                    # ---- Step 6: Action Classification ----
                    action_result = None
                    if pose_result is not None:
                        action_result = self.action_classifier.classify(
                            pose_result, tid
                        )
                    
                    # ---- Step 7: Zone Check ----
                    h, w = frame.shape[:2]
                    zone_events = self.zone_manager.check_zones(
                        tid, person.center, w, h, current_time
                    )
                    
                    # ---- Step 8: Behavior Analysis ----
                    score = self.behavior_engine.analyze(
                        person, action_result, pose_result, zone_events
                    )
                    score.person_name = person_name
                    behavior_scores[tid] = score
                    
                    # ---- Step 9: Alert Check ----
                    alert = self.alert_system.check_and_alert(
                        score, self.frame_count
                    )
                    
                    if alert:
                        # Start evidence recording
                        self.evidence_recorder.start_recording(
                            alert.alert_id, tid, person_name,
                            score.total_score, score.reasons
                        )
                        
                        # Flag person if high alert
                        if alert.alert_level == "high" and person_name != "Unknown":
                            self.recognizer.flag_person(
                                person_name, 
                                f"High alert: {', '.join(score.reasons[:2])}"
                            )
                    
                    # ---- Draw Annotations ----
                    annotated_frame = self._draw_person_annotations(
                        annotated_frame, person, score,
                        action_result, pose_result if show_skeleton else None
                    )
                
                # ========================================
                # PIPELINE STEP 10: Evidence Frame Feed
                # ========================================
                self.evidence_recorder.feed_frame(annotated_frame)
                
                # ========================================
                # PIPELINE STEP 11: FPS Calculation
                # ========================================
                self._fps_frame_count += 1
                elapsed = current_time - self._fps_start_time
                if elapsed >= 1.0:
                    self.fps = self._fps_frame_count / elapsed
                    self._fps_frame_count = 0
                    self._fps_start_time = current_time
                
                # ========================================
                # PIPELINE STEP 12: Dashboard Update
                # ========================================
                if self.dashboard_state:
                    # Encode frame as JPEG for WebSocket streaming
                    _, jpeg = cv2.imencode(
                        '.jpg', annotated_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, config.dashboard.stream_quality]
                    )
                    self.dashboard_state.current_frame = jpeg.tobytes()
                    self.dashboard_state.current_scores = behavior_scores
                    self.dashboard_state.fps = self.fps
                    self.dashboard_state.frame_count = self.frame_count
                    self.dashboard_state.persons_count = len(tracked_persons)
                
                # ========================================
                # PIPELINE STEP 13: Local Display
                # ========================================
                # Draw FPS on local window
                cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Persons: {len(tracked_persons)}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
                
                cv2.imshow("AI Behavior Detection", annotated_frame)
                
                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('z'):
                    show_zones = not show_zones
                elif key == ord('s'):
                    show_skeleton = not show_skeleton
                elif key == ord('r'):
                    self._register_face_interactive(frame)
        
        except KeyboardInterrupt:
            print("\n[System] Interrupted by user")
        
        finally:
            self.is_running = False
            if self.dashboard_state:
                self.dashboard_state.is_running = False
            
            cap.release()
            cv2.destroyAllWindows()
            self.pose_estimator.release()
            print("[System] Shutdown complete")
    
    def _draw_person_annotations(self, frame, person, score,
                                  action=None, pose=None):
        """Draw bounding box, labels, and skeleton for a person."""
        x1, y1, x2, y2 = person.bbox
        
        # Color based on behavior score
        if score.alert_level == "high":
            color = (0, 0, 255)       # Red
        elif score.alert_level == "medium":
            color = (0, 165, 255)     # Orange
        elif score.alert_level == "low":
            color = (0, 255, 255)     # Yellow
        else:
            color = (0, 255, 0)       # Green
        
        # Bounding box
        thickness = 3 if score.alert_level in ("high", "medium") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Label background
        label = f"#{person.track_id} {person.person_name}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - 22), (x1 + label_size[0] + 8, y1),
                     color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Action label
        if action:
            action_label = f"{action.action} ({action.confidence:.0%})"
            cv2.putText(frame, action_label, (x1, y2 + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Behavior score
        score_label = f"Score: {score.total_score:.0f}"
        cv2.putText(frame, score_label, (x1, y2 + 36),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        
        # Alert indicator
        if score.alert_level != "none":
            alert_text = f"⚠ {score.alert_level.upper()} ALERT"
            cv2.putText(frame, alert_text, (x1, y1 - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Trajectory line
        if len(person.trajectory) > 1:
            points = list(person.trajectory)
            for i in range(1, len(points)):
                alpha = i / len(points)
                pt_color = tuple(int(c * alpha) for c in color)
                cv2.line(frame, points[i-1], points[i], pt_color, 2)
        
        # Skeleton overlay
        if pose is not None:
            frame = self.pose_estimator.draw_skeleton(frame, pose, color)
        
        # Flagged person indicator
        if self.recognizer.is_flagged(person.person_name):
            cv2.putText(frame, "🚩 FLAGGED", (x2 - 90, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame
    
    def _register_face_interactive(self, frame):
        """Interactive face registration from live feed."""
        name = input("[Register] Enter person's name: ").strip()
        if name:
            success = self.recognizer.register_face(name, frame)
            if success:
                print(f"[Register] Registered: {name}")
            else:
                print("[Register] Registration failed")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Behavior & Motion Detection System"
    )
    parser.add_argument(
        "--source", type=str, default=config.video.source,
        help="Video source: webcam index (0), file path, or RTSP URL"
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Run without the web dashboard"
    )
    parser.add_argument(
        "--model", type=str, default=config.detection.model_name,
        help="YOLOv8 model name (yolov8n.pt, yolov8s.pt, yolov8m.pt)"
    )
    parser.add_argument(
        "--confidence", type=float, default=config.detection.confidence_threshold,
        help="Detection confidence threshold (0-1)"
    )
    parser.add_argument(
        "--port", type=int, default=config.dashboard.port,
        help="Dashboard port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Apply CLI overrides
    config.video.source = args.source
    config.detection.model_name = args.model
    config.detection.confidence_threshold = args.confidence
    config.dashboard.port = args.port
    
    # Create and run system
    system = BehaviorDetectionSystem(
        source=args.source,
        enable_dashboard=not args.no_dashboard
    )
    system.run()


if __name__ == "__main__":
    main()

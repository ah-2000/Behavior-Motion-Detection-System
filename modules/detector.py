"""
Person Detector — YOLOv8-based human detection.
Filters for person class only with configurable confidence.
"""
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import DetectionConfig


@dataclass
class Detection:
    """Single person detection result."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0
    
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
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_ltwh(self) -> Tuple[int, int, int, int]:
        """Convert to (left, top, width, height) for DeepSORT."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)


class PersonDetector:
    """YOLOv8-based person detection with confidence filtering."""
    
    def __init__(self, cfg: DetectionConfig = None):
        self.cfg = cfg or DetectionConfig()
        
        if YOLO is None:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        
        # Determine device
        device = self.cfg.device
        if device == "auto":
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.device = device
        self.model = YOLO(self.cfg.model_name)
        print(f"[Detector] YOLOv8 loaded: {self.cfg.model_name} on {self.device}")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in a frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            List of Detection objects (persons only)
        """
        results = self.model(
            frame,
            conf=self.cfg.confidence_threshold,
            iou=self.cfg.iou_threshold,
            classes=[self.cfg.person_class_id],
            device=self.device,
            imgsz=self.cfg.img_size,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                if cls == self.cfg.person_class_id:
                    detections.append(Detection(
                        bbox=tuple(bbox),
                        confidence=conf,
                        class_id=cls
                    ))
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame (for debugging)."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {det.confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return annotated

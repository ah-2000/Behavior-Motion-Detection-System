"""
Zone Manager — Define and monitor scene zones.
Supports restricted areas, normal zones, exits for spatial behavior analysis.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from config import ZoneConfig


@dataclass
class Zone:
    """A defined zone in the camera view."""
    name: str
    zone_type: str                          # "normal", "restricted", "inventory", "exit"
    points: List[Tuple[float, float]]       # Normalized [0,1] polygon points
    color: Tuple[int, int, int] = (0, 255, 0)
    
    def get_pixel_points(self, frame_w: int, frame_h: int) -> np.ndarray:
        """Convert normalized points to pixel coordinates."""
        pixel_points = []
        for x, y in self.points:
            pixel_points.append([int(x * frame_w), int(y * frame_h)])
        return np.array(pixel_points, dtype=np.int32)
    
    def contains_point(self, px: int, py: int, frame_w: int, frame_h: int) -> bool:
        """Check if a pixel point is inside this zone."""
        polygon = self.get_pixel_points(frame_w, frame_h)
        result = cv2.pointPolygonTest(polygon, (float(px), float(py)), False)
        return result >= 0


# Zone type colors
ZONE_COLORS = {
    "normal": (0, 200, 0),         # Green
    "restricted": (0, 0, 255),     # Red
    "inventory": (0, 165, 255),    # Orange
    "exit": (255, 200, 0),         # Cyan
}


@dataclass
class ZoneEvent:
    """Event generated when a person interacts with a zone."""
    track_id: int
    zone_name: str
    zone_type: str
    event_type: str       # "enter", "exit", "linger"
    duration: float       # Seconds in zone (for linger events)
    timestamp: float


class ZoneManager:
    """Manages scene zones and detects zone violations."""
    
    def __init__(self, cfg: ZoneConfig = None):
        self.cfg = cfg or ZoneConfig()
        
        # Parse zones from config
        self.zones: List[Zone] = []
        for zone_def in self.cfg.zones:
            color = ZONE_COLORS.get(zone_def["type"], (128, 128, 128))
            self.zones.append(Zone(
                name=zone_def["name"],
                zone_type=zone_def["type"],
                points=zone_def["points"],
                color=color
            ))
        
        # Track person zone occupancy: track_id -> {zone_name: entry_time}
        self._occupancy: Dict[int, Dict[str, float]] = {}
        
        print(f"[ZoneManager] Initialized with {len(self.zones)} zones")
    
    def check_zones(self, track_id: int, center: Tuple[int, int],
                    frame_w: int, frame_h: int,
                    current_time: float) -> List[ZoneEvent]:
        """
        Check which zones a person is in and generate events.
        
        Args:
            track_id: Person's tracker ID
            center: Person's center point (px, py)
            frame_w, frame_h: Frame dimensions
            current_time: Current timestamp
            
        Returns:
            List of ZoneEvent objects
        """
        events = []
        px, py = center
        
        if track_id not in self._occupancy:
            self._occupancy[track_id] = {}
        
        current_zones = set()
        
        for zone in self.zones:
            in_zone = zone.contains_point(px, py, frame_w, frame_h)
            
            if in_zone:
                current_zones.add(zone.name)
                
                if zone.name not in self._occupancy[track_id]:
                    # Person just entered this zone
                    self._occupancy[track_id][zone.name] = current_time
                    events.append(ZoneEvent(
                        track_id=track_id,
                        zone_name=zone.name,
                        zone_type=zone.zone_type,
                        event_type="enter",
                        duration=0.0,
                        timestamp=current_time
                    ))
                else:
                    # Person still in zone — check for lingering
                    entry_time = self._occupancy[track_id][zone.name]
                    duration = current_time - entry_time
                    
                    if zone.zone_type == "restricted" and duration > 5.0:
                        events.append(ZoneEvent(
                            track_id=track_id,
                            zone_name=zone.name,
                            zone_type=zone.zone_type,
                            event_type="linger",
                            duration=duration,
                            timestamp=current_time
                        ))
        
        # Check for zone exits
        prev_zones = set(self._occupancy[track_id].keys())
        exited_zones = prev_zones - current_zones
        
        for zone_name in exited_zones:
            zone = self._get_zone_by_name(zone_name)
            if zone:
                entry_time = self._occupancy[track_id].pop(zone_name)
                duration = current_time - entry_time
                events.append(ZoneEvent(
                    track_id=track_id,
                    zone_name=zone_name,
                    zone_type=zone.zone_type,
                    event_type="exit",
                    duration=duration,
                    timestamp=current_time
                ))
        
        return events
    
    def get_zone_at_point(self, px: int, py: int,
                          frame_w: int, frame_h: int) -> Optional[Zone]:
        """Get the zone containing a specific point."""
        for zone in self.zones:
            if zone.contains_point(px, py, frame_w, frame_h):
                return zone
        return None
    
    def get_person_zones(self, track_id: int) -> Dict[str, float]:
        """Get zones a person is currently in with entry times."""
        return self._occupancy.get(track_id, {})
    
    def _get_zone_by_name(self, name: str) -> Optional[Zone]:
        for zone in self.zones:
            if zone.name == name:
                return zone
        return None
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw all zones on frame as semi-transparent overlays."""
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        for zone in self.zones:
            points = zone.get_pixel_points(w, h)
            
            # Semi-transparent fill
            cv2.fillPoly(overlay, [points], zone.color)
            
            # Border
            cv2.polylines(frame, [points], True, zone.color, 2)
            
            # Label
            centroid = np.mean(points, axis=0).astype(int)
            label = f"{zone.name} ({zone.zone_type})"
            cv2.putText(frame, label, tuple(centroid),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend overlay
        alpha = 0.15
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return result
    
    def add_zone(self, name: str, zone_type: str,
                 points: List[Tuple[float, float]]):
        """Add a new zone at runtime."""
        color = ZONE_COLORS.get(zone_type, (128, 128, 128))
        self.zones.append(Zone(
            name=name, zone_type=zone_type,
            points=points, color=color
        ))
        print(f"[ZoneManager] Added zone: {name} ({zone_type})")
    
    def remove_zone(self, name: str):
        """Remove a zone by name."""
        self.zones = [z for z in self.zones if z.name != name]

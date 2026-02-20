"""
Alert System — Multi-channel alert management with cooldown.
Triggers sound, visual, and optional email/Telegram notifications.
"""
import os
import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from config import AlertConfig
from modules.behavior_engine import BehaviorScore


@dataclass
class Alert:
    """A triggered alert record."""
    alert_id: str
    track_id: int
    person_name: str
    alert_level: str            # "low", "medium", "high"
    behavior_score: float
    reasons: List[str]
    timestamp: float
    frame_number: int = 0
    evidence_path: str = ""
    acknowledged: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @property
    def formatted_time(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")


ALERT_COLORS = {
    "low": (0, 255, 255),         # Yellow
    "medium": (0, 165, 255),      # Orange  
    "high": (0, 0, 255),          # Red
}


class AlertSystem:
    """
    Manages alert generation, cooldowns, logging, and notifications.
    """
    
    def __init__(self, cfg: AlertConfig = None):
        self.cfg = cfg or AlertConfig()
        
        # Alert history
        self._alerts: List[Alert] = []
        self._alert_counter = 0
        
        # Per-person cooldown tracker: track_id -> last_alert_time
        self._cooldowns: Dict[int, float] = {}
        
        # Callback for real-time alert push (set by dashboard)
        self._alert_callback = None
        
        # Setup logging
        logging.basicConfig(
            filename=self.cfg.log_file,
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s'
        )
        self._logger = logging.getLogger("AlertSystem")
        
        print(f"[AlertSystem] Initialized — cooldown={self.cfg.cooldown_seconds}s")
    
    def check_and_alert(self, score: BehaviorScore,
                        frame_number: int = 0) -> Optional[Alert]:
        """
        Check behavior score and generate alert if thresholds are met.
        Respects cooldown to prevent alert spam.
        
        Args:
            score: BehaviorScore from the behavior engine
            frame_number: Current frame number for evidence linking
            
        Returns:
            Alert object if triggered, None otherwise
        """
        # Only alert on non-normal levels
        if score.alert_level == "none":
            return None
        
        # Check cooldown
        track_id = score.track_id
        current_time = time.time()
        
        if track_id in self._cooldowns:
            elapsed = current_time - self._cooldowns[track_id]
            if elapsed < self.cfg.cooldown_seconds:
                return None  # Still in cooldown
        
        # Generate alert
        self._alert_counter += 1
        alert = Alert(
            alert_id=f"ALERT-{self._alert_counter:06d}",
            track_id=track_id,
            person_name=score.person_name,
            alert_level=score.alert_level,
            behavior_score=score.total_score,
            reasons=score.reasons,
            timestamp=current_time,
            frame_number=frame_number
        )
        
        # Store and log
        self._alerts.append(alert)
        self._cooldowns[track_id] = current_time
        
        self._log_alert(alert)
        
        # Push via callback (for WebSocket dashboard)
        if self._alert_callback:
            try:
                self._alert_callback(alert)
            except Exception as e:
                print(f"[AlertSystem] Callback error: {e}")
        
        return alert
    
    def _log_alert(self, alert: Alert):
        """Log alert to file and console."""
        level_emoji = {"low": "🟡", "medium": "🟠", "high": "🔴"}
        emoji = level_emoji.get(alert.alert_level, "⚪")
        
        msg = (f"{emoji} {alert.alert_level.upper()} ALERT | "
               f"Person: {alert.person_name} (ID: {alert.track_id}) | "
               f"Score: {alert.behavior_score:.1f} | "
               f"Reasons: {', '.join(alert.reasons)}")
        
        print(f"[AlertSystem] {msg}")
        self._logger.warning(msg)
    
    def set_callback(self, callback):
        """Set callback function for real-time alert pushing."""
        self._alert_callback = callback
    
    def get_recent_alerts(self, count: int = 50) -> List[Alert]:
        """Get the most recent alerts."""
        return self._alerts[-count:]
    
    def get_alerts_for_person(self, track_id: int) -> List[Alert]:
        """Get all alerts for a specific person."""
        return [a for a in self._alerts if a.track_id == track_id]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        total = len(self._alerts)
        by_level = {"low": 0, "medium": 0, "high": 0}
        for alert in self._alerts:
            by_level[alert.alert_level] = by_level.get(alert.alert_level, 0) + 1
        
        return {
            "total_alerts": total,
            "by_level": by_level,
            "unacknowledged": sum(1 for a in self._alerts if not a.acknowledged),
            "unique_persons": len(set(a.track_id for a in self._alerts))
        }
    
    def link_evidence(self, alert_id: str, evidence_path: str):
        """Link an evidence clip to an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.evidence_path = evidence_path
                return
    
    def export_alerts(self, filepath: str = "alerts_export.json"):
        """Export all alerts to JSON."""
        data = [a.to_dict() for a in self._alerts]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[AlertSystem] Exported {len(data)} alerts to {filepath}")

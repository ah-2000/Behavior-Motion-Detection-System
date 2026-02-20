"""
Frame Preprocessor — Lighting & Quality Normalization
Handles poor lighting, varying camera conditions with CLAHE and gamma correction.
"""
import cv2
import numpy as np
from config import PreprocessConfig


class FramePreprocessor:
    """Normalizes frame quality for robust detection under varying conditions."""
    
    def __init__(self, cfg: PreprocessConfig = None):
        self.cfg = cfg or PreprocessConfig()
        if self.cfg.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.cfg.clahe_clip_limit,
                tileGridSize=self.cfg.clahe_grid_size
            )
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps to a frame."""
        if frame is None or frame.size == 0:
            return frame
        
        processed = frame.copy()
        
        # 1. Auto gamma correction
        if self.cfg.enable_gamma:
            processed = self._auto_gamma(processed)
        
        # 2. CLAHE (adaptive histogram equalization)
        if self.cfg.enable_clahe:
            processed = self._apply_clahe(processed)
        
        return processed
    
    def _auto_gamma(self, frame: np.ndarray) -> np.ndarray:
        """Automatically adjust gamma based on frame brightness."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Calculate gamma to bring brightness closer to target
        if mean_brightness < 10:
            mean_brightness = 10  # Avoid division issues
        
        gamma = np.log(self.cfg.target_brightness / 255.0) / np.log(mean_brightness / 255.0)
        gamma = np.clip(gamma, 0.3, 3.0)  # Clamp to reasonable range
        
        # Build lookup table and apply
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")
        
        return cv2.LUT(frame, table)
    
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE to the luminance channel (LAB color space)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to luminance
        l_enhanced = self.clahe.apply(l_channel)
        
        # Merge back
        enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    def get_brightness(self, frame: np.ndarray) -> float:
        """Get mean brightness of a frame (0-255)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

"""
Action Classifier — LSTM-based human action recognition.
Classifies sequences of pose landmarks into action labels.
Includes both inference and training pipeline.
"""
import os
import numpy as np
from typing import Optional, List, Dict
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn

from config import ActionConfig
from modules.pose_estimator import PoseResult


@dataclass
class ActionPrediction:
    """Result of action classification."""
    action: str
    confidence: float
    all_scores: Dict[str, float]
    track_id: int = -1


class ActionLSTM(nn.Module):
    """2-layer LSTM for skeleton-based action recognition."""
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int, num_classes: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, sequence_length, num_features)
        Returns:
            logits: (batch, num_classes)
        """
        # LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Classify
        logits = self.fc(out)
        return logits


class ActionClassifier:
    """
    Manages LSTM model for action classification.
    Maintains per-person pose sequence buffers for temporal analysis.
    """
    
    def __init__(self, cfg: ActionConfig = None):
        self.cfg = cfg or ActionConfig()
        
        # Per-person pose sequence buffers: track_id -> deque of feature vectors
        self._buffers: Dict[int, deque] = {}
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build model
        self.model = ActionLSTM(
            input_size=self.cfg.num_features,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            num_classes=self.cfg.num_classes
        ).to(self.device)
        
        # Load weights if available
        self._model_loaded = False
        if os.path.exists(self.cfg.model_path):
            try:
                state_dict = torch.load(self.cfg.model_path, 
                                         map_location=self.device,
                                         weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self._model_loaded = True
                print(f"[ActionClassifier] Loaded model from {self.cfg.model_path}")
            except Exception as e:
                print(f"[ActionClassifier] Could not load model: {e}")
        
        if not self._model_loaded:
            print(f"[ActionClassifier] No pre-trained model found. "
                  f"Using heuristic-based classification until trained.")
            self.model.eval()
        
        print(f"[ActionClassifier] Device: {self.device}, "
              f"Actions: {self.cfg.action_labels}")
    
    def classify(self, pose_result: PoseResult, track_id: int) -> ActionPrediction:
        """
        Classify the action of a tracked person based on pose history.
        
        Args:
            pose_result: Current frame pose estimation
            track_id: Person's tracker ID
            
        Returns:
            ActionPrediction with action label and confidence
        """
        # Get or create buffer for this person
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self.cfg.sequence_length)
        
        # Add current pose features to buffer
        features = pose_result.to_feature_vector()
        self._buffers[track_id].append(features)
        
        # If we don't have a trained model, use heuristics
        if not self._model_loaded:
            return self._heuristic_classify(pose_result, track_id)
        
        # Need full sequence for LSTM
        buffer = self._buffers[track_id]
        if len(buffer) < self.cfg.sequence_length:
            # Pad with zeros for initial frames
            padded = np.zeros((self.cfg.sequence_length, self.cfg.num_features))
            start = self.cfg.sequence_length - len(buffer)
            for i, feat in enumerate(buffer):
                padded[start + i] = feat
            sequence = padded
        else:
            sequence = np.array(list(buffer))
        
        # Run LSTM inference
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get prediction
        pred_idx = int(np.argmax(probs))
        pred_conf = float(probs[pred_idx])
        pred_action = self.cfg.action_labels[pred_idx]
        
        # Build scores dict
        all_scores = {
            self.cfg.action_labels[i]: float(probs[i])
            for i in range(len(self.cfg.action_labels))
        }
        
        # Apply confidence threshold
        if pred_conf < self.cfg.confidence_threshold:
            pred_action = "standing"  # Default to standing if uncertain
        
        return ActionPrediction(
            action=pred_action,
            confidence=pred_conf,
            all_scores=all_scores,
            track_id=track_id
        )
    
    def _heuristic_classify(self, pose: PoseResult, track_id: int) -> ActionPrediction:
        """
        Rule-based classification when no trained LSTM model is available.
        Uses pose geometry to infer actions.
        """
        scores = {action: 0.0 for action in self.cfg.action_labels}
        
        # Calculate key angles and features
        is_bending = pose.is_bending()
        hands_low = pose.hands_below_waist
        concealing = pose.hand_to_bag_gesture
        
        # Spine angle (shoulder-hip-knee)
        try:
            spine_angle = pose.get_angle(11, 23, 25)
        except Exception:
            spine_angle = 170.0
        
        # Elbow angles
        try:
            left_elbow_angle = pose.get_angle(11, 13, 15)
            right_elbow_angle = pose.get_angle(12, 14, 16)
        except Exception:
            left_elbow_angle = right_elbow_angle = 170.0
        
        # Check wrist positions relative to shoulders for reaching
        try:
            left_wrist_y = pose.normalized_landmarks[15, 1]
            right_wrist_y = pose.normalized_landmarks[16, 1]
            left_shoulder_y = pose.normalized_landmarks[11, 1]
            right_shoulder_y = pose.normalized_landmarks[12, 1]
            reaching_up = (left_wrist_y < left_shoulder_y - 0.1 or 
                          right_wrist_y < right_shoulder_y - 0.1)
        except Exception:
            reaching_up = False
        
        # Classify based on heuristics
        if concealing:
            scores["hiding_object"] = 0.75
            scores["bending"] = 0.15
        elif is_bending and hands_low:
            scores["reaching_down"] = 0.65
            scores["bending"] = 0.25
        elif is_bending:
            scores["bending"] = 0.70
            scores["reaching_down"] = 0.15
        elif reaching_up:
            scores["reaching_up"] = 0.70
            scores["standing"] = 0.15
        elif spine_angle > 160:
            scores["standing"] = 0.70
            scores["walking"] = 0.15
        else:
            scores["standing"] = 0.50
            scores["walking"] = 0.25
        
        # Determine carrying based on arm position
        if left_elbow_angle < 90 or right_elbow_angle < 90:
            scores["carrying_object"] = max(scores.get("carrying_object", 0), 0.55)
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        # Get top prediction
        best_action = max(scores, key=scores.get)
        best_conf = scores[best_action]
        
        return ActionPrediction(
            action=best_action,
            confidence=best_conf,
            all_scores=scores,
            track_id=track_id
        )
    
    def clear_buffer(self, track_id: int):
        """Clear the sequence buffer for a specific person."""
        if track_id in self._buffers:
            del self._buffers[track_id]
    
    def clear_all_buffers(self):
        """Clear all sequence buffers."""
        self._buffers.clear()

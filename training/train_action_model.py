"""
LSTM Action Classifier — Training Pipeline
==========================================
Train a custom action recognition model using pose landmark sequences.

Usage:
    1. Collect training data:
       python training/train_action_model.py --collect --source video.mp4

    2. Train the model:
       python training/train_action_model.py --train

    3. Evaluate:
       python training/train_action_model.py --evaluate
       
Data Format:
    training/data/
    ├── standing/
    │   ├── sequence_001.npy
    │   └── ...
    ├── walking/
    ├── bending/
    ├── hiding_object/
    └── ...
    
Each .npy file contains a (sequence_length, 132) array of pose landmarks.
"""
import os
import sys
import cv2
import json
import numpy as np
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from config import config
from modules.action_classifier import ActionLSTM
from modules.pose_estimator import PoseEstimator
from modules.detector import PersonDetector


# ============================================================
# DATASET
# ============================================================
class ActionDataset(Dataset):
    """Dataset of pose landmark sequences with action labels."""
    
    def __init__(self, data_dir: str, sequence_length: int, action_labels: list):
        self.sequence_length = sequence_length
        self.action_labels = action_labels
        self.samples = []   # (sequence_array, label_index)
        
        for label_idx, label in enumerate(action_labels):
            label_dir = os.path.join(data_dir, label)
            if not os.path.isdir(label_dir):
                continue
            
            for npy_file in os.listdir(label_dir):
                if npy_file.endswith('.npy'):
                    filepath = os.path.join(label_dir, npy_file)
                    try:
                        sequence = np.load(filepath)
                        if sequence.shape[0] >= sequence_length:
                            # Take last N frames
                            sequence = sequence[-sequence_length:]
                        else:
                            # Pad with zeros
                            padded = np.zeros((sequence_length, sequence.shape[1]))
                            padded[-sequence.shape[0]:] = sequence
                            sequence = padded
                        
                        self.samples.append((sequence, label_idx))
                    except Exception as e:
                        print(f"  Warning: Could not load {filepath}: {e}")
        
        print(f"[Dataset] Loaded {len(self.samples)} samples across "
              f"{len(action_labels)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence, label = self.samples[idx]
        return torch.FloatTensor(sequence), label


# ============================================================
# DATA COLLECTOR
# ============================================================
class DataCollector:
    """Interactive tool to collect labeled pose sequences from video."""
    
    def __init__(self, source, data_dir="training/data"):
        self.source = source
        self.data_dir = data_dir
        self.detector = PersonDetector(config.detection)
        self.pose_estimator = PoseEstimator(config.pose)
        self.action_labels = config.action.action_labels
        self.seq_length = config.action.sequence_length
        
        # Create label directories
        for label in self.action_labels:
            os.makedirs(os.path.join(data_dir, label), exist_ok=True)
        
        self.current_sequence = []
        self.current_label = None
        self.recording = False
    
    def collect(self):
        """Run interactive collection with keyboard controls."""
        source = self.source
        if source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.source}")
            return
        
        print("\n" + "=" * 60)
        print("  ACTION DATA COLLECTOR")
        print("=" * 60)
        print("\nControls:")
        for i, label in enumerate(self.action_labels):
            print(f"  Key [{i}] — Start recording '{label}'")
        print(f"  Key [SPACE] — Stop recording and save")
        print(f"  Key [x] — Discard current recording")
        print(f"  Key [q] — Quit\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and get pose
            detections = self.detector.detect(frame)
            
            display = frame.copy()
            
            if detections:
                det = detections[0]  # Use first detected person
                bbox = det.bbox
                
                pose = self.pose_estimator.estimate(frame, bbox)
                
                if pose is not None:
                    display = self.pose_estimator.draw_skeleton(display, pose)
                    
                    if self.recording:
                        features = pose.to_feature_vector()
                        self.current_sequence.append(features)
                
                cv2.rectangle(display, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Status overlay
            status = "NOT RECORDING"
            status_color = (128, 128, 128)
            if self.recording:
                status = (f"RECORDING: {self.current_label} "
                         f"[{len(self.current_sequence)}/{self.seq_length} frames]")
                status_color = (0, 0, 255)
            
            cv2.putText(display, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            cv2.imshow("Data Collector", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space — save
                if self.recording and len(self.current_sequence) >= 10:
                    self._save_sequence()
                self.recording = False
            elif key == ord('x'):  # Discard
                self.current_sequence = []
                self.recording = False
                print("  Discarded current recording")
            elif chr(key).isdigit():
                idx = int(chr(key))
                if idx < len(self.action_labels):
                    self.current_label = self.action_labels[idx]
                    self.current_sequence = []
                    self.recording = True
                    print(f"  Recording: {self.current_label}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.pose_estimator.release()
    
    def _save_sequence(self):
        """Save recorded sequence as .npy file."""
        if not self.current_sequence or not self.current_label:
            return
        
        sequence = np.array(self.current_sequence)
        label_dir = os.path.join(self.data_dir, self.current_label)
        
        existing = len([f for f in os.listdir(label_dir) if f.endswith('.npy')])
        filename = f"sequence_{existing + 1:04d}.npy"
        filepath = os.path.join(label_dir, filename)
        
        np.save(filepath, sequence)
        print(f"  ✓ Saved: {filepath} ({sequence.shape[0]} frames)")
        
        self.current_sequence = []


# ============================================================
# TRAINER
# ============================================================
def train_model(data_dir="training/data", epochs=100, batch_size=16, lr=0.001):
    """Train the LSTM action classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Device: {device}")
    
    # Load dataset
    dataset = ActionDataset(
        data_dir=data_dir,
        sequence_length=config.action.sequence_length,
        action_labels=config.action.action_labels
    )
    
    if len(dataset) < 10:
        print("[ERROR] Need at least 10 samples to train!")
        print(f"  Current: {len(dataset)} samples")
        print(f"  Collect data first: python training/train_action_model.py --collect --source 0")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"[Training] Train: {train_size}, Val: {val_size}")
    
    # Build model
    model = ActionLSTM(
        input_size=config.action.num_features,
        hidden_size=config.action.hidden_size,
        num_layers=config.action.num_layers,
        num_classes=config.action.num_classes
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total * 100
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / max(val_total, 1) * 100
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Train Acc: {train_acc:.1f}%, "
                  f"Val Acc: {val_acc:.1f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), config.action.model_path)
    
    print(f"\n[Training] Complete! Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"[Training] Model saved to: {config.action.model_path}")


# ============================================================
# EVALUATOR
# ============================================================
def evaluate_model(data_dir="training/data"):
    """Evaluate the trained model and print confusion matrix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = ActionDataset(
        data_dir=data_dir,
        sequence_length=config.action.sequence_length,
        action_labels=config.action.action_labels
    )
    
    if len(dataset) == 0:
        print("[ERROR] No data found!")
        return
    
    loader = DataLoader(dataset, batch_size=32)
    
    model = ActionLSTM(
        input_size=config.action.num_features,
        hidden_size=config.action.hidden_size,
        num_layers=config.action.num_layers,
        num_classes=config.action.num_classes
    ).to(device)
    
    if not os.path.exists(config.action.model_path):
        print(f"[ERROR] No model found at {config.action.model_path}")
        return
    
    model.load_state_dict(
        torch.load(config.action.model_path, map_location=device, weights_only=True)
    )
    model.eval()
    
    # Confusion matrix
    num_classes = config.action.num_classes
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = outputs.max(1)
            
            for true, pred in zip(labels.numpy(), predicted.cpu().numpy()):
                confusion[true][pred] += 1
    
    # Print results
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    
    total = confusion.sum()
    correct = confusion.diagonal().sum()
    print(f"\n  Overall Accuracy: {correct/total*100:.1f}%")
    
    print(f"\n  Per-class accuracy:")
    for i, label in enumerate(config.action.action_labels):
        class_total = confusion[i].sum()
        if class_total > 0:
            class_acc = confusion[i][i] / class_total * 100
            print(f"    {label:20s}: {class_acc:5.1f}% ({confusion[i][i]}/{class_total})")
    
    print(f"\n  Confusion Matrix:")
    header = "  " + " " * 20 + " ".join(f"{l[:5]:>6}" for l in config.action.action_labels)
    print(header)
    for i, label in enumerate(config.action.action_labels):
        row = " ".join(f"{confusion[i][j]:6d}" for j in range(num_classes))
        print(f"  {label:20s}{row}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action Model Training Pipeline")
    parser.add_argument("--collect", action="store_true", help="Collect training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--source", type=str, default="0", help="Video source for collection")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--data-dir", type=str, default="training/data", help="Data directory")
    
    args = parser.parse_args()
    
    if args.collect:
        collector = DataCollector(args.source, args.data_dir)
        collector.collect()
    elif args.train:
        train_model(args.data_dir, args.epochs)
    elif args.evaluate:
        evaluate_model(args.data_dir)
    else:
        parser.print_help()

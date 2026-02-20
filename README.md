# 🛡️ AI-Powered Behavior & Motion Detection System

Real-time video surveillance system that detects suspicious behavior using AI — combining person detection, tracking, face recognition, pose estimation, and action classification into a unified behavior scoring engine.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Detection-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Dashboard-red)

---

## 🎯 What Does It Do?

The system processes live video feeds (webcam or CCTV) and:

1. **Detects** every person in the frame using YOLOv8
2. **Tracks** them across frames with persistent IDs (DeepSORT)
3. **Recognizes** known faces against a pre-registered database (DeepFace)
4. **Estimates** body pose and skeleton joints (MediaPipe)
5. **Classifies** actions — standing, walking, bending, running, hiding objects, loitering
6. **Scores** behavior suspicion (0–100) by combining 5 weighted signals
7. **Alerts** in real-time with sound, visual overlays, and evidence video clips

---

## 🔑 Key Features

| Feature | Description |
|---------|-------------|
| **Person Detection & Tracking** | YOLOv8 + DeepSORT with trajectory paths and speed analysis |
| **Face Recognition** | DeepFace-based identification from a `known_faces/` database |
| **Pose Estimation** | 33-joint skeleton via MediaPipe for gesture analysis |
| **Action Classification** | LSTM model with heuristic fallback (works without training) |
| **Behavior Scoring Engine** | Multi-signal consensus: action (35%) + trajectory (25%) + pose (20%) + zone (15%) + time (5%) |
| **Zone Monitoring** | Define restricted/inventory/exit areas — detects entry, exit, lingering |
| **False Positive Reduction** | Temporal smoothing + persistence check + multi-signal consensus |
| **Real-time Alerts** | Sound + visual alerts with configurable severity thresholds |
| **Evidence Recording** | Auto-saves 20s video clips (10s before + 10s after alert) with metadata |
| **Web Dashboard** | Live video feed, alert panel, person tracking, behavior scores at `http://localhost:8000` |
| **Frame Preprocessing** | CLAHE + auto-gamma correction for low-light environments |

---

## 📁 Project Structure

```
Behaviour Detection/
├── main.py                  # Main pipeline — runs everything
├── config.py                # All settings in one place
├── requirements.txt         # Python dependencies
├── modules/
│   ├── preprocessor.py      # Frame lighting normalization
│   ├── detector.py          # YOLOv8 person detection
│   ├── tracker.py           # DeepSORT multi-person tracking
│   ├── recognizer.py        # DeepFace face recognition
│   ├── pose_estimator.py    # MediaPipe skeleton extraction
│   ├── action_classifier.py # LSTM + heuristic action classification
│   ├── behavior_engine.py   # Multi-signal behavior scoring
│   ├── zone_manager.py      # Restricted zone monitoring
│   ├── alert_system.py      # Alert triggering & notifications
│   └── evidence_recorder.py # Video clip capture on alerts
├── dashboard/
│   ├── app.py               # FastAPI backend + WebSocket
│   └── static/
│       ├── index.html       # Dashboard UI
│       ├── style.css         # Dark theme styling
│       └── app.js           # Frontend logic
├── training/
│   └── train_action_model.py # LSTM training pipeline
├── known_faces/             # Face images for recognition
│   └── <person_name>/
│       └── photo.jpg
└── evidence/                # Auto-saved alert video clips
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Run the System

```bash
# Run with default webcam
python main.py

# Run with a video file
python main.py --source path/to/video.mp4

# Run without dashboard
python main.py --no-dashboard
```

### 3. Open Dashboard

Navigate to **http://localhost:8000** in your browser for the live monitoring dashboard.

### 4. Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the system |
| `r` | Register a face from the live feed |
| `z` | Toggle zone overlay |
| `s` | Toggle skeleton drawing |

---

## 👤 Register Known Faces

Create a subfolder inside `known_faces/` with the person's name, and add face photos:

```
known_faces/
├── Ahmad/
│   ├── photo1.jpg
│   └── photo2.jpg
├── Employee1/
│   └── face.jpg
```

> **Tip:** Add 2-3 photos per person from different angles for better accuracy.

---

## 🧠 How Behavior Scoring Works

The system evaluates each tracked person using **5 signals**, combined into a 0–100 score:

```
Behavior Score = Action (35%) + Trajectory (25%) + Pose (20%) + Zone (15%) + Time (5%)
```

| Score Range | Alert Level | What Happens |
|-------------|-------------|--------------|
| 0–39 | None | Normal activity |
| 40–64 | 🟡 Low | Logged, mild visual indicator |
| 65–84 | 🟠 Medium | Sound alert + dashboard notification |
| 85–100 | 🔴 High | Full alert + evidence video recording |

### False Positive Reduction
- **Multi-signal consensus** — Single suspicious signal won't trigger high alerts
- **Temporal smoothing** — Score averaged over a sliding window
- **Persistence check** — Must stay suspicious for several consecutive frames
- **Cooldown** — Same person can't re-trigger alerts within 30 seconds

---

## 🏋️ Train Custom Action Model

The heuristic classifier works out of the box. For better accuracy, train the LSTM on your own data:

```bash
# Step 1: Collect labeled data from a video
python training/train_action_model.py --collect --source 0

# Step 2: Train the model
python training/train_action_model.py --train --data training/collected_data.npz

# Step 3: Evaluate
python training/train_action_model.py --evaluate --data training/collected_data.npz
```

---

## ⚙️ Configuration

All settings are centralized in **`config.py`**. Key parameters:

| Setting | Default | Description |
|---------|---------|-------------|
| `video_source` | `0` | Webcam index or video file path |
| `detection.confidence` | `0.5` | YOLOv8 detection threshold |
| `behavior.alert_threshold_high` | `85.0` | High alert trigger score |
| `alert.cooldown_seconds` | `30.0` | Cooldown between alerts for same person |
| `evidence.pre_buffer_seconds` | `10.0` | Seconds recorded before alert |

---

## 🛠️ Tech Stack

- **YOLOv8** — Person Detection (Ultralytics)
- **DeepSORT** — Multi-Person Tracking
- **DeepFace** — Face Recognition (ArcFace model)
- **MediaPipe** — Pose Estimation (33 landmarks)
- **PyTorch** — LSTM Action Classifier
- **FastAPI** — Dashboard Backend + WebSocket
- **OpenCV** — Video Processing & Frame Rendering

---

## 📄 License

This project is for educational and research purposes.

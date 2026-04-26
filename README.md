# 🎯 Adaptive Trajectory Tracker & Visual Odometry

A unified computer vision web application that **automatically detects camera stability** and switches between two pipelines:

- **📹 Stable Camera → Object Tracking** — Detects and traces moving objects using classical CV (MOG2, Kalman Filter, Hungarian Algorithm).
- **🚗 Moving Camera → Visual Odometry** — Maps the camera's trajectory through the environment using Essential Matrix decomposition.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run object_tracking/app.py
```

Upload any video and the engine adapts automatically.

---

## 📂 Project Structure

```
├── object_tracking/             # Adaptive Tracker App (main)
│   ├── app.py                  # Streamlit entry point
│   ├── __init__.py             # Package exports
│   ├── background_subtractor.py # MOG2 background model
│   ├── contour_finder.py       # Contour detection
│   ├── detector.py             # Object detection pipeline
│   ├── disjoint_set.py         # Union-Find
│   ├── hungarian.py            # Hungarian assignment algorithm
│   ├── kalman_tracker.py       # Kalman filter
│   ├── multi_tracker.py        # Orchestrator
│
├── visual_odometry/             # Stereo/Mono Visual Odometry modules
│   ├── app.py                  # Standalone Mono VO Streamlit app
│   ├── calibration.py          # KITTI calibration loader
│   ├── disparity.py            # Stereo disparity
│   ├── feature_tracker.py      # Feature tracking
│   ├── image_loader.py         # Image pair loader
│   └── pose_estimator.py       # Pose estimation (PnP + RANSAC)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 How It Works

### Camera Motion Detection
The app computes **median optical flow displacement** between consecutive frames. If the median exceeds a threshold (1.5px), the camera is classified as **moving**.

### Stable Camera: Object Tracking Pipeline

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | **MOG2 Background Subtraction** | Separates moving foreground from static background |
| 2 | **Morphological Operations** | Cleans noise and fills gaps in foreground mask |
| 3 | **Contour Detection** | Extracts bounding boxes and centroids |
| 4 | **Kalman Filter** | Predicts each object's next position |
| 5 | **Hungarian Algorithm** | Optimally matches detections to existing tracks |

### Moving Camera: Visual Odometry Pipeline

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | **Good Features to Track** | Detects strong corner features |
| 2 | **Lucas-Kanade Optical Flow** | Tracks features across frames |
| 3 | **Essential Matrix (RANSAC)** | Estimates relative camera rotation & translation |
| 4 | **Pose Recovery** | Decomposes Essential Matrix into R and t |
| 5 | **Trajectory Accumulation** | Builds global camera path from incremental poses |

---

## 📥 Downloads

After processing completes, the app provides downloadable trajectory maps:
- **🔴 Camera Moving Map** — Final VO trajectory path
- **🟢 All Objects Stable Map** — Accumulated object trajectories with ID labels

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV** — All CV computations
- **NumPy** — Linear algebra
- **Streamlit** — Web UI

---

## 📄 License

MIT License

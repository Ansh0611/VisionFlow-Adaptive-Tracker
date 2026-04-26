# VisionFlow: Adaptive Trajectory Tracker 🎯

VisionFlow is a dual-engine computer vision system designed to track motion in any video environment. It intelligently switches between **Multi-Object Tracking** (for stable cameras) and **Visual Odometry** (for moving cameras).

---

## 🚀 Main Idea
The core challenge in computer vision is that tracking objects requires a stable background, while tracking camera motion requires a dynamic one. VisionFlow solves this by implementing an **Adaptive Engine** that:
1.  **Analyzes Camera Motion**: Constantly monitors the median displacement of features.
2.  **Stable Mode**: If the camera is still, it detects and tracks individual moving objects (people, cars, etc.).
3.  **Moving Mode**: If the camera moves, it switches to Visual Odometry to map the camera's own path in 3D space.

---

## 🛠 The Pipelines

### 1. Multi-Object Tracking (STABLE Mode)
Used when the camera is stationary to extract trajectories of moving entities.

**The Pipeline:**
-   **Background Subtraction (MOG2)**: Uses a Mixture of Gaussians model to identify "foreground" pixels that differ from the learned static background.
-   **Morphology**: Blurs and dilates the foreground mask to fill holes and remove noise.
-   **Contour Detection**: Groups foreground pixels into bounding boxes.
-   **Data Association (Hungarian Algorithm)**: Matches new detections to existing tracks by minimizing the cost (distance) between predicted and actual positions.
-   **State Estimation (Kalman Filter)**: Predicts the next position of each object based on its velocity, smoothing out noisy detections.

**Key Math:**
-   **Kalman Update**: $x_{k} = A x_{k-1} + w_{k-1}$ (Prediction) and $z_k = H x_k + v_k$ (Measurement).
-   **Cost Matrix**: Euclidean distance between predicted $(x_p, y_p)$ and detected $(x_d, y_d)$ centers.

### 2. Monocular Visual Odometry (MOVING Mode)
Used when the camera is in motion to estimate the camera's trajectory.

**The Pipeline:**
-   **Feature Detection**: Uses `goodFeaturesToTrack` (Shi-Tomasi) to find stable corners in the image.
-   **Feature Tracking**: Uses **Lucas-Kanade Optical Flow** to follow those points across frames.
-   **Essential Matrix Estimation**: Computes the geometric relationship between two camera views using the **5-point algorithm** inside a RANSAC loop.
-   **Pose Recovery**: Decomposes the Essential Matrix into a Rotation ($R$) and Translation ($t$).

**Key Math:**
-   **Essential Matrix ($E$)**: $E = [t]_{\times}R$, where $x'^T E x = 0$ for matching points $x, x'$.
-   **Pose Accumulation**: $T_{total} = T_{total} + R_{total} \cdot t_{new}$ and $R_{total} = R_{total} \cdot R_{new}$.

---

## 🖥 User Interface (Streamlit)
The application provides a premium dark-themed web interface:
-   **Live Feed**: Side-by-side view of the annotated video and the generated trajectory map.
-   **Adaptive HUD**: Real-time status showing if the engine is in "STABLE" or "MOVING" mode.
-   **Downloads**: Export the final generated maps as PNG images.

---

## 📦 Requirements
-   **Python 3.8+**
-   **OpenCV**: Core vision processing.
-   **Streamlit**: Web interface.
-   **NumPy/SciPy**: Heavy-duty mathematics and matrix operations.
-   **Pillow**: Image encoding for live streaming.

---

## 📖 How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

# VisionFlow: Adaptive Trajectory Tracker 🎯

VisionFlow is an intelligent motion analysis system that works like a "digital eye." It can tell the difference between objects moving in front of it and when the camera itself is being moved. It automatically switches its internal math to give you the most accurate trajectory map possible.

---

## 🧠 The "Main Idea" (The Brain)
Most tracking systems fail if you move the camera because they get confused by the background moving. VisionFlow uses a **Decision Engine**:
- **Step 1:** It looks at 200 tiny "landmarks" in every frame.
- **Step 2:** It measures how much those landmarks moved on average.
- **Step 3:** 
    - If motion is **Low (< 3 pixels)**: It assumes the camera is on a tripod and starts tracking objects.
    - If motion is **High (> 3 pixels)**: It assumes someone is carrying the camera and starts mapping the environment.

---

## 🏎️ Pipeline 1: Object Tracking (The "Stable" View)
When the camera is still, VisionFlow acts like a security guard, watching for any changes in the scene.

### The 5-Step Pipeline:
1.  **Learning the Background (MOG2)**: The computer builds a "memory" of the static scene. Anything that stays still for a while becomes the background.
2.  **Highlighting the New (Thresholding)**: It subtracts the "memory" from the "current frame." The leftovers are moving objects.
3.  **Cleaning the Image (Morphology)**: It uses "Blur" to remove tiny noise and "Dilation" to make the detected objects solid so they don't look like broken pieces.
4.  **Finding Centers (Contour Analysis)**: It draws a box around the moving pixels and calculates the exact center point ($X, Y$).
5.  **Smart Matching (Kalman + Hungarian)**:
    - **The Hungarian Algorithm**: If there are 5 people on screen, it makes sure "Person A" in Frame 1 is the same as "Person A" in Frame 2 by calculating the shortest distance between them.
    - **The Kalman Filter**: If a person walks behind a tree for a second, this math "guesses" where they should be based on their previous speed and direction.

### 📐 The Simple Math:
- **Euclidean Distance**: $d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$. Used to find which detection is closest to which existing track.
- **Velocity Prediction**: $New\_Position = Old\_Position + Velocity \times Time$. The Kalman filter uses this to keep trajectories smooth even when detections are jittery.

---

## 📍 Pipeline 2: Visual Odometry (The "Moving" View)
When you move the camera, the system stops looking for "objects" and starts looking at the "world" to see where *you* are going.

### The 4-Step Pipeline:
1.  **Finding Landmarks**: It finds "sharp" points (corners, edges) that are easy to follow.
2.  **Feature Tracking (Optical Flow)**: It watches those points move from Frame A to Frame B. If a point was at $(10, 10)$ and moved to $(12, 10)$, it knows the camera tilted.
3.  **The "5-Point" Geometry**: By looking at just 5 points moving together, the system can solve a massive 3D puzzle to figure out the camera's rotation and forward/backward movement.
4.  **Path Accumulation**: It adds up all these tiny micro-movements to draw your total path on a 2D map.

### 📐 The Simple Math:
- **The Essential Matrix ($E$)**: This is like a "translation book" that converts pixel movements into real-world 3D rotation ($R$) and translation ($t$).
- **Pose Update**: $Location = Location + (Direction \times Distance)$. Every frame, we update your coordinate on the map.

---

## 🛠️ Technical Implementation
- **Streamlit**: Handles the "Web App" part—turning Python code into a beautiful dashboard.
- **OpenCV**: The "Heavy Lifter" for all the image processing and geometry.
- **JPEG Streaming**: We convert frames to JPEG on-the-fly. This makes the "Live Feed" work over the internet because JPEGs are much smaller than raw video data.
- **Adaptive Canvas**: The map on the right side of the screen automatically "shifts" if you move outside of its borders, ensuring you never go off-map.

---

## 🚀 How to Run Locally
1.  Open your terminal.
2.  Type `pip install -r requirements.txt`.
3.  Type `streamlit run app.py`.
4.  Upload any video and watch the "Adaptive Engine" think!

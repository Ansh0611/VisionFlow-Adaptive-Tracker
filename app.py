"""
Adaptive Trajectory Tracker — Streamlit Web Application
===============================================
Unified Live Application:
  - Automatically detects camera motion.
  - If Camera STABLE: Runs classical CV multi-object tracking (MOG2).
  - If Camera MOVING: Runs Monocular Visual Odometry to map camera path.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys

# Ensure the root directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from object_tracking import MultiObjectTracker, KalmanTrack

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Trajectory Tracker",
    page_icon="🎯",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}

h1 {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.2rem !important;
}

h2, h3 {
    color: #94a3b8 !important;
    font-weight: 500;
}

.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #818cf8) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    font-size: 1.05rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5) !important;
}

.stat-card {
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stat-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 4px;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(56, 189, 248, 0.3) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Monocular Visual Odometry Engine ────────────────────────
class MonoVisualOdometry:
    def __init__(self, focal, pp):
        self.focal = focal
        self.pp = pp
        self.R_total = np.eye(3)
        self.t_total = np.zeros((3, 1))
        self.prev_gray = None
        self.prev_pts = None
        self.frame_idx = 0
    
    def process_frame(self, gray_img):
        if self.frame_idx == 0:
            self.prev_pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            self.prev_gray = gray_img
            self.frame_idx += 1
            return self.t_total.ravel().copy(), None, None
        
        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            if self.prev_pts is None or len(self.prev_pts) < 10:
                return self.t_total.ravel().copy(), None, None

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_img, self.prev_pts, None)
        
        if curr_pts is None or status is None:
            return self.t_total.ravel().copy(), None, None
            
        good_old = self.prev_pts[status == 1]
        good_new = curr_pts[status == 1]
        
        if len(good_old) < 15:
            self.prev_pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            self.prev_gray = gray_img
            return self.t_total.ravel().copy(), None, None
            
        E, mask = cv2.findEssentialMat(
            good_new, good_old, 
            focal=self.focal, pp=self.pp, 
            method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        
        if E is not None and E.shape == (3, 3):
            _, R, t, mask_pose = cv2.recoverPose(E, good_new, good_old, focal=self.focal, pp=self.pp, mask=mask)
            scale = 1.0 
            self.t_total = self.t_total + self.R_total @ (t * scale)
            self.R_total = self.R_total @ R

        if len(good_new) < 200:
            self.prev_pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        else:
            self.prev_pts = good_new.reshape(-1, 1, 2)
            
        self.prev_gray = gray_img
        self.frame_idx += 1
        
        return self.t_total.ravel().copy(), good_old, good_new

# ── Camera Motion Detector ──────────────────────────────────
class CameraMotionDetector:
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.prev_gray = None
        self.prev_pts = None

    def is_moving(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            return False, 0.0

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            if self.prev_pts is None or len(self.prev_pts) < 10:
                self.prev_gray = gray
                return False, 0.0

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
        
        if curr_pts is None or status is None:
            self.prev_gray = gray
            return False, 0.0

        good_old = self.prev_pts[status == 1]
        good_new = curr_pts[status == 1]
        
        if len(good_old) == 0:
            self.prev_gray = gray
            return False, 0.0

        displacements = np.linalg.norm(good_new - good_old, axis=1)
        median_disp = float(np.median(displacements))
        
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)

        return median_disp > self.threshold, median_disp


# ── Header ───────────────────────────────────────────────────
st.title("🎯 Adaptive Tracker & VO")
st.markdown(
    "<p style='color: #94a3b8; font-size: 1.1rem; margin-top: -10px;'>"
    "Upload any video. If the camera is <b>stable</b>, it detects and tracks moving objects. "
    "If the camera is <b>moving</b>, it maps the camera's trajectory environment.</p>",
    unsafe_allow_html=True
)

# ── Default Parameters ──────────────────────────
min_area = 500
max_area = 100000
max_distance = 150
max_missed = 15
bg_history = 500
var_threshold = 50
trail_length = 0
frame_skip = 1

# ── Main Area ────────────────────────────────────────────────
uploaded = st.file_uploader("📁 Drop your video here", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    col_info = st.columns(4)
    with col_info[0]:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{total_frames}</div>"
                    f"<div class='stat-label'>Total Frames</div></div>", unsafe_allow_html=True)
    with col_info[1]:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{fps:.1f}</div>"
                    f"<div class='stat-label'>FPS</div></div>", unsafe_allow_html=True)
    with col_info[2]:
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{width}×{height}</div>"
                    f"<div class='stat-label'>Resolution</div></div>", unsafe_allow_html=True)
    with col_info[3]:
        duration = total_frames / fps if fps > 0 else 0
        st.markdown(f"<div class='stat-card'><div class='stat-number'>{duration:.1f}s</div>"
                    f"<div class='stat-label'>Duration</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    start_col1, _, _ = st.columns([1, 1, 3])
    with start_col1:
        start = st.button("▶️ Start Adaptive Engine", use_container_width=True)

    if start:
        st.markdown("### 🎬 Live Feed: Adaptive Architecture")
        progress_bar = st.progress(0, text="Initializing Engine...")
        img_display = st.empty()

        # Engine Init
        KalmanTrack.reset_id_counter()
        tracker = MultiObjectTracker(
            min_area=min_area, max_area=max_area,
            max_distance=max_distance, max_missed=max_missed,
            history=bg_history, var_threshold=var_threshold,
            trail_length=trail_length
        )
        
        focal = float(width)
        pp = (width/2.0, height/2.0)
        vo = MonoVisualOdometry(focal, pp)
        motion_detector = CameraMotionDetector(threshold=1.5)

        cap = cv2.VideoCapture(tfile.name)

        # Unified Display Config
        traj_h = height
        traj_w = height  
        scale_draw_vo = 10.0
        
        vo_canvas = np.zeros((traj_h, traj_w, 3), dtype=np.uint8)
        cx_canvas, cy_canvas = traj_w//2, traj_h - 100

        # Accumulating canvas for all stable object tracks
        global_obj_canvas = np.zeros((traj_h, traj_w, 3), dtype=np.uint8)
        cv2.putText(global_obj_canvas, "All Object Trajectories", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        labeled_ids = set()

        frame_idx = 0
        max_active = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_skip > 1 and frame_idx % frame_skip != 0:
                continue

            # 1. Detect Motion
            is_moving, motion_mag = motion_detector.is_moving(frame)

            annotated = frame.copy()
            current_canvas = np.zeros_like(vo_canvas) 

            if is_moving:
                # ── MOVING CAMERA MODE ──
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pos, p_old, p_new = vo.process_frame(gray)
                
                if p_new is not None:
                    for p in p_new:
                        x, y = int(p[0]), int(p[1])
                        cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

                draw_x = int(cx_canvas + pos[0] * scale_draw_vo)
                draw_y = int(cy_canvas - pos[2] * scale_draw_vo)

                # Auto shift canvas if VO goes out of bounds
                margin = 50
                shift_x = shift_y = 0
                if draw_x < margin: shift_x = margin + 50 - draw_x
                elif draw_x > traj_w - margin: shift_x = (traj_w - margin - 50) - draw_x
                
                if draw_y < margin: shift_y = margin + 50 - draw_y
                elif draw_y > traj_h - margin: shift_y = (traj_h - margin - 50) - draw_y
                
                if shift_x != 0 or shift_y != 0:
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    vo_canvas = cv2.warpAffine(vo_canvas, M, (traj_w, traj_h))
                    cx_canvas += shift_x
                    cy_canvas += shift_y
                    draw_x += int(shift_x)
                    draw_y += int(shift_y)

                draw_x = np.clip(draw_x, 2, traj_w-2)
                draw_y = np.clip(draw_y, 2, traj_h-2)

                if vo.frame_idx <= 2:
                    cv2.circle(vo_canvas, (draw_x, draw_y), 6, (0, 255, 0), 2)
                    cv2.putText(vo_canvas, 'START', (draw_x+8, draw_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                
                # Draw VO Path
                cv2.circle(vo_canvas, (draw_x, draw_y), 2, (100, 255, 100), -1)

                current_canvas = vo_canvas.copy()
                cv2.putText(current_canvas, "Camera Trajectory", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(current_canvas, f"X:{pos[0]:.2f} Z:{pos[2]:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)

                hud_status = f"MOVING ({motion_mag:.1f}px)"
                hud_color = (0, 100, 255) 
                
                # Reset Tracker to avoid MOG2 contamination during movement
                tracker.reset()

            else:
                # ── STABLE CAMERA MODE ──
                annotated, _, stats = tracker.process_frame(frame)
                max_active = max(max_active, stats['active_tracks'])

                obj_canvas = np.zeros((traj_h, traj_w, 3), dtype=np.uint8)
                scale_x = traj_w / width
                scale_y = traj_h / height

                for t in tracker.tracks:
                    trail = t.trajectory
                    if len(trail) >= 2:
                        pts = [(int(p[0]*scale_x), int(p[1]*scale_y)) for p in trail]
                        cv2.polylines(obj_canvas, [np.array(pts)], False, t.color, 1, cv2.LINE_AA)
                        
                        # Accumulate the latest segment on the global map
                        latest_pts = [(int(p[0]*scale_x), int(p[1]*scale_y)) for p in trail[-2:]]
                        cv2.polylines(global_obj_canvas, [np.array(latest_pts)], False, t.color, 1, cv2.LINE_AA)
                        
                        # Add ID label to global map once it has enough points
                        if t.track_id not in labeled_ids and len(trail) > 10:
                            label_pos = (int(trail[0][0]*scale_x), int(trail[0][1]*scale_y))
                            cv2.putText(global_obj_canvas, f"ID:{t.track_id}", (label_pos[0]+5, label_pos[1]-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, t.color, 1, cv2.LINE_AA)
                            labeled_ids.add(t.track_id)
                        
                    if len(trail) > 0:
                        cp = trail[-1]
                        cpt = (int(cp[0]*scale_x), int(cp[1]*scale_y))
                        cv2.circle(obj_canvas, cpt, 3, t.color, -1)

                current_canvas = obj_canvas.copy()
                cv2.putText(current_canvas, "Object Trajectories", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(current_canvas, f"Active: {stats['active_tracks']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 1)

                hud_status = f"STABLE ({motion_mag:.1f}px)"
                hud_color = (100, 255, 100) 
                
            # Global HUD
            cv2.rectangle(annotated, (0,0), (300, 80), (0,0,0), -1)
            cv2.putText(annotated, hud_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hud_color, 2)
            cv2.putText(annotated, f"Frame {frame_idx}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Frame Composite
            vid_resized = cv2.resize(annotated, (int(width * traj_h / height), traj_h))
            combined = np.hstack([vid_resized, current_canvas])
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            
            img_display.image(combined_rgb, channels="RGB", use_container_width=True)
            progress_bar.progress(min(frame_idx / total_frames, 1.0), text=f"Rendering: {frame_idx}/{total_frames}")

        cap.release()
        os.remove(tfile.name)
        progress_bar.progress(1.0, text="✅ Adaptive Tracking Complete!")

        st.markdown("---")
        st.markdown("### 📥 Download Trajectory Maps")
        col1, col2 = st.columns(2)
        
        ret_vo, buffer_vo = cv2.imencode('.png', vo_canvas)
        if ret_vo:
            col1.download_button(
                label="🔴 Download Camera Moving Map",
                data=buffer_vo.tobytes(),
                file_name="camera_moving_map.png",
                mime="image/png",
                use_container_width=True
            )
            
        ret_obj, buffer_obj = cv2.imencode('.png', global_obj_canvas)
        if ret_obj:
            col2.download_button(
                label="🟢 Download All Objects Stable Map",
                data=buffer_obj.tobytes(),
                file_name="all_objects_stable_map.png",
                mime="image/png",
                use_container_width=True
            )

else:
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 3rem 1rem;'>
        <p style='font-size: 4rem; margin-bottom: 0;'>📹</p>
        <p style='color: #64748b; font-size: 1.2rem;'>
            Upload a video to begin tracking.<br>
            The engine automatically switches between <b>Object Tracking</b> and <b>Camera Odometry</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

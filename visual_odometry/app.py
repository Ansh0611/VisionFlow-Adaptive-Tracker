import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(layout="wide", page_title="Monocular Visual Odometry")

st.markdown("""
<style>
.main { background-color: #0f172a; color: white; }
h1, h2, h3 { color: #38bdf8; font-family: 'Inter', sans-serif; }
.stButton>button { background-color: #38bdf8; color: #0f172a; border-radius: 8px; font-weight: bold; border: none; padding: 0.5rem 2rem; }
.stButton>button:hover { background-color: #0ea5e9; color: white; }
.upload-text { padding: 10px; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

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
            # Standard monocular lacking scale requires arbitrary constant forward acceleration 
            # Or we assume t is unit vector and multiply by a constant scale factor
            scale = 1.0 
            # When camera moves forward, the calculated t typically points toward +Z 
            # (or -Z depending on coordinate convention, here we just accumulate raw and render)
            self.t_total = self.t_total + self.R_total @ (t * scale)
            self.R_total = self.R_total @ R

        if len(good_new) < 200:
            self.prev_pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        else:
            self.prev_pts = good_new.reshape(-1, 1, 2)
            
        self.prev_gray = gray_img
        self.frame_idx += 1
        
        return self.t_total.ravel().copy(), good_old, good_new

st.title("📹 Monocular Visual Odometry Web App")
st.markdown("<div class='upload-text'>Upload a standard MP4 video to process its trajectory seamlessly on the web relying natively on Essential Matrices to replace Stereo configurations!</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Select Video", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    tfile.close()
    
    col1, col2 = st.columns([1, 4])
    with col1:
        start = st.button("Start Tracking")
    
    img_placeholder = st.empty()
    
    if start:
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Estimate Generic Intrinsics for Arbitrary Uncalibrated Video 
        focal = float(width) # Simple fallback heuristic
        pp = (width/2.0, height/2.0)
        
        vo = MonoVisualOdometry(focal, pp)
        
        # Canvas matching UI bounds
        canvas_dim = 600
        traj_canvas = np.zeros((canvas_dim, canvas_dim, 3), dtype=np.uint8)
        cx_canvas, cy_canvas = canvas_dim//2, canvas_dim - 100
        scale_draw = 10.0 # Arbitrary visual scale since Monocular has unit scale
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pos, p_old, p_new = vo.process_frame(gray)
            
            # Decorate the camera feed
            if p_new is not None:
                for p in p_new:
                    x, y = int(p[0]), int(p[1])
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                    
            # ── Draw live top-down trajectory map (X-Z view usually)
            # In Mono recoverPose, forward motion is typically +Z. 
            draw_x = int(cx_canvas + pos[0] * scale_draw)
            draw_y = int(cy_canvas - pos[2] * scale_draw)
            
            margin = 50
            shift_x = 0
            shift_y = 0
            if draw_x < margin: shift_x = margin + 100 - draw_x
            elif draw_x > canvas_dim - margin: shift_x = (canvas_dim - margin - 100) - draw_x
            
            if draw_y < margin: shift_y = margin + 100 - draw_y
            elif draw_y > canvas_dim - margin: shift_y = (canvas_dim - margin - 100) - draw_y
            
            if shift_x != 0 or shift_y != 0:
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                traj_canvas = cv2.warpAffine(traj_canvas, M, (canvas_dim, canvas_dim))
                cx_canvas += shift_x
                cy_canvas += shift_y
                draw_x += int(shift_x)
                draw_y += int(shift_y)
                
            draw_x = np.clip(draw_x, 2, canvas_dim-2)
            draw_y = np.clip(draw_y, 2, canvas_dim-2)
            
            if vo.frame_idx <= 2:
                cv2.circle(traj_canvas, (draw_x, draw_y), 6, (0, 255, 0), 2)
                cv2.putText(traj_canvas, 'START', (draw_x+8, draw_y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            
            # Draw trail
            cv2.circle(traj_canvas, (draw_x, draw_y), 2, (38, 189, 248), -1)

            # Standardise visual alignment 
            disp_h = canvas_dim
            disp_w = int(width * (disp_h / height))
            resized_frame = cv2.resize(frame, (disp_w, disp_h))
            
            combined = np.hstack([resized_frame, traj_canvas])
            
            # Decorate global display overlays
            cv2.putText(combined, "Live Monocular Feed", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(combined, f"Frame: {vo.frame_idx}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
            cv2.putText(combined, "Live Geometric Map", (disp_w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(combined, f"X:{pos[0]:.2f} Z:{pos[2]:.2f}", (disp_w + 20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,255), 2)
            
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            img_placeholder.image(combined_rgb, use_container_width=True)

        cap.release()
        os.remove(tfile.name)
        st.success("✅ Tracking successfully computed across the entire video timeline!")

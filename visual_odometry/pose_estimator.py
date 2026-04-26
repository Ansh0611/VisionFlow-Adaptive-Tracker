# stereo_vo/pose_estimator.py
import cv2
import numpy as np

print("pose_estimator.py loaded - version DEBUG")   # confirm file is fresh


def lift_to_3d(pts_2d_prev, pts_2d_curr, depth_map, calib):
    fx = calib['fx']
    cx = calib['cx']
    cy = calib['cy']
    h, w = depth_map.shape

    pts_3d_list = []
    pts_2d_curr_list = []

    pts_prev_flat = pts_2d_prev.reshape(-1, 2)
    pts_curr_flat = pts_2d_curr.reshape(-1, 2)

    for i in range(len(pts_prev_flat)):
        pt_prev = pts_prev_flat[i]
        pt_curr = pts_curr_flat[i]
        
        u, v = int(round(pt_prev[0])), int(round(pt_prev[1]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        Z = float(depth_map[v, u])
        if Z <= 0.5 or Z > 80.0:
            continue
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fx
        pts_3d_list.append([X, Y, Z])
        pts_2d_curr_list.append([pt_curr[0], pt_curr[1]])

    if len(pts_3d_list) < 6:
        return None, None

    pts3d = np.array(pts_3d_list, dtype=np.float32)
    pts2d_curr = np.array(pts_2d_curr_list, dtype=np.float32)

    # DEBUG — only print once
    if not hasattr(lift_to_3d, '_printed'):
        lift_to_3d._printed = True
        print(f"  [debug lift_to_3d] Z range: "
              f"{pts3d[:,2].min():.2f} -> {pts3d[:,2].max():.2f} m")
        print(f"  [debug lift_to_3d] X range: "
              f"{pts3d[:,0].min():.2f} -> {pts3d[:,0].max():.2f} m")
        print(f"  [debug lift_to_3d] sample pt: {pts3d[0]}")

    return pts3d, pts2d_curr


def estimate_pose(pts_3d, pts_2d_curr, calib):
    K = np.array([
        [calib['fx'],          0, calib['cx']],
        [          0, calib['fx'], calib['cy']],
        [          0,          0,           1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros(4)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d.astype(np.float64),
        pts_2d_curr.astype(np.float64),
        K,
        dist_coeffs,
        reprojectionError = 2.0,
        iterationsCount   = 200,
        confidence        = 0.999,
        flags = cv2.SOLVEPNP_ITERATIVE
    )

    if not success or inliers is None or len(inliers) < 10:
        return None, None, None

    R, _ = cv2.Rodrigues(rvec)

    return R, tvec, inliers


class VisualOdometry:

    MIN_FEATURES = 150

    def __init__(self, calib):
        self.calib      = calib
        self.R_total    = np.eye(3)
        self.t_total    = np.zeros((3, 1))
        self.prev_gray  = None
        self.prev_depth = None
        self.prev_pts   = None
        self.frame_idx  = 0
        self.trajectory = [np.zeros(3)]

    def process_frame(self, left_gray, depth_map):
        from feature_tracker import detect_features, track_features

        if self.frame_idx == 0:
            self.prev_gray  = left_gray
            self.prev_depth = depth_map
            self.prev_pts   = detect_features(left_gray)
            self.frame_idx  = 1
            return {
                'R': self.R_total.copy(), 't': self.t_total.copy(),
                'position': np.zeros(3), 'n_tracked': len(self.prev_pts),
                'n_inliers': 0, 'status': 'initialized'
            }

        curr_pts, prev_pts_good, _ = track_features(
            self.prev_gray, left_gray, self.prev_pts)

        if curr_pts is None or len(curr_pts) < 20:
            self.prev_gray  = left_gray
            self.prev_depth = depth_map
            self.prev_pts   = detect_features(left_gray)
            self.frame_idx += 1
            self.trajectory.append(self.t_total.ravel().copy())
            return {
                'R': self.R_total.copy(), 't': self.t_total.copy(),
                'position': self.t_total.ravel().copy(),
                'n_tracked': 0, 'n_inliers': 0, 'status': 'tracking_failed'
            }

        pts_3d, pts_2d_curr = lift_to_3d(
            prev_pts_good, curr_pts, self.prev_depth, self.calib)

        if pts_3d is None or len(pts_3d) < 10:
            self.prev_gray  = left_gray
            self.prev_depth = depth_map
            self.prev_pts   = curr_pts.reshape(-1, 1, 2)
            self.frame_idx += 1
            self.trajectory.append(self.t_total.ravel().copy())
            return {
                'R': self.R_total.copy(), 't': self.t_total.copy(),
                'position': self.t_total.ravel().copy(),
                'n_tracked': len(curr_pts), 'n_inliers': 0,
                'status': 'insufficient_depth'
            }

        R_rel, t_rel, inliers = estimate_pose(
            pts_3d, pts_2d_curr, self.calib)

        if R_rel is not None:
            # DEBUG — print t_rel for first 3 frames
            if self.frame_idx <= 3:
                print(f"  [debug frame {self.frame_idx}] "
                      f"t_rel={t_rel.ravel()}  "
                      f"mag={np.linalg.norm(t_rel):.4f}m")

            # PnP gives T_prev->curr (object to camera).
            # We want to update T_world->camera or T_camera->world.
            # R_total and t_total here represent camera pose in world coordinate frame.
            # So T_cam->world = T_prev_cam->world * T_curr->prev_cam
            # T_curr->prev_cam is the inverse of T_prev_cam->curr (which is R_rel, t_rel).
            # Inverse of (R_rel, t_rel) is (R_rel.T, -R_rel.T @ t_rel)
            R_inv = R_rel.T
            t_inv = -R_rel.T @ t_rel
            
            self.t_total = self.t_total + self.R_total @ t_inv
            self.R_total = self.R_total @ R_inv
            n_inliers = len(inliers)
            status = 'ok'
        else:
            n_inliers = 0
            status = 'pnp_failed'

        self.trajectory.append(self.t_total.ravel().copy())

        if len(curr_pts) < self.MIN_FEATURES:
            self.prev_pts = detect_features(left_gray)
        else:
            self.prev_pts = curr_pts.reshape(-1, 1, 2)

        self.prev_gray  = left_gray
        self.prev_depth = depth_map
        self.frame_idx += 1

        return {
            'R': self.R_total.copy(), 't': self.t_total.copy(),
            'position': self.t_total.ravel().copy(),
            'n_tracked': len(curr_pts),
            'n_inliers': n_inliers, 'status': status
        }

    def get_trajectory(self):
        return np.array(self.trajectory)
"""
Step 5 — Multi-Object Tracker (Orchestrator)
=============================================

This module ties together all previous components into a single coherent
pipeline that processes one frame at a time:

    Frame → Background Subtraction → Morphology + Contours → Detections
         → Kalman Predict → Hungarian Assign → Update/Create/Delete tracks

The tracker maintains a list of active KalmanTrack objects, each with its
own color-coded trajectory trail. It handles:

  - **Track birth:** when a new detection appears that doesn't match any
    existing track, a new KalmanTrack is spawned.
  - **Track death:** when a track goes un-matched for too many consecutive
    frames (max_missed), it is deleted.
  - **Track update:** when a detection matches, the Kalman Filter is updated
    with the measurement, and the trajectory grows.
"""

import cv2
import numpy as np
from typing import List, Tuple

from .background_subtractor import BackgroundSubtractor
from .detector import ObjectDetector, Detection
from .kalman_tracker import KalmanTrack
from .hungarian import hungarian_assign


class MultiObjectTracker:
    """
    Full multi-object tracking pipeline using classical CV.

    Parameters
    ----------
    min_area : int
        Minimum contour area for a valid detection.
    max_area : int
        Maximum contour area for a valid detection.
    max_distance : float
        Maximum Euclidean distance for Hungarian matching gate.
    max_missed : int
        Number of consecutive missed frames before a track is deleted.
    history : int
        MOG2 background model history length.
    var_threshold : float
        MOG2 variance threshold.
    trail_length : int
        Maximum number of past points to draw in the trajectory trail.
        0 = draw entire history.
    """

    def __init__(self, min_area=500, max_area=100000, max_distance=150,
                 max_missed=15, history=500, var_threshold=50,
                 trail_length=0):
        self.bg_sub  = BackgroundSubtractor(history=history,
                                            var_threshold=var_threshold)
        self.detector = ObjectDetector(min_area=min_area, max_area=max_area)
        self.max_distance = max_distance
        self.max_missed   = max_missed
        self.trail_length = trail_length

        self.tracks: List[KalmanTrack] = []
        self.frame_count = 0
        self.total_tracks_created = 0

    def process_frame(self, frame):
        """
        Process a single video frame through the full pipeline.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3) BGR image

        Returns
        -------
        annotated : np.ndarray
            The original frame with bounding boxes + trajectory trails drawn.
        mask_vis : np.ndarray
            The cleaned foreground mask (for debug / side-by-side view).
        stats : dict
            Frame-level statistics.
        """
        self.frame_count += 1

        # ── Step 1: Background subtraction ─────────────────
        fg_mask = self.bg_sub.apply(frame)

        # ── Step 2: Detect objects ─────────────────────────
        detections, cleaned_mask = self.detector.detect(fg_mask)
        det_centroids = [d.centroid for d in detections]

        # ── Step 3: Predict all active tracks forward ──────
        predictions = []
        for track in self.tracks:
            pred = track.predict()
            predictions.append(pred)

        # ── Step 4: Hungarian assignment ───────────────────
        matches, unmatched_tracks, unmatched_dets = hungarian_assign(
            predictions, det_centroids, self.max_distance
        )

        # ── Step 5a: Update matched tracks ─────────────────
        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(det_centroids[d_idx])

        # ── Step 5b: Mark missed tracks ────────────────────
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].mark_missed()

        # ── Step 5c: Spawn new tracks for unmatched detections ─
        for d_idx in unmatched_dets:
            new_track = KalmanTrack(det_centroids[d_idx])
            self.tracks.append(new_track)
            self.total_tracks_created += 1

        # ── Step 5d: Delete dead tracks ────────────────────
        self.tracks = [t for t in self.tracks if t.missed_frames <= self.max_missed]

        # ── Step 6: Draw annotations ──────────────────────
        annotated = frame.copy()
        self._draw_annotations(annotated, detections)

        # Make mask 3-channel for display
        mask_vis = cv2.cvtColor(cleaned_mask, cv2.COLOR_GRAY2BGR)

        stats = {
            'frame': self.frame_count,
            'active_tracks': len(self.tracks),
            'detections': len(detections),
            'total_tracks': self.total_tracks_created
        }

        return annotated, mask_vis, stats

    def _draw_annotations(self, frame, detections):
        """Draw bounding boxes, centroids, trajectory trails, and IDs."""

        # Draw detection bounding boxes (thin white)
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)

        # Draw each track's trajectory trail + current ID
        for track in self.tracks:
            trail = track.trajectory
            if self.trail_length > 0:
                trail = trail[-self.trail_length:]

            # Draw polyline trail
            if len(trail) >= 2:
                pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, track.color, 1,
                              cv2.LINE_AA)

            # Draw current position dot
            if len(trail) > 0:
                cx, cy = trail[-1]
                cv2.circle(frame, (cx, cy), 3, track.color, -1)

                # Draw ID label
                label = f"ID:{track.track_id}"
                cv2.putText(frame, label, (cx + 6, cy - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            track.color, 1, cv2.LINE_AA)

                # Draw velocity arrow
                vx, vy = track.get_velocity()
                speed = np.sqrt(vx**2 + vy**2)
                if speed > 1.0:
                    arrow_scale = 5.0
                    end_x = int(cx + vx * arrow_scale)
                    end_y = int(cy + vy * arrow_scale)
                    cv2.arrowedLine(frame, (cx, cy), (end_x, end_y),
                                   track.color, 1, tipLength=0.3)

    def reset(self):
        """Reset tracker state for a new video."""
        self.tracks.clear()
        self.frame_count = 0
        self.total_tracks_created = 0
        KalmanTrack.reset_id_counter()
        # Re-create background subtractor to clear learned background
        self.bg_sub = BackgroundSubtractor(
            history=self.bg_sub.mog2.getHistory(),
            var_threshold=self.bg_sub.mog2.getVarThreshold()
        )

"""
Step 3 — Kalman Filter for Single-Object State Estimation
=========================================================

Theory
------

The Kalman Filter is an optimal recursive estimator for linear dynamical
systems with Gaussian noise. It maintains a belief about an object's state
(position + velocity) and refines that belief each time a new measurement
(detection centroid) arrives.

### State Vector

We track each object with a 4D state:

    x = [x, y, ẋ, ẏ]ᵀ

Where (x, y) is the 2D position and (ẋ, ẏ) is the velocity.

### State Transition Model (constant-velocity assumption)

    x_{k} = F · x_{k-1} + w

    F = | 1  0  dt  0 |       w ~ N(0, Q)
        | 0  1  0  dt |
        | 0  0  1   0 |
        | 0  0  0   1 |

With dt = 1 frame interval. This says: position += velocity × dt.

### Measurement Model

We only observe position (from the detection centroid):

    z_k = H · x_k + v

    H = | 1  0  0  0 |       v ~ N(0, R)
        | 0  1  0  0 |

### Prediction Step

    x̂_{k|k-1} = F · x̂_{k-1|k-1}
    P_{k|k-1}  = F · P_{k-1|k-1} · Fᵀ + Q

This projects the state forward in time. P is the uncertainty covariance.

### Update Step (when a detection is matched)

    Innovation:              ỹ = z_k - H · x̂_{k|k-1}
    Innovation covariance:   S = H · P_{k|k-1} · Hᵀ + R
    Kalman Gain:             K = P_{k|k-1} · Hᵀ · S⁻¹
    Updated state:           x̂_{k|k} = x̂_{k|k-1} + K · ỹ
    Updated covariance:      P_{k|k} = (I - K · H) · P_{k|k-1}

The Kalman Gain K balances between trusting the prediction vs the measurement:
  - If R is large (noisy measurement) → K is small → trust prediction more
  - If Q is large (unpredictable motion) → K is large → trust measurement more

### Why Kalman Filter?

1. **Smoothing:** Reduces jitter from noisy detections.
2. **Prediction:** When an object is briefly occluded, the filter predicts
   where it should be, keeping the track alive.
3. **Velocity estimation:** Gives us ẋ, ẏ for free, enabling motion analysis.
"""

import cv2
import numpy as np
from typing import List, Tuple


class KalmanTrack:
    """
    A single tracked object with its own Kalman Filter instance.

    Attributes
    ----------
    track_id : int
        Unique identifier for this track.
    kf : cv2.KalmanFilter
        The OpenCV Kalman Filter object.
    trajectory : list of (int, int)
        History of all centroid positions for trail drawing.
    missed_frames : int
        Number of consecutive frames with no matched detection.
    color : tuple
        BGR color for drawing this track's trajectory.
    age : int
        Total number of frames this track has existed.
    """

    _next_id = 0

    def __init__(self, initial_centroid: Tuple[int, int],
                 process_noise: float = 1e-2,
                 measurement_noise: float = 1e-1):
        """
        Parameters
        ----------
        initial_centroid : (x, y)
            First detected position.
        process_noise : float
            Diagonal of Q matrix. Higher → model trusts measurements more.
        measurement_noise : float
            Diagonal of R matrix. Higher → model trusts predictions more.
        """
        self.track_id = KalmanTrack._next_id
        KalmanTrack._next_id += 1

        # ── Build the Kalman Filter ──────────────────────────
        # 4 state variables, 2 measurement variables, 0 control inputs
        self.kf = cv2.KalmanFilter(4, 2, 0)

        # State transition matrix F (constant velocity model, dt=1)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix H (we observe x, y only)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance Q
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise

        # Measurement noise covariance R
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Initial state covariance P₀ (high uncertainty at start)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

        # Initialize state: position = centroid, velocity = 0
        x0, y0 = initial_centroid
        self.kf.statePost = np.array([x0, y0, 0, 0], dtype=np.float32).reshape(4, 1)

        # Track metadata
        self.trajectory: List[Tuple[int, int]] = [(x0, y0)]
        self.missed_frames = 0
        self.age = 1

        # Assign a unique color using golden-ratio hue spacing for visual distinction
        hue = (self.track_id * 0.618033988749895) % 1.0
        rgb = self._hsv_to_bgr(hue, 0.9, 0.95)
        self.color = rgb

    def predict(self) -> Tuple[int, int]:
        """
        Predict the next state (runs the Kalman prediction step).

        Returns
        -------
        predicted_centroid : (int, int)
        """
        pred = self.kf.predict()
        px, py = int(pred[0, 0]), int(pred[1, 0])
        return (px, py)

    def update(self, centroid: Tuple[int, int]):
        """
        Update the Kalman state with a matched detection measurement.

        Parameters
        ----------
        centroid : (x, y)
            Measured position from this frame's detection.
        """
        measurement = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        self.kf.correct(measurement)

        self.trajectory.append(centroid)
        self.missed_frames = 0
        self.age += 1

    def mark_missed(self):
        """Called when no detection matched this track in the current frame."""
        self.missed_frames += 1
        self.age += 1
        # Use last predicted position for the trail
        pred = self.kf.statePost
        px, py = int(pred[0, 0]), int(pred[1, 0])
        self.trajectory.append((px, py))

    def get_velocity(self) -> Tuple[float, float]:
        """Return the current estimated velocity (ẋ, ẏ) in pixels/frame."""
        state = self.kf.statePost
        return float(state[2, 0]), float(state[3, 0])

    @staticmethod
    def _hsv_to_bgr(h, s, v):
        """Convert HSV [0-1] to BGR tuple [0-255]."""
        hsv_pixel = np.array([[[int(h * 179), int(s * 255), int(v * 255)]]],
                             dtype=np.uint8)
        bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
        return tuple(int(c) for c in bgr_pixel[0, 0])

    @staticmethod
    def reset_id_counter():
        """Reset the global ID counter (useful between videos)."""
        KalmanTrack._next_id = 0

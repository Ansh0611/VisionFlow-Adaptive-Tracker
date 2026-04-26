"""
Step 4 — Hungarian Algorithm (Kuhn-Munkres) for Optimal Assignment
==================================================================

Theory
------

### The Assignment Problem

At each frame we have:
  - N existing tracks with predicted positions: {p̂₁, p̂₂, ..., p̂_N}
  - M new detections with measured centroids:    {z₁, z₂, ..., z_M}

We need to figure out which detection belongs to which track.

### Cost Matrix

Build an N×M cost matrix C where:

    C_ij = ‖p̂_i - z_j‖₂   (Euclidean distance)

This measures how far track i's prediction is from detection j.

### Optimal Assignment

We want to find a one-to-one mapping (assignment) σ that minimizes the
total cost:

    min  Σ  C_{i, σ(i)}
     σ

This is the classical **Linear Assignment Problem** (LAP).

### Hungarian Algorithm (Kuhn, 1955 / Munkres, 1957)

Solves LAP optimally in O(n³) time via:
  1. Row reduction: subtract row minimum from each row
  2. Column reduction: subtract column minimum from each column
  3. Cover all zeros with minimum lines
  4. If #lines = n → optimal solution found
  5. Otherwise, adjust matrix and repeat from step 3

We use scipy.optimize.linear_sum_assignment which implements this.

### Distance Gating

After solving, we reject any assignment where C_ij > max_distance.
This prevents matching a track to a detection on the other side of the frame.

Assignments that survive the gate produce three groups:
  - **Matched pairs:** track i ↔ detection j  →  Kalman Update
  - **Unmatched tracks:** tracks with no detection  →  mark missed
  - **Unmatched detections:** detections with no track  →  spawn new track
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


def hungarian_assign(
    track_predictions: List[Tuple[int, int]],
    detection_centroids: List[Tuple[int, int]],
    max_distance: float = 150.0
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Solve the assignment problem between existing tracks and new detections.

    Parameters
    ----------
    track_predictions : list of (x, y)
        Predicted centroids from each track's Kalman Filter.
    detection_centroids : list of (x, y)
        Measured centroids from the current frame's detections.
    max_distance : float
        Maximum allowed distance for a valid match. Pairs exceeding
        this threshold are treated as unmatched.

    Returns
    -------
    matches : list of (track_idx, detection_idx)
        Successfully matched pairs.
    unmatched_tracks : list of int
        Track indices with no matching detection.
    unmatched_detections : list of int
        Detection indices with no matching track.
    """
    n_tracks = len(track_predictions)
    n_dets   = len(detection_centroids)

    # Edge cases
    if n_tracks == 0:
        return [], [], list(range(n_dets))
    if n_dets == 0:
        return [], list(range(n_tracks)), []

    # ── Build cost matrix ──────────────────────────────────
    cost = np.zeros((n_tracks, n_dets), dtype=np.float64)
    for i, (tx, ty) in enumerate(track_predictions):
        for j, (dx, dy) in enumerate(detection_centroids):
            cost[i, j] = np.sqrt((tx - dx) ** 2 + (ty - dy) ** 2)

    # ── Solve optimal assignment ───────────────────────────
    row_indices, col_indices = linear_sum_assignment(cost)

    # ── Apply distance gate ────────────────────────────────
    matches = []
    unmatched_tracks = set(range(n_tracks))
    unmatched_detections = set(range(n_dets))

    for r, c in zip(row_indices, col_indices):
        if cost[r, c] <= max_distance:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_detections.discard(c)

    return matches, sorted(unmatched_tracks), sorted(unmatched_detections)

"""
Step 2 — Object Detection via Morphological Operations + Contour Extraction
============================================================================

Theory
------

### 2a. Morphological Operations

Given a binary foreground mask A and a structuring element B (e.g. 5×5 ellipse):

**Erosion** — shrinks white regions, removes small noise:
    A ⊖ B = { z | B_z ⊆ A }
    A pixel survives ONLY if the entire kernel B fits inside the white region.

**Dilation** — expands white regions, fills small gaps:
    A ⊕ B = { z | (B_z ∩ A) ≠ ∅ }
    A pixel becomes white if ANY part of kernel B overlaps with white.

**Opening** = Erosion → Dilation:
    A ∘ B = (A ⊖ B) ⊕ B
    Removes small noise blobs while preserving object shape.

**Closing** = Dilation → Erosion:
    A • B = (A ⊕ B) ⊖ B
    Fills small holes inside objects while preserving shape.

Our pipeline: Open (kill noise) → Close (fill holes) → Dilate (merge fragments)

### 2b. Contour Extraction (Suzuki & Abe, 1985)

After cleaning the mask, we find external contours — the outermost border of
each connected white region. Each contour is a list of (x, y) points forming
a closed polygon.

For each contour we compute:
  - Area = number of pixels inside (used to filter tiny blobs)
  - Bounding rect = (x, y, w, h)  — axis-aligned box
  - Centroid = (cx, cy) = center of mass of the contour

We filter by minimum area to ignore noise and return clean detections.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    """A single detected object in one frame."""
    bbox: tuple       # (x, y, w, h)
    centroid: tuple    # (cx, cy)
    area: float        # contour area in pixels


class ObjectDetector:
    """Extracts object detections from a foreground mask."""

    def __init__(self, min_area: int = 500, max_area: int = 100000):
        """
        Parameters
        ----------
        min_area : int
            Minimum contour area (pixels²). Anything smaller is noise.
            For a shuttlecock this can be ~200; for cars ~2000.
        max_area : int
            Maximum contour area. Anything larger is likely a lighting
            change or camera shake, not a real object.
        """
        self.min_area = min_area
        self.max_area = max_area

        # Structuring elements for morphological operations
        # Ellipse kernel works better than rectangle for natural shapes
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, fg_mask) -> List[Detection]:
        """
        Clean the foreground mask and extract object detections.

        Parameters
        ----------
        fg_mask : np.ndarray (H, W) binary mask {0, 255}

        Returns
        -------
        detections : list[Detection]
        """
        # ── Morphological pipeline ─────────────────────────────
        # Step A: Opening — remove small noise specks
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel_open)

        # Step B: Closing — fill holes inside objects
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel_close)

        # Step C: Dilate — merge nearby fragments into one blob
        cleaned = cv2.dilate(cleaned, self.kernel_dilate, iterations=2)

        # ── Contour extraction ─────────────────────────────────
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)

            # Centroid via image moments
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                cx, cy = x + w // 2, y + h // 2
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            detections.append(Detection(
                bbox=(x, y, w, h),
                centroid=(cx, cy),
                area=area
            ))

        return detections, cleaned

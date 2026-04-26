"""
Contour Finder — detects moving objects in video frames.

Ported from ObjectTracker/src/tracker/contour_finder.cpp.

Pipeline:
  1. Background subtraction using MOG2 (Mixture of Gaussians)
  2. Thresholding + shadow removal
  3. Median blur (removes salt-and-pepper noise)
  4. Dilation (fills holes left by blur/shadow removal)
  5. Contour detection
  6. Size filtering (remove contours < threshold * max_area)
  7. Contour merging (union-find for nearby contours)
  8. Recompute mass centers and bounding boxes
"""

import cv2
import numpy as np
from collections import defaultdict
from .disjoint_set import DisjointSets


def _distance_between_rects(a, b):
    """Minimum distance between corners of two bounding rectangles."""
    corners_a = [
        (a[0], a[1]),
        (a[0], a[1] + a[3]),
        (a[0] + a[2], a[1]),
        (a[0] + a[2], a[1] + a[3]),
    ]
    corners_b = [
        (b[0], b[1]),
        (b[0], b[1] + b[3]),
        (b[0] + b[2], b[1]),
        (b[0] + b[2], b[1] + b[3]),
    ]
    min_dist = float('inf')
    for ca in corners_a:
        for cb in corners_b:
            d = np.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
            if d < min_dist:
                min_dist = d
    return min_dist


class ContourFinder:
    """Finds blobs representing moving objects using background subtraction."""

    def __init__(self,
                 history: int = 1000,
                 n_mixtures: int = 3,
                 contour_size_threshold: float = 0.1,
                 median_filter_size: int = 9,
                 contour_merge_threshold: float = 0.01):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=16,
            detectShadows=True
        )
        self.bg.setNMixtures(n_mixtures)
        self.bg.setShadowThreshold(0.7)

        self.contour_size_threshold = contour_size_threshold
        self.median_filter_size = median_filter_size
        self.contour_merge_threshold = contour_merge_threshold
        self.suppress_rectangles = []

    def find_contours(self, frame):
        h, w = frame.shape[:2]
        diagonal = np.sqrt(h * h + w * w)

        # 1. Background subtraction
        foreground = self.bg.apply(frame)

        # 2. Threshold to remove shadows (MOG2 marks shadows as 127)
        _, foreground = cv2.threshold(foreground, 130, 255, cv2.THRESH_BINARY)

        # 3. Median blur to remove salt-and-pepper noise
        foreground = cv2.medianBlur(foreground, self.median_filter_size)

        # 4. Dilate to fill holes
        for _ in range(4):
            foreground = cv2.dilate(foreground, None)

        # 5. Find contours
        contours, _ = cv2.findContours(
            foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = list(contours)

        # 6. Filter small contours
        contours = self._filter_bad_contours(contours)

        # 7. Get mass centers and bounding boxes
        mass_centers, bounding_boxes = self._get_centers_and_boxes(contours)

        # 8. Suppress mass centers
        contours, mass_centers, bounding_boxes = self._suppress_mass_centers(
            contours, mass_centers, bounding_boxes
        )

        # 9. Merge nearby contours
        contours = self._merge_contours(contours, mass_centers, bounding_boxes, diagonal)

        # 10. Recompute
        mass_centers, bounding_boxes = self._get_centers_and_boxes(contours)

        return contours, mass_centers, bounding_boxes, foreground

    def _filter_bad_contours(self, contours):
        if not contours:
            return contours
        areas = [cv2.contourArea(c) for c in contours]
        max_area = max(areas) if areas else 0
        if max_area == 0:
            return []
        threshold = self.contour_size_threshold * max_area
        return [c for c, a in zip(contours, areas) if a > threshold]

    def _get_centers_and_boxes(self, contours):
        mass_centers = []
        bounding_boxes = []
        for contour in contours:
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                cx, cy = 0.0, 0.0
            mass_centers.append((cx, cy))
            approx = cv2.approxPolyDP(contour, 3, True)
            bounding_boxes.append(cv2.boundingRect(approx))
        return mass_centers, bounding_boxes

    def _suppress_mass_centers(self, contours, mass_centers, bounding_boxes):
        if not self.suppress_rectangles:
            return contours, mass_centers, bounding_boxes
        keep = []
        for i, mc in enumerate(mass_centers):
            suppressed = False
            for rect in self.suppress_rectangles:
                rx, ry, rw, rh = rect
                if rx <= mc[0] <= rx + rw and ry <= mc[1] <= ry + rh:
                    suppressed = True
                    break
            if not suppressed:
                keep.append(i)
        return (
            [contours[i] for i in keep],
            [mass_centers[i] for i in keep],
            [bounding_boxes[i] for i in keep],
        )

    def _merge_contours(self, contours, mass_centers, bounding_boxes, diagonal):
        n = len(contours)
        if n <= 1:
            return contours
        sets = DisjointSets(n)
        for i in range(n):
            for j in range(i + 1, n):
                dist = _distance_between_rects(bounding_boxes[i], bounding_boxes[j])
                if dist < self.contour_merge_threshold * diagonal:
                    sets.union(i, j)
        groups = defaultdict(list)
        for i in range(n):
            groups[sets.find(i)].append(i)
        merged = []
        for indices in groups.values():
            if len(indices) == 1:
                merged.append(contours[indices[0]])
            else:
                combined = np.concatenate([contours[idx] for idx in indices], axis=0)
                merged.append(combined)
        return merged

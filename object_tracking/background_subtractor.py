"""
Step 1 — Background Subtraction using MOG2
===========================================

Theory
------
Each pixel is modeled as a Mixture of K Gaussians (typically K=5):

    p(x) = Σ_{k=1}^{K}  w_k · N(x | μ_k, σ_k²)

Where:
  - x   = pixel intensity (or RGB vector)
  - w_k = weight of the k-th Gaussian (how often this mode appears)
  - μ_k = mean intensity of the k-th Gaussian
  - σ_k = standard deviation of the k-th Gaussian

For each new frame, every pixel is tested against the existing Gaussians:
  1. If it matches a Gaussian (within 2.5σ), that Gaussian's parameters
     are updated via an exponential moving average.
  2. If no match, the least-probable Gaussian is replaced with a new one
     centered on the current pixel value.

The top B Gaussians (ranked by w/σ) that exceed a cumulative weight
threshold T are classified as "background". Everything else is "foreground".

OpenCV's MOG2 (Zivkovic 2004) additionally:
  - Adaptively selects K per-pixel (not fixed)
  - Optionally detects shadows (gray pixels vs white foreground)

Parameters
----------
  history   : number of recent frames used to build the model
  varThreshold : Mahalanobis distance threshold for match (higher = less sensitive)
  detectShadows : mark shadows as gray (127) instead of white (255)
"""

import cv2


class BackgroundSubtractor:
    """Wraps cv2.createBackgroundSubtractorMOG2 with sensible defaults."""

    def __init__(self, history: int = 500, var_threshold: float = 50,
                 detect_shadows: bool = True):
        """
        Parameters
        ----------
        history : int
            Number of frames the model remembers. Longer = more stable
            background, but slower adaptation to lighting changes.
        var_threshold : float
            Squared Mahalanobis distance threshold. Controls sensitivity:
            lower → more pixels flagged as foreground (noisier).
            higher → only very different pixels flagged (might miss objects).
        detect_shadows : bool
            If True, shadow pixels are marked 127 in the mask instead of 255.
            We then threshold them out so only true foreground remains.
        """
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

    def apply(self, frame):
        """
        Apply background subtraction to a single frame.

        Parameters
        ----------
        frame : np.ndarray (H, W, 3) BGR image

        Returns
        -------
        fg_mask : np.ndarray (H, W) binary mask  {0, 255}
        """
        # MOG2 returns: 0=background, 127=shadow, 255=foreground
        raw_mask = self.mog2.apply(frame)

        # Threshold out shadows → pure binary
        _, fg_mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

        return fg_mask

# stereo_vo/feature_tracker.py
import cv2
import numpy as np


# Lucas-Kanade optical flow settings
LK_PARAMS = dict(
    winSize  = (21, 21),   # patch size to search around each point
                            # larger = handles bigger motions but slower
    maxLevel = 3,           # pyramid levels: 0=full res only, 3=4 scales
                            # more levels = handles faster motion
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        30,      # max iterations
        0.01     # stop if movement < 0.01 px
    )
)


def detect_features(gray_img, max_features=500):
    """
    Detect good feature points to track using ORB.

    gray_img     : grayscale image
    max_features : how many points to detect (more = more robust but slower)

    Returns: (N, 1, 2) float32 array of pixel coordinates
             — this shape is what cv2.calcOpticalFlowPyrLK expects
    """
    # ORB detector
    orb = cv2.ORB_create(
        nfeatures   = max_features,
        scaleFactor = 1.2,    # pyramid scale between levels
        nlevels     = 8,      # number of pyramid levels
        edgeThreshold = 15,   # don't detect near image edges
        patchSize   = 31      # patch size for descriptor
    )

    keypoints = orb.detect(gray_img, None)

    if len(keypoints) == 0:
        return np.zeros((0, 1, 2), dtype=np.float32)

    # Convert keypoints to the (N,1,2) format LK flow expects
    points = np.array([[kp.pt] for kp in keypoints], dtype=np.float32)
    return points


def track_features(prev_gray, curr_gray, prev_pts):
    """
    Track feature points from prev frame to curr frame using
    Lucas-Kanade optical flow with a forward-backward consistency check.

    The forward-backward check works like this:
      1. Track points forward:  prev → curr  (get curr_pts)
      2. Track points backward: curr → prev  (get prev_pts_back)
      3. If prev_pts_back is far from prev_pts → bad track → discard

    This removes wrongly tracked points before they corrupt our pose estimate.

    Returns:
      curr_pts_good  : (M, 1, 2) tracked positions in curr frame
      prev_pts_good  : (M, 1, 2) corresponding positions in prev frame
      n_removed      : how many points were rejected
    """
    if len(prev_pts) == 0:
        return None, None, 0

    # ── Forward tracking: prev → curr ────────────────────────────────────
    curr_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **LK_PARAMS
    )

    # ── Backward tracking: curr → prev ───────────────────────────────────
    prev_pts_back, status_bk, _ = cv2.calcOpticalFlowPyrLK(
        curr_gray, prev_gray, curr_pts, None, **LK_PARAMS
    )

    # ── Consistency check ─────────────────────────────────────────────────
    # Compute how far each point drifted when tracked back
    # shape: (N, 2)
    fb_error = np.abs(prev_pts - prev_pts_back).reshape(-1, 2).max(axis=1)

    # A point is good if:
    #   - tracked successfully in both directions (status == 1)
    #   - round-trip error < 1.0 pixel
    good = (
        (status_fwd.ravel()  == 1) &
        (status_bk.ravel()   == 1) &
        (fb_error < 1.0)
    )

    n_total   = len(prev_pts)
    n_good    = good.sum()
    n_removed = n_total - n_good

    if n_good == 0:
        return None, None, n_removed

    return curr_pts[good], prev_pts[good], n_removed


def draw_tracks(img, prev_pts, curr_pts, color=(0, 255, 100)):
    """
    Draw feature tracks as arrows on an image.
    Each arrow shows where a point WAS (tail) and where it IS NOW (head).

    img      : BGR image to draw on (will be copied)
    prev_pts : (N, 2) or (N, 1, 2) previous positions
    curr_pts : (N, 2) or (N, 1, 2) current positions
    """
    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    prev = prev_pts.reshape(-1, 2).astype(int)
    curr = curr_pts.reshape(-1, 2).astype(int)

    for p, c in zip(prev, curr):
        # Draw the track line
        cv2.line(vis, tuple(p), tuple(c), color, 1, cv2.LINE_AA)
        # Draw current position as a dot
        cv2.circle(vis, tuple(c), 3, (0, 200, 255), -1)
        # Draw previous position as a smaller dot
        cv2.circle(vis, tuple(p), 2, (255, 100, 0), -1)

    # Show count
    cv2.putText(vis, f"Tracking {len(curr)} points",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    return vis
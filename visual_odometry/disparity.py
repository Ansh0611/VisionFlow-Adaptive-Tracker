# stereo_vo/disparity.py
import cv2
import numpy as np


def compute_disparity(left_img, right_img):
    """
    Compute a disparity map from a rectified stereo pair using SGBM.

    left_img, right_img : rectified BGR or grayscale images (same size)
    Returns             : float32 disparity map in pixels
                          0 = invalid / no match found
    """
    # SGBM works on grayscale
    if len(left_img.shape) == 3:
        left_gray  = cv2.cvtColor(left_img,  cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray  = left_img
        right_gray = right_img

    # ── SGBM parameters ──────────────────────────────────────────────────
    # These are the knobs you can tune. Explanation of each below.

    block_size     = 9     # size of matching patch (must be odd, 5–11 is typical)
                           # larger = smoother but loses fine detail
                           # smaller = more detail but noisier

    num_disparities = 96   # how many pixels to search left in the right image
                           # must be divisible by 16
                           # 96 means we search 96px to the left
                           # with our baseline (53cm) this gives min depth ~4m

    # P1, P2 control smoothness — think of them as "how much do we penalise
    # a disparity jump between neighbouring pixels"
    # P2 > P1 always. Rule of thumb: P1 = 8*ch*bs², P2 = 32*ch*bs²
    channels = 1
    P1 = 8  * channels * block_size ** 2    # penalty for ±1 disparity change
    P2 = 32 * channels * block_size ** 2    # penalty for larger disparity jump

    # Create the SGBM matcher
    matcher_left = cv2.StereoSGBM_create(
        minDisparity      = 0,              # start searching from 0 px shift
        numDisparities    = num_disparities,
        blockSize         = block_size,
        P1                = P1,
        P2                = P2,
        disp12MaxDiff     = 1,   # max diff between left→right and right→left match
        uniquenessRatio   = 10,  # reject match if 2nd best is within 10% of best
        speckleWindowSize = 100, # remove blobs smaller than 100 px
        speckleRange      = 32,  # max disparity variation within a speckle blob
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # ── Compute disparity ─────────────────────────────────────────────────
    # SGBM returns values multiplied by 16 (fixed-point)
    # so we divide by 16 to get real pixel values
    disp_raw = matcher_left.compute(left_gray, right_gray)
    disparity = disp_raw.astype(np.float32) / 16.0

    # Mark invalid pixels as 0
    disparity[disparity < 0] = 0

    return disparity


def disparity_to_depth(disparity, calib):
    """
    Convert disparity map → metric depth map using Z = f*B/d.

    disparity : float32 array (pixels), 0 = invalid
    calib     : dict from load_kitti_calib()
    Returns   : float32 depth map in metres, 0 = invalid
    """
    fx       = calib['fx']
    baseline = calib['baseline']

    # Avoid division by zero — set invalid pixels to depth 0
    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (fx * baseline) / disparity[valid]

    # Clip unreliable very-far depths (>80m gets noisy)
    depth[depth > 80.0] = 0

    return depth


def colorize_disparity(disparity):
    """
    Convert raw disparity values to a colour image for display.
    Uses INFERNO colormap: black=far/invalid, yellow=close.
    """
    # Normalise to 0-255
    disp_vis = disparity.copy()
    disp_vis = np.clip(disp_vis, 0, None)
    disp_norm = cv2.normalize(disp_vis, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(disp_norm, cv2.COLORMAP_INFERNO)


def colorize_depth(depth):
    """
    Convert metric depth map to a colour image.
    Uses JET colormap: blue=far, red=close (intuitive).
    Clips to 0-50m range for good contrast.
    """
    depth_vis = np.clip(depth, 0, 50)
    # Invert so close = warm colour
    depth_inv = 50 - depth_vis
    depth_norm = cv2.normalize(depth_inv, None, 0, 255,
                                cv2.NORM_MINMAX).astype(np.uint8)
    # Make invalid pixels black
    depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
    depth_colour[depth == 0] = [0, 0, 0]
    return depth_colour
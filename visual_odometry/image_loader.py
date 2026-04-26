# stereo_vo/image_loader.py
import cv2
import numpy as np
import os
import glob

def load_image_pair(sequence_dir, frame_index, grayscale=False):
    """
    Load one left+right image pair from a KITTI colour sequence.
    
    sequence_dir : path to sequences/00/
    frame_index  : which frame to load (0, 1, 2, ...)
    grayscale    : if True, convert to grayscale (needed later for matching)
    
    Returns: (left_img, right_img)  as numpy arrays
    """
    # KITTI colour: image_2 = left,  image_3 = right
    left_dir  = os.path.join(sequence_dir, 'image_2')
    right_dir = os.path.join(sequence_dir, 'image_3')

    # Get sorted list of all image files
    left_files  = sorted(glob.glob(os.path.join(left_dir,  '*.png')))
    right_files = sorted(glob.glob(os.path.join(right_dir, '*.png')))

    if len(left_files) == 0:
        raise FileNotFoundError(f"No images found in {left_dir}")

    if frame_index >= len(left_files):
        raise IndexError(f"Frame {frame_index} doesn't exist. "
                         f"Sequence has {len(left_files)} frames.")

    # Load the images
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    left  = cv2.imread(left_files[frame_index],  flag)
    right = cv2.imread(right_files[frame_index], flag)

    if left is None or right is None:
        raise IOError(f"Could not read frame {frame_index}")

    return left, right, len(left_files)


def compute_rectification_maps(calib, image_size):
    """
    Pre-compute the pixel remapping tables for rectification.
    
    For KITTI these maps are nearly identity (images already rectified),
    but computing them properly makes our pipeline work for any camera.
    
    image_size : (width, height)
    Returns    : (map1_left, map2_left, map1_right, map2_right)
                 — pass these to cv2.remap() to rectify any image pair
    """
    w, h = image_size

    # For KITTI: K_left=K_right, D=zeros, R=identity, so maps ≈ identity
    # For a real uncalibrated camera these would do real warping
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        calib['K_left'],           # camera matrix
        calib['D_left'],           # distortion (zeros for KITTI)
        np.eye(3),                 # rectification rotation (identity for KITTI)
        calib['P0'],               # new projection matrix
        (w, h),
        cv2.CV_32FC1               # output map type
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        calib['K_right'],
        calib['D_right'],
        np.eye(3),
        calib['P1'],
        (w, h),
        cv2.CV_32FC1
    )
    return map1_left, map2_left, map1_right, map2_right


def rectify_pair(left, right, maps):
    """
    Apply rectification maps to an image pair.
    
    maps   : the 4 maps from compute_rectification_maps()
    Returns: (left_rectified, right_rectified)
    """
    map1_left, map2_left, map1_right, map2_right = maps

    left_rect  = cv2.remap(left,  map1_left,  map2_left,  cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map1_right, map2_right, cv2.INTER_LINEAR)

    return left_rect, right_rect


def draw_epipolar_lines(left, right, num_lines=10):
    """
    Draw horizontal epipolar lines on a side-by-side image pair.
    
    If rectification is correct, every line touches the same feature
    in both images. Lines being horizontal confirms rectification worked.
    """
    h, w = left.shape[:2]

    # Stack images side by side
    if len(left.shape) == 2:                     # grayscale → colour for drawing
        left  = cv2.cvtColor(left,  cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([left, right])

    # Draw evenly spaced horizontal lines across both images
    colours = [
        (0,255,128), (0,200,255), (255,100,0),
        (200,0,255), (255,220,0), (0,128,255),
        (255,0,128), (128,255,0), (0,255,200), (255,128,128)
    ]
    step = h // (num_lines + 1)
    for i in range(num_lines):
        y = step * (i + 1)
        colour = colours[i % len(colours)]
        # Draw line across the full combined width
        cv2.line(combined, (0, y), (w * 2, y), colour, 1)

    return combined
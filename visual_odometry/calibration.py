# stereo_vo/calibration.py
import numpy as np
import cv2

def load_kitti_calib(calib_path):
    """
    Parse KITTI calib.txt and return all stereo parameters.
    Uses P2 (left colour) and P3 (right colour) for colour datasets.
    """
    data = {}
    with open(calib_path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            data[key.strip()] = np.array([float(x) for x in val.split()])

    # ── Use P2/P3 for colour cameras (image_2 / image_3) ──────────────
    P2 = data['P2'].reshape(3, 4)   # left colour camera
    P3 = data['P3'].reshape(3, 4)   # right colour camera

    # Intrinsics: upper-left 3×3 of each P matrix
    K_left  = P2[:3, :3]
    K_right = P3[:3, :3]

    fx = float(K_left[0, 0])
    fy = float(K_left[1, 1])
    cx = float(K_left[0, 2])
    cy = float(K_left[1, 2])

    # Baseline from P3[0,3] = -fx * baseline
    baseline = float(-P3[0, 3] / fx)

    # KITTI colour images are pre-rectified → distortion = zero
    D_left  = np.zeros(5)
    D_right = np.zeros(5)

    # Disparity-to-depth Q matrix
    cx2 = float(K_right[0, 2])
    Q = np.array([
        [1,  0,  0,         -cx        ],
        [0,  1,  0,         -cy        ],
        [0,  0,  0,          fx        ],
        [0,  0, -1/baseline, (cx-cx2)/baseline]
    ], dtype=np.float64)

    calib = {
        'P0': P2,           # keeping key as P0 so rest of code works
        'P1': P3,           # keeping key as P1 so rest of code works
        'K_left':  K_left,
        'K_right': K_right,
        'D_left':  D_left,
        'D_right': D_right,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'baseline': baseline,
        'Q': Q,
    }
    return calib


def print_calib_summary(calib):
    print("=" * 45)
    print("  Stereo Calibration Summary")
    print("=" * 45)
    print(f"  Focal length  fx = {calib['fx']:.2f} px")
    print(f"  Focal length  fy = {calib['fy']:.2f} px")
    print(f"  Principal pt  cx = {calib['cx']:.2f} px")
    print(f"  Principal pt  cy = {calib['cy']:.2f} px")
    print(f"  Baseline       B = {calib['baseline']*100:.2f} cm")
    print(f"  Min depth (d=96) = {calib['fx']*calib['baseline']/96:.2f} m")
    print(f"  Max depth (d=1)  = {calib['fx']*calib['baseline']/1:.1f} m")
    print("=" * 45)
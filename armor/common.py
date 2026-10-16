"""Shared constants and geometry / heatmap helpers (numpy + OpenCV, no torch)."""
import math
import numpy as np
import cv2

# ----------------------------------------------------------------------------------------------
# WFLW 98-point conventions
# ----------------------------------------------------------------------------------------------
LEFT_EYE_OUTER, RIGHT_EYE_OUTER = 60, 72      # inter-ocular normalisation (WFLW protocol)

# horizontal-flip permutation (index i -> its mirrored landmark)
WFLW_FLIP = (
    list(range(32, -1, -1))                                   # 0-32  face contour
    + [46, 45, 44, 43, 42, 50, 49, 48, 47]                    # 33-41 left brow  -> right brow
    + [37, 36, 35, 34, 33, 41, 40, 39, 38]                    # 42-50 right brow -> left brow
    + [51, 52, 53, 54]                                        # 51-54 nose bridge
    + [59, 58, 57, 56, 55]                                    # 55-59 nose bottom
    + [72, 71, 70, 69, 68, 75, 74, 73]                        # 60-67 left eye -> right eye
    + [64, 63, 62, 61, 60, 67, 66, 65]                        # 68-75 right eye -> left eye
    + [82, 81, 80, 79, 78, 77, 76]                            # 76-82 outer upper lip
    + [87, 86, 85, 84, 83]                                    # 83-87 outer lower lip
    + [92, 91, 90, 89, 88]                                    # 88-92 inner upper lip
    + [95, 94, 93]                                            # 93-95 inner lower lip
    + [97, 96]                                                # 96-97 pupils
)
assert len(WFLW_FLIP) == 98 and sorted(WFLW_FLIP) == list(range(98))

# edge (boundary) groups rendered as poly-lines; (indices, closed?)
EDGE_GROUPS = [
    (list(range(0, 33)), False),            # 0  face contour
    (list(range(33, 42)), True),            # 1  left eyebrow (loop)
    (list(range(42, 51)), True),            # 2  right eyebrow (loop)
    ([51, 52, 53, 54], False),              # 3  nose bridge
    ([55, 56, 57, 58, 59], False),          # 4  nose bottom
    (list(range(60, 68)), True),            # 5  left eye
    (list(range(68, 76)), True),            # 6  right eye
    ([76, 77, 78, 79, 80, 81, 82], False),  # 7  outer upper lip
    ([82, 83, 84, 85, 86, 87, 76], False),  # 8  outer lower lip
    ([88, 89, 90, 91, 92], False),          # 9  inner upper lip
    ([92, 93, 94, 95, 88], False),          # 10 inner lower lip
    ([96], False),                          # 11 left pupil
    ([97], False),                          # 12 right pupil
]
N_EDGES = len(EDGE_GROUPS)

ATTR_NAMES = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------------------------
# Geometry: one float32 affine with an exact inverse for every crop / back-projection
# ----------------------------------------------------------------------------------------------
def crop_affine(center, side, out_size, rot_deg=0.0, flip=False):
    """2x3 matrix mapping *source image* coords -> *output crop* coords.

    The crop is the square of edge `side` (source pixels) centred on `center`, rotated by
    `rot_deg` (degrees, counter-clockwise in image coordinates) and resized to `out_size`.
    """
    cx, cy = float(center[0]), float(center[1])
    s = out_size / float(side)
    a = math.radians(rot_deg)
    cos, sin = math.cos(a) * s, math.sin(a) * s
    # rotate+scale about the centre, then translate the centre to the crop centre
    M = np.array([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=np.float64)
    t = np.array([cx, cy])
    M[:, 2] = (out_size - 1) / 2.0 - M[:, :2] @ t
    if flip:  # x' = (out_size-1) - x
        F = np.array([[-1.0, 0.0, out_size - 1.0], [0.0, 1.0, 0.0]])
        M = np.vstack([F @ np.vstack([M, [0, 0, 1]]), [0, 0, 1]])[:2]
    return M.astype(np.float32)


def invert_affine(M):
    return cv2.invertAffineTransform(M.astype(np.float32)).astype(np.float32)


def transform_points(pts, M):
    """Apply a 2x3 affine to an (N,2) array of points."""
    pts = np.asarray(pts, dtype=np.float32)
    return pts @ M[:, :2].T + M[:, 2]


def warp(img, M, out_size, border=cv2.BORDER_CONSTANT, value=(0, 0, 0)):
    return cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=border, borderValue=value)


def square_from_box(box, scale):
    """(x1,y1,x2,y2) -> (center, side) of the square crop used by the pipeline."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    side = max(x2 - x1, y2 - y1) * scale
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), side


# ----------------------------------------------------------------------------------------------
# Heatmap rendering
# ----------------------------------------------------------------------------------------------
def render_landmark_heatmaps(pts, size, sigma):
    """Gaussian heatmaps (N,size,size) float32 for points given in heatmap-pixel coordinates."""
    pts = np.asarray(pts, dtype=np.float32)
    xs = np.arange(size, dtype=np.float32)
    gx = np.exp(-((xs[None, :] - pts[:, 0:1]) ** 2) / (2 * sigma ** 2))  # (N,size)
    gy = np.exp(-((xs[None, :] - pts[:, 1:2]) ** 2) / (2 * sigma ** 2))
    hm = gy[:, :, None] * gx[:, None, :]
    return hm.astype(np.float32)


def render_edge_heatmaps(pts, size, sigma, groups=EDGE_GROUPS):
    """Boundary heatmaps (N_EDGES,size,size): exp(-d^2/2s^2) of the distance to the poly-line."""
    pts = np.asarray(pts, dtype=np.float32)
    shift = 4
    out = np.zeros((len(groups), size, size), dtype=np.float32)
    for g, (idx, closed) in enumerate(groups):
        canvas = np.zeros((size, size), dtype=np.uint8)
        p = pts[idx]
        if not np.all(np.isfinite(p)):
            continue
        q = np.round(p * (1 << shift)).astype(np.int32)
        if len(idx) == 1:
            cv2.circle(canvas, (int(q[0, 0]), int(q[0, 1])), 0, 255, -1, lineType=cv2.LINE_8, shift=shift)
        else:
            cv2.polylines(canvas, [q.reshape(-1, 1, 2)], closed, 255, 1, lineType=cv2.LINE_8, shift=shift)
        if canvas.max() == 0:
            continue
        dist = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 3)
        out[g] = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    return out


def normalize_image(img_rgb_uint8):
    """HWC uint8 RGB -> CHW float32 normalised."""
    x = img_rgb_uint8.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def denormalize_image(chw):
    x = np.asarray(chw).transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)

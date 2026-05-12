import math

import cv2
import numpy as np


def make_tracker():
    for module in (getattr(cv2, "legacy", None), cv2):
        if module is None:
            continue
        for attr in ("TrackerMOSSE_create", "TrackerKCF_create", "TrackerCSRT_create"):
            if hasattr(module, attr):
                return getattr(module, attr)()
    return None


def clamp_bbox(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def order_quad(pts):
    pts  = np.array(pts, dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


def right_angle_score(quad):
    q, score = quad.astype(np.float32), 0.0
    for i in range(4):
        p0, p1, p2 = q[i], q[(i + 1) % 4], q[(i - 1) % 4]
        v1, v2 = p1 - p0, p2 - p0
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cosang = abs(float(np.dot(v1, v2) / (n1 * n2)))
        score += max(0.0, 1.0 - min(1.0, cosang * 5.0))
    return score / 4.0


def classify_shape(approx, area, min_circ):
    v           = len(approx)
    _, _, w, h  = cv2.boundingRect(approx)
    peri        = cv2.arcLength(approx, True)
    circularity = (4.0 * math.pi * area) / (peri * peri + 1e-6)
    angle_deg   = float(cv2.minAreaRect(approx)[2])
    quad_pts    = None

    if circularity >= min_circ:
        try:
            (_, _), enc_radius = cv2.minEnclosingCircle(approx)
            enc_area = math.pi * enc_radius * enc_radius
            if enc_area > 1e-6 and area / enc_area >= 0.80:
                return "circle", None, angle_deg
        except cv2.error:
            pass
    if v == 3:
        class_name = "triangle"
    elif v == 4 and cv2.isContourConvex(approx):
        aspect     = w / float(h) if h > 0 else 1.0
        class_name = "square" if 0.85 <= aspect <= 1.15 else "rect"
        quad_pts   = order_quad(approx.reshape(4, 2))
    elif v == 5:
        class_name = "pentagon"
    elif v == 6:
        class_name = "hexagon"
    elif circularity >= min_circ and v >= 7:
        class_name = "circle"
    else:
        class_name = "poly"

    return class_name, quad_pts, angle_deg

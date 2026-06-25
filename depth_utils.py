import cv2
import numpy as np

from config import (
    MIN_DEPTH_M, MAX_DEPTH_M, DANGER_DISTANCE_M, WARNING_DISTANCE_M,
    MIN_DEPTH_VALID_RATIO,
)


def valid_depth_values(depth_array):
    return depth_array[
        np.isfinite(depth_array) &
        (depth_array >= MIN_DEPTH_M) &
        (depth_array <= MAX_DEPTH_M)
    ]


def resize_depth_to_frame(depth_frame, frame_w, frame_h):
    return cv2.resize(depth_frame, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)


def resize_frame_and_depth(frame_bgr, depth_frame, max_width):
    """Downscale RGB + depth together to reduce CPU load and UI delay."""
    if frame_bgr is None or depth_frame is None:
        return frame_bgr, depth_frame
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr, resize_depth_to_frame(depth_frame, w, h)
    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    frame_small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    depth_small = cv2.resize(depth_frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return frame_small, depth_small


def get_center_lidar_depth(depth_frame):
    if depth_frame is None:
        return None
    h, w = depth_frame.shape[:2]
    cx = w // 2
    cy = h // 2
    area = depth_frame[
        max(0, cy - 10):min(h, cy + 10),
        max(0, cx - 10):min(w, cx + 10)
    ]
    valid = valid_depth_values(area)
    if valid.size == 0:
        return None
    return float(np.median(valid))


def get_lidar_distance_in_bbox(depth_resized, x1, y1, x2, y2):
    if depth_resized is None:
        return None

    h, w = depth_resized.shape[:2]
    x1 = max(0, min(w - 1, int(x1)))
    x2 = max(0, min(w - 1, int(x2)))
    y1 = max(0, min(h - 1, int(y1)))
    y2 = max(0, min(h - 1, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1
    cx1 = x1 + int(box_w * 0.28)
    cx2 = x2 - int(box_w * 0.28)
    cy1 = y1 + int(box_h * 0.25)
    cy2 = y2 - int(box_h * 0.20)
    if cx2 <= cx1 or cy2 <= cy1:
        return None

    roi = depth_resized[cy1:cy2, cx1:cx2]
    valid = valid_depth_values(roi)
    if valid.size < 20:
        return None
    return float(np.median(valid))


def get_distance_status(distance_m):
    if distance_m is None:
        return "UNKNOWN", (160, 160, 160)
    if distance_m < DANGER_DISTANCE_M:
        return "CRITICAL", (45, 45, 230)
    if distance_m < WARNING_DISTANCE_M:
        return "WARNING", (0, 210, 255)
    return "CLEAR", (60, 220, 90)


def draw_depth_view(depth_frame):
    if depth_frame is None:
        return None
    small = cv2.resize(depth_frame, (320, 240), interpolation=cv2.INTER_NEAREST)
    small = np.nan_to_num(small, nan=0.0, posinf=0.0, neginf=0.0)
    small = np.clip(small, 0.0, MAX_DEPTH_M)
    normalized = (small / MAX_DEPTH_M * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def get_depth_health(depth_frame):
    """Return whether depth is valid enough for mapping."""
    if depth_frame is None:
        return False, 0.0, None
    arr = np.array(depth_frame)
    if arr.size == 0:
        return False, 0.0, None
    valid = valid_depth_values(arr)
    valid_ratio = float(valid.size / arr.size)
    median_depth = None if valid.size == 0 else float(np.median(valid))
    depth_ok = valid_ratio >= MIN_DEPTH_VALID_RATIO
    return depth_ok, valid_ratio, median_depth

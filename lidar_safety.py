import time
import cv2
import numpy as np

import app_state as st
from config import (
    DANGER_DISTANCE_M, WARNING_DISTANCE_M, FRONT_PERCENTILE,
    DANGER_RATIO_THRESHOLD, WARNING_RATIO_THRESHOLD, SAFETY_HOLD_SEC,
    DRAW_LOW_LIDAR_ZONE,
)
from depth_utils import valid_depth_values

# Smaller and more even debug zones. They are only a visual guide; safety still uses depth values.
SAFETY_ZONES = {
    # LEFT / CENTER / RIGHT are equal-size visual boxes.
    # LOW is kept for safety calculation but hidden by default to avoid a messy camera view.
    "LEFT":   (0.20, 0.38, 0.40, 0.72),
    "CENTER": (0.41, 0.59, 0.40, 0.72),
    "RIGHT":  (0.62, 0.80, 0.40, 0.72),
    "LOW":    (0.41, 0.59, 0.74, 0.92),
}


def toggle_lidar_zones():
    st.show_lidar_zones = not st.show_lidar_zones
    print(f"LiDAR zone display: {'ON' if st.show_lidar_zones else 'OFF'}")


def zone_distance(depth_frame, x1f, x2f, y1f, y2f):
    if depth_frame is None:
        return None, 0.0, 0.0

    h, w = depth_frame.shape[:2]
    x1 = max(0, min(w - 1, int(w * x1f)))
    x2 = max(0, min(w, int(w * x2f)))
    y1 = max(0, min(h - 1, int(h * y1f)))
    y2 = max(0, min(h, int(h * y2f)))
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, 0.0

    roi = depth_frame[y1:y2, x1:x2]
    valid = valid_depth_values(roi)
    if valid.size < 25:
        return None, 0.0, 0.0

    distance = float(np.percentile(valid, FRONT_PERCENTILE))
    danger_ratio = float(np.mean(valid < DANGER_DISTANCE_M))
    warning_ratio = float(np.mean(valid < WARNING_DISTANCE_M))
    return distance, danger_ratio, warning_ratio


def get_front_safety(depth_frame):
    """Multi-zone LiDAR safety."""
    nearest = None
    danger_votes = 0
    warning_votes = 0
    st.last_lidar_zone_debug = {}

    for name, z in SAFETY_ZONES.items():
        dist, danger_ratio, warning_ratio = zone_distance(depth_frame, *z)
        zone_state = "UNKNOWN"
        if dist is not None:
            if dist < DANGER_DISTANCE_M or danger_ratio >= DANGER_RATIO_THRESHOLD:
                zone_state = "CRITICAL"
                danger_votes += 1
            elif dist < WARNING_DISTANCE_M or warning_ratio >= WARNING_RATIO_THRESHOLD:
                zone_state = "WARNING"
                warning_votes += 1
            else:
                zone_state = "CLEAR"
            if nearest is None or dist < nearest:
                nearest = dist

        st.last_lidar_zone_debug[name] = {
            "rect": z,
            "distance": dist,
            "state": zone_state,
            "danger_ratio": danger_ratio,
            "warning_ratio": warning_ratio,
            "mapped_obstacle": False,
        }

    raw_state = "CLEAR"
    if danger_votes >= 1:
        raw_state = "CRITICAL"
    elif warning_votes >= 1:
        raw_state = "WARNING"

    st.safety_history.append(raw_state)
    if "CRITICAL" in st.safety_history:
        stable_state = "CRITICAL"
    elif list(st.safety_history).count("WARNING") >= 2:
        stable_state = "WARNING"
    else:
        stable_state = "CLEAR"
    return stable_state, nearest


def apply_safety_hold(state, distance):
    now = time.time()
    if state in ("CRITICAL", "WARNING"):
        st.last_alert_state = state
        st.last_alert_distance = distance
        st.last_alert_time = now
        return state, distance

    if now - st.last_alert_time < SAFETY_HOLD_SEC and st.last_alert_state in ("CRITICAL", "WARNING"):
        return st.last_alert_state, st.last_alert_distance

    st.last_alert_state = "CLEAR"
    st.last_alert_distance = None
    return "CLEAR", None


def get_lidar_zone_name_for_pixel(x, y, frame_w, frame_h):
    xf = x / max(1, frame_w)
    yf = y / max(1, frame_h)
    for name in ("CENTER", "LEFT", "RIGHT", "LOW"):
        x1, x2, y1, y2 = SAFETY_ZONES[name]
        if x1 <= xf <= x2 and y1 <= yf <= y2:
            return name
    return None


def mark_lidar_zone_obstacle(zone_name, distance=None):
    if not st.show_obstacle_zone_alert or zone_name is None:
        return
    if zone_name not in st.last_lidar_zone_debug:
        return

    info = st.last_lidar_zone_debug[zone_name]
    info["mapped_obstacle"] = True
    if distance is not None:
        old = info.get("distance", None)
        if old is None or distance < old:
            info["distance"] = distance
        if distance < DANGER_DISTANCE_M:
            info["state"] = "CRITICAL"
        elif distance < WARNING_DISTANCE_M and info.get("state") != "CRITICAL":
            info["state"] = "WARNING"


def _zone_color(state, mapped_obstacle=False):
    if state == "CRITICAL":
        return (0, 0, 255)
    if mapped_obstacle:
        return (255, 0, 255)
    if state == "WARNING":
        return (0, 210, 255)
    if state == "CLEAR":
        return (90, 170, 230)
    return (100, 100, 100)


def draw_front_zones(frame, compact=True, show_caption=False):
    if not st.show_lidar_zones:
        return

    h, w = frame.shape[:2]
    zones_to_draw = st.last_lidar_zone_debug if st.last_lidar_zone_debug else {
        name: {"rect": rect, "distance": None, "state": "UNKNOWN", "mapped_obstacle": False}
        for name, rect in SAFETY_ZONES.items()
    }

    font_scale = 0.24 if compact else 0.38
    thickness = 1

    for name, info in zones_to_draw.items():
        if name == "LOW" and not DRAW_LOW_LIDAR_ZONE:
            continue
        x1f, x2f, y1f, y2f = info["rect"]
        state = info.get("state", "UNKNOWN")
        dist = info.get("distance", None)
        mapped_obstacle = info.get("mapped_obstacle", False)

        x1 = int(w * x1f)
        x2 = int(w * x2f)
        y1 = int(h * y1f)
        y2 = int(h * y2f)

        color = _zone_color(state, mapped_obstacle)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{name[0]}" if dist is None else f"{name[0]} {dist:.2f}"
        label_y = max(12, y1 - 3)
        cv2.putText(frame, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    if show_caption:
        cv2.putText(frame, "LiDAR zones", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 210, 240), 1)

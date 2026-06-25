import json
import os
from datetime import datetime

import cv2
import numpy as np

import app_state as st
from config import SAVE_ROOT, LOCAL_GRID_RESOLUTION_M, GLOBAL_GRID_RESOLUTION_M, MAP_COMBINED_PANEL_W, MAP_COMBINED_PANEL_H


def ensure_save_root():
    os.makedirs(SAVE_ROOT, exist_ok=True)


def sanitize_name(name):
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return safe or "operator"


def build_map_preview(local_map, global_map):
    if global_map is not None:
        return global_map
    if local_map is not None:
        return local_map
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[:] = (25, 25, 25)
    cv2.putText(img, "No map preview yet", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
    return img


def build_camera_map_preview(frame_bgr, local_map, global_map):
    """Build a bounded side-by-side MAP preview.

    Important: this function intentionally creates a smaller image.
    A very large PhotoImage can force Tkinter widgets to request more space,
    which makes the camera preview grow into the right options panel/log area.
    """
    if frame_bgr is None:
        return build_map_preview(local_map, global_map)

    panel_h = MAP_COMBINED_PANEL_H
    panel_w = MAP_COMBINED_PANEL_W
    cam = cv2.resize(frame_bgr, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    map_img = build_map_preview(local_map, global_map)
    map_img = cv2.resize(map_img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    cv2.putText(map_img, "Mapping Preview", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2)
    return np.hstack([cam, map_img])


def save_map_bundle(info, last_local_map, last_global_map, last_frame, last_depth, last_intrinsics, pose_source, robot_pose):
    ensure_save_root()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{ts}_{sanitize_name(info.get('user_name', 'operator'))}"
    folder = os.path.join(SAVE_ROOT, folder_name)
    os.makedirs(folder, exist_ok=True)

    np.save(os.path.join(folder, "local_occ.npy"), st.local_occ)
    np.save(os.path.join(folder, "local_free.npy"), st.local_free)
    np.save(os.path.join(folder, "global_occ.npy"), st.global_occ)
    np.save(os.path.join(folder, "global_free.npy"), st.global_free)

    preview = build_map_preview(last_local_map, last_global_map)
    cv2.imwrite(os.path.join(folder, "preview.png"), preview)

    metadata = {
        "timestamp": ts,
        "user_name": info.get("user_name", ""),
        "project_note": info.get("project_note", ""),
        "autosave_minutes": info.get("autosave_minutes", 0),
        "mode": st.config.get("name", "UNKNOWN"),
        "pose_source": pose_source,
        "robot_pose_xytheta": None if robot_pose is None else [float(robot_pose[0]), float(robot_pose[1]), float(robot_pose[2])],
        "vo_pose_ok": bool(st.vo_pose_ok),
        "map_resolution": {
            "local_m_per_cell": LOCAL_GRID_RESOLUTION_M,
            "global_m_per_cell": GLOBAL_GRID_RESOLUTION_M,
        },
        "map_shape": {
            "local": list(st.local_occ.shape),
            "global": list(st.global_occ.shape),
        },
        "files": {
            "local_occ": "local_occ.npy",
            "local_free": "local_free.npy",
            "global_occ": "global_occ.npy",
            "global_free": "global_free.npy",
            "preview": "preview.png",
            "point_cloud": "pointcloud.ply",
        }
    }

    with open(os.path.join(folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return folder


def load_map_bundle(folder):
    paths = {
        "local_occ": os.path.join(folder, "local_occ.npy"),
        "local_free": os.path.join(folder, "local_free.npy"),
        "global_occ": os.path.join(folder, "global_occ.npy"),
        "global_free": os.path.join(folder, "global_free.npy"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("Missing map files: " + ", ".join(missing))

    lo = np.load(paths["local_occ"])
    lf = np.load(paths["local_free"])
    go = np.load(paths["global_occ"])
    gf = np.load(paths["global_free"])

    if lo.shape != st.local_occ.shape or lf.shape != st.local_free.shape:
        raise ValueError("Loaded local map shape does not match current config.")
    if go.shape != st.global_occ.shape or gf.shape != st.global_free.shape:
        raise ValueError("Loaded global map shape does not match current config.")

    st.local_occ[:] = lo
    st.local_free[:] = lf
    st.global_occ[:] = go
    st.global_free[:] = gf

    metadata = {}
    meta_path = os.path.join(folder, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return metadata

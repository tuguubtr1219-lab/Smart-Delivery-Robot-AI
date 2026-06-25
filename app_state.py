"""Shared mutable runtime state.

Modules import this file as `import app_state as st` so changes remain visible
across the whole app.
"""
from collections import deque
import numpy as np

from config import (
    DRIVE_CONFIG, LOCAL_GRID_H, LOCAL_GRID_W, GLOBAL_GRID_H, GLOBAL_GRID_W,
    SAFETY_HISTORY,
)

config = DRIVE_CONFIG.copy()

# Occupancy maps
local_occ = np.zeros((LOCAL_GRID_H, LOCAL_GRID_W), dtype=np.float32)
local_free = np.zeros((LOCAL_GRID_H, LOCAL_GRID_W), dtype=np.float32)
local_occ_hits = np.zeros((LOCAL_GRID_H, LOCAL_GRID_W), dtype=np.uint8)

global_occ = np.zeros((GLOBAL_GRID_H, GLOBAL_GRID_W), dtype=np.float32)
global_free = np.zeros((GLOBAL_GRID_H, GLOBAL_GRID_W), dtype=np.float32)
global_occ_hits = np.zeros((GLOBAL_GRID_H, GLOBAL_GRID_W), dtype=np.uint8)

global_pose_available = False
global_robot_pose = (0.0, 0.0, 0.0)

# Safety state
safety_history = deque(maxlen=SAFETY_HISTORY)
last_alert_state = "CLEAR"
last_alert_distance = None
last_alert_time = 0.0
last_lidar_zone_debug = {}

# Debug display toggles
show_lidar_zones = True
show_camera_side_panel = True
show_obstacle_zone_alert = True

# Visual odometry state
prev_rgbd_o3d = None
vo_global_pose = np.identity(4, dtype=np.float64)
vo_pose_ok = False
vo_last_success = False

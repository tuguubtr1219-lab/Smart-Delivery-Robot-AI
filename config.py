"""
Central configuration for the Smart Delivery Robot AI project.
Tune this file first when you want more speed, cleaner overlays, or different mapping behavior.
"""

DB_FILE = "face_database.pkl"
SIMILARITY_THRESHOLD = 0.45

YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.40
YOLO_DEVICE = 0
YOLO_IMG_SIZE_DRIVE = 256
YOLO_IMG_SIZE_MAP = 288

# GUI / performance controls
# The loop wakes up often, but duplicate camera frames are skipped in gui.py.
PROCESS_INTERVAL_MS = 10
RENDER_EVERY_N_FRAMES = 1
MAX_PROCESS_FPS = 20
DISPLAY_MAX_FPS = 20
SAFETY_EVERY_N_FRAMES = 2
QR_SCAN_EVERY_N_FRAMES = 5

# Record3D can provide large RGBD frames. Downscale as soon as the frame enters Python
# to reduce memory copy cost and UI delay.
CAMERA_STORE_MAX_W = 400
MAX_PROCESS_FRAME_W = 320

# Only run expensive YOLO in Drive mode if you explicitly enable it.
ENABLE_YOLO_IN_DRIVE = False

# Keep preview image inside the center area so it cannot push side panels/log.
GUI_PREVIEW_MAX_W = 980
GUI_PREVIEW_MAX_H = 520
GUI_PREVIEW_W = 960
GUI_PREVIEW_H = 620
MAP_COMBINED_PANEL_W = 430
MAP_COMBINED_PANEL_H = 330

# Overlay controls
HIDE_LIDAR_OVERLAY_WHILE_MAPPING = True
COMPACT_OVERLAY = True
DRAW_LOW_LIDAR_ZONE = False

# Small robot indoor thresholds
DANGER_DISTANCE_M = 0.15
WARNING_DISTANCE_M = 0.22

MIN_DEPTH_M = 0.04
MAX_DEPTH_M = 5.00

# Local map
LOCAL_GRID_W = 220
LOCAL_GRID_H = 220
LOCAL_GRID_RESOLUTION_M = 0.06
LOCAL_ROBOT_CELL = (LOCAL_GRID_W // 2, LOCAL_GRID_H - 25)
LOCAL_GOAL_CELL = (LOCAL_GRID_W // 2, 25)

# Global map
GLOBAL_GRID_W = 700
GLOBAL_GRID_H = 700
GLOBAL_GRID_RESOLUTION_M = 0.05
GLOBAL_ORIGIN_CELL = (GLOBAL_GRID_W // 2, GLOBAL_GRID_H // 2)

LATERAL_SCALE = 0.62

# Mapping filters
FLOOR_IGNORE_Y_RATIO = 0.5
OBSTACLE_MARK_Y_MIN = 0.20
OBSTACLE_MARK_Y_MAX = 0.5
NEAR_OBSTACLE_MARK_M = 0.4
OBSTACLE_CONFIRM_HITS = 3
FREE_CLEAR_STRONG_DEC = 28
ROBOT_CLEAR_RADIUS_CELLS = 7
FALSE_OBSTACLE_CLEAR_RADIUS = 2
MAP_CLOSE_FREE_RAY_RATIO = 0.92

# Confidence map
OCC_DECAY = 0.996
FREE_DECAY = 0.997
OCC_HIT_INC = 16
FREE_HIT_INC = 14
OCC_FREE_PENALTY = 14
FREE_OCC_PENALTY = 10
OCC_THRESHOLD = 65
FREE_THRESHOLD = 18
OCC_MAX = 255
FREE_MAX = 255

INFLATION_RADIUS_CELLS = 4

# Safety
FRONT_PERCENTILE = 3
DANGER_RATIO_THRESHOLD = 0.010
WARNING_RATIO_THRESHOLD = 0.035
SAFETY_HISTORY = 5
SAFETY_HOLD_SEC = 0.30
DEPTH_VIEW_EVERY_N_FRAMES = 4

# Record3D metadata read rate. Reading pose/intrinsics every frame can add delay.
INTRINSICS_READ_EVERY_N_FRAMES = 60
POSE_READ_EVERY_N_FRAMES = 5

# Visual Odometry settings
VO_EVERY_N_FRAMES_DRIVE = 999999
VO_EVERY_N_FRAMES_MAP = 24
VO_DEPTH_TRUNC_M = 5.0
VO_MIN_TRANSLATION_M = 0.001
VO_MAX_TRANSLATION_M = 0.25

# Performance settings for two modes
DRIVE_CONFIG = {
    "name": "DRIVE",
    "yolo_every": 15,  
    "face_every": 15,
    "local_map_every": 999999,
    "global_map_every": 999999,
    "draw_map_every": 999999,
    "depth_step": 64,
    "yolo_imgsz": YOLO_IMG_SIZE_DRIVE,
    "show_camera": True,
    "show_local_map": False,
    "show_global_map": False,
    "show_depth": False,
    "use_vo": False,
    "vo_every": VO_EVERY_N_FRAMES_DRIVE,
}

MAP_CONFIG = {
    "name": "MAP",
    # Mapping mode disables heavy camera AI and focuses on depth + optional VO + map.
    "yolo_every": 999999,
    "face_every": 999999,
    "local_map_every": 16,
    "global_map_every": 32,
    "draw_map_every": 32,
    "depth_step": 68,
    "yolo_imgsz": YOLO_IMG_SIZE_MAP,
    "show_camera": True,
    "show_local_map": True,
    "show_global_map": True,
    "show_depth": False,
    "use_vo": False,
    "vo_every": VO_EVERY_N_FRAMES_MAP,
}

WINDOW_NAME = "Smart Delivery Robot AI - Modular"
LOCAL_MAP_WINDOW = "Local LiDAR Map"
GLOBAL_MAP_WINDOW = "Global SLAM-like Map"
DEPTH_WINDOW = "LiDAR Depth View"

# Mapping safety controls
MIN_DEPTH_VALID_RATIO = 0.03
REQUIRE_POSE_FOR_GLOBAL_MAP = True

SAVE_ROOT = "saved_maps"

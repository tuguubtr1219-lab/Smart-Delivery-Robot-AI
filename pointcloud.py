import cv2
import numpy as np

from config import MIN_DEPTH_M, MAX_DEPTH_M
from depth_utils import resize_depth_to_frame


def save_point_cloud_ply(filename, frame_bgr, depth_frame, intrinsics=None, stride=5):
    if frame_bgr is None or depth_frame is None:
        print("No RGBD frame to save.")
        return

    frame_h, frame_w = frame_bgr.shape[:2]
    depth_resized = resize_depth_to_frame(depth_frame, frame_w, frame_h)

    fx = frame_w * 0.9
    fy = frame_w * 0.9
    cx = frame_w / 2.0
    cy = frame_h / 2.0

    if intrinsics is not None and np.array(intrinsics).shape == (3, 3):
        K = np.array(intrinsics, dtype=np.float32)
        if float(K[0, 0]) > 1 and float(K[1, 1]) > 1:
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])

    points = []
    for v in range(0, frame_h, stride):
        for u in range(0, frame_w, stride):
            z = float(depth_resized[v, u])
            if not np.isfinite(z) or z <= MIN_DEPTH_M or z > MAX_DEPTH_M:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            b, g, r = frame_bgr[v, u]
            points.append((x, -y, z, int(r), int(g), int(b)))

    with open(filename, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]}\n")

    print(f"Saved point cloud: {filename} ({len(points)} points)")

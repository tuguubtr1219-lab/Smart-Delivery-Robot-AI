import cv2
import numpy as np
import open3d as o3d

import app_state as st
from config import MIN_DEPTH_M, MAX_DEPTH_M, VO_DEPTH_TRUNC_M, VO_MAX_TRANSLATION_M


def make_rgbd_o3d(frame_bgr, depth_frame):
    if frame_bgr is None or depth_frame is None:
        return None

    # Downsample for speed. VO does not need full resolution.
    target_w = 320
    target_h = 240
    frame_bgr = cv2.resize(frame_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    depth_frame = cv2.resize(depth_frame, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    depth_clean = depth_frame.astype(np.float32)
    depth_clean = np.nan_to_num(depth_clean, nan=0.0, posinf=0.0, neginf=0.0)
    depth_clean[(depth_clean < MIN_DEPTH_M) | (depth_clean > MAX_DEPTH_M)] = 0.0

    color_o3d = o3d.geometry.Image(frame_rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(depth_clean.astype(np.float32))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,
        depth_trunc=VO_DEPTH_TRUNC_M,
        convert_rgb_to_intensity=False
    )


def make_intrinsic_o3d(width, height, intrinsics=None):
    if intrinsics is not None and np.array(intrinsics).shape == (3, 3):
        K = np.array(intrinsics, dtype=np.float64)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        if fx <= 1 or fy <= 1:
            fx = width * 0.9
            fy = width * 0.9
            cx = width / 2.0
            cy = height / 2.0
    else:
        fx = width * 0.9
        fy = width * 0.9
        cx = width / 2.0
        cy = height / 2.0
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def compute_rgbd_visual_odometry(prev_rgbd, curr_rgbd, intrinsic):
    if prev_rgbd is None or curr_rgbd is None or intrinsic is None:
        return False, np.identity(4)

    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4, dtype=np.float64)
    try:
        success, trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            prev_rgbd,
            curr_rgbd,
            intrinsic,
            odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            option
        )
    except Exception as e:
        print("VO error:", e)
        return False, np.identity(4)

    if not success:
        return False, np.identity(4)

    translation = np.linalg.norm(trans[:3, 3])
    if translation > VO_MAX_TRANSLATION_M:
        print(f"VO rejected jump: {translation:.3f}m")
        return False, np.identity(4)
    return True, trans


def update_visual_odometry(frame_bgr, depth_frame, intrinsics):
    curr_rgbd = make_rgbd_o3d(frame_bgr, depth_frame)
    intrinsic = make_intrinsic_o3d(320, 240, None)
    if curr_rgbd is None:
        st.vo_last_success = False
        return None

    if st.prev_rgbd_o3d is None:
        st.prev_rgbd_o3d = curr_rgbd
        st.vo_pose_ok = False
        st.vo_last_success = False
        return None

    success, trans = compute_rgbd_visual_odometry(st.prev_rgbd_o3d, curr_rgbd, intrinsic)
    if success:
        st.vo_global_pose = st.vo_global_pose @ trans
        st.vo_pose_ok = True
        st.vo_last_success = True
    else:
        st.vo_last_success = False

    st.prev_rgbd_o3d = curr_rgbd
    return st.vo_global_pose.copy() if st.vo_pose_ok else None


def reset_visual_odometry():
    st.prev_rgbd_o3d = None
    st.vo_global_pose = np.identity(4, dtype=np.float64)
    st.vo_pose_ok = False
    st.vo_last_success = False


def extract_pose_xytheta(pose):
    if pose is None:
        return None
    arr = np.array(pose)
    if arr.shape == (4, 4):
        x = float(arr[0, 3])
        z = float(arr[2, 3])
        theta = float(np.arctan2(arr[0, 2], arr[2, 2]))
        return x, z, theta
    if arr.size >= 16:
        mat = arr.reshape(4, 4)
        x = float(mat[0, 3])
        z = float(mat[2, 3])
        theta = float(np.arctan2(mat[0, 2], mat[2, 2]))
        return x, z, theta
    return None


def extract_pose_xytheta_from_matrix(matrix):
    if matrix is None:
        return None
    arr = np.array(matrix)
    if arr.shape != (4, 4):
        return None
    x = float(arr[0, 3])
    z = float(arr[2, 3])
    theta = float(np.arctan2(arr[0, 2], arr[2, 2]))
    return x, z, theta

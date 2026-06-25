import threading
import cv2
import numpy as np
from record3d import Record3DStream

from config import CAMERA_STORE_MAX_W, INTRINSICS_READ_EVERY_N_FRAMES, POSE_READ_EVERY_N_FRAMES


class Record3DCamera:
    """Small wrapper around Record3DStream.

    Version 3 stores a downscaled latest frame and a frame_id. The GUI skips duplicate
    frame_id values, which prevents processing the same RGBD frame many times and keeps
    the display much closer to real time.
    """

    def __init__(self):
        self.stream = Record3DStream()
        self.rgb_frame = None
        self.depth_frame = None
        self.intrinsics = None
        self.pose = None
        self.frame_id = 0
        self.callback_count = 0
        self.running = False
        self.lock = threading.Lock()

        self.stream.on_new_frame = self.on_new_frame
        self.stream.on_stream_stopped = self.on_stream_stopped

    def try_get_pose(self):
        """Record3D Python SDK versions differ, so try several pose method names."""
        candidates = [
            "get_camera_pose",
            "get_camera_pose_matrix",
            "get_pose",
            "get_camera_position",
        ]
        for name in candidates:
            if hasattr(self.stream, name):
                try:
                    value = getattr(self.stream, name)()
                    if value is not None:
                        return np.array(value)
                except Exception:
                    pass
        return None

    @staticmethod
    def _resize_rgbd(frame_bgr, depth_frame, max_width):
        if frame_bgr is None or depth_frame is None:
            return frame_bgr, depth_frame
        h, w = frame_bgr.shape[:2]
        if w <= max_width:
            # Depth might not match RGB resolution; align it once here.
            if depth_frame.shape[:2] != (h, w):
                depth_frame = cv2.resize(depth_frame, (w, h), interpolation=cv2.INTER_NEAREST)
            return frame_bgr, depth_frame

        scale = max_width / float(w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        frame_small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        depth_small = cv2.resize(depth_frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return frame_small, depth_small

    def on_new_frame(self):
        if not self.running:
            return
        rgb = self.stream.get_rgb_frame()
        depth = self.stream.get_depth_frame()
        if rgb is None or depth is None:
            return

        self.callback_count += 1
        frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame_bgr, depth = self._resize_rgbd(frame_bgr, depth, CAMERA_STORE_MAX_W)

        # Reading intrinsics/pose can be non-trivial. Reuse previous values between reads.
        intrinsics = self.intrinsics
        if self.callback_count == 1 or self.callback_count % INTRINSICS_READ_EVERY_N_FRAMES == 0:
            try:
                intrinsics = self.stream.get_intrinsic_mat()
            except Exception:
                intrinsics = self.intrinsics

        pose = self.pose
        if self.callback_count % POSE_READ_EVERY_N_FRAMES == 0:
            pose = self.try_get_pose()

        with self.lock:
            self.rgb_frame = frame_bgr
            self.depth_frame = depth
            self.intrinsics = intrinsics
            self.pose = pose
            self.frame_id += 1

    def on_stream_stopped(self):
        print("Record3D stream stopped.")

    def start(self):
        self.running = True
        print("Connecting to Record3D via USB...")
        devices = self.stream.get_connected_devices()

        if len(devices) == 0:
            raise RuntimeError(
                "No Record3D USB device found.\n\n"
                "Please connect your iPhone with a USB cable, open the Record3D app, "
                "and make sure USB streaming is enabled."
            )
        print("Found Record3D devices:")
        for i, device in enumerate(devices):
            print(f"{i}: {device}")

        ok = self.stream.connect(devices[0])
        if not ok:
            raise RuntimeError(
                "Failed to connect to Record3D device.\n\n"
                "Please check the USB cable, unlock your iPhone, and restart the Record3D app."
            )
        print("Connected to Record3D.")


    def stop(self):
        self.running = False

        try:
            self.stream.on_new_frame = None
            self.stream.on_stream_stopped = None
        except Exception:
            pass

        try:
            if hasattr(self.stream, "stop"):
                self.stream.stop()
            elif hasattr(self.stream, "disconnect"):
                self.stream.disconnect()
        except Exception as e:
            print("Record3D stop warning:", e)

        with self.lock:
            self.rgb_frame = None
            self.depth_frame = None
            self.intrinsics = None
            self.pose = None

    def read_latest(self):
        with self.lock:
            if self.rgb_frame is None or self.depth_frame is None:
                return False, None, None, None, None, self.frame_id
            frame = self.rgb_frame.copy()
            depth = self.depth_frame.copy()
            intrinsics = None if self.intrinsics is None else np.array(self.intrinsics).copy()
            pose = None if self.pose is None else np.array(self.pose).copy()
            frame_id = self.frame_id
        return True, frame, depth, intrinsics, pose, frame_id

    def read(self):
        """Backward-compatible 5-value read."""
        ret, frame, depth, intrinsics, pose, _ = self.read_latest()
        return ret, frame, depth, intrinsics, pose

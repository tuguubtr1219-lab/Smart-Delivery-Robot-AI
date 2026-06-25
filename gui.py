import os
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from models import get_yolo_model

import cv2
import numpy as np
from PIL import Image, ImageTk

import app_state as st
from camera_record3d import Record3DCamera
from config import *
from depth_utils import (
    resize_frame_and_depth, get_depth_health, get_center_lidar_depth,
    get_lidar_distance_in_bbox, get_distance_status,
)
from dialogs import SaveMapDialog, AddPersonDialog
from drawing import draw_dashboard, draw_simple_label, draw_safety_badge
from face_recognition_utils import add_new_person_with_name, recognize_face
from lidar_safety import (
    toggle_lidar_zones, get_front_safety, apply_safety_hold,
    get_lidar_zone_name_for_pixel, mark_lidar_zone_obstacle, draw_front_zones,
)
from mapping import (
    reset_local_map, reset_global_map, get_map_layers, astar, draw_grid_map,
    global_world_to_grid, in_grid, update_map_from_depth, update_map_from_yolo_bbox,
)
from models import get_yolo_model, get_face_app
from save_load import (
    ensure_save_root, build_camera_map_preview, save_map_bundle, load_map_bundle,
)
from visual_odometry import (
    update_visual_odometry, reset_visual_odometry,
    extract_pose_xytheta, extract_pose_xytheta_from_matrix,
)
from locker import (
    LockerController, QRCodeWindow, create_qr_image, save_qr_png,
    make_delivery_code, make_delivery_payload, decode_qr_from_frame,
)


class RobotMapperGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Smart Delivery Robot AI - Modular RGBD Control")
        self.geometry("1480x840")
        self.minsize(1250, 720)
        self.configure(bg="#0B1120")
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.camera = None
        self.camera_started = False
        self.camera_connecting = False
        self.running = True
        self.frame_count = 0
        self.last_camera_frame_id = -1
        self.last_time = time.time()
        self.fps = 0.0
        self.last_frame = None
        self.last_depth = None
        self.last_intrinsics = None
        self.last_local_map = None
        self.last_global_map = None
        self.last_local_path = []
        self.last_yolo_boxes = []
        self.last_recognized_faces = []
        self.pose_source = "OFF"
        self.robot_pose = None
        self.pose_ok = False
        self.vo_enabled = False
        self.mapping_enabled = False
        self.mapping_status = "Mapping paused"
        self.autosave_minutes = 0.0
        self.last_autosave_ts = time.time()
        self.last_status_update = 0.0
        self.last_process_ts = 0.0
        self.last_render_ts = 0.0
        self.last_safety_state = "CLEAR"
        self.last_safety_distance = None
        self.photo_ref = None

        # Delivery/locker demo state. Simulated by default; can send LOCK/UNLOCK to ESP32 later.
        self.locker = LockerController()
        self.delivery_active = False
        self.delivery_code = None
        self.delivery_payload = None
        self.delivery_qr_window = None

        self._setup_style()
        self._build_ui()
        self.set_mode_ui()
        self.render_placeholder("Camera is OFF\n")

        self.preload_models_async()

        self.after(PROCESS_INTERVAL_MS, self.processing_tick)
    
    def show_loading_overlay(self, message="Loading AI models..."):
        self.loading_win = tk.Toplevel(self)
        self.loading_win.title("Please wait")
        self.loading_win.configure(bg="#111827")
        self.loading_win.resizable(False, False)
        self.loading_win.transient(self)
        self.loading_win.grab_set()

        w, h = 460, 190

        # Force Tkinter to calculate the main window size first
        self.update_idletasks()

        # Get real screen size
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        # Put loading window at the center of the screen
        x = (screen_w - w) // 2
        y = (screen_h - h) // 2

        self.loading_win.geometry(f"{w}x{h}+{x}+{y}")

        # Keep it above the main window
        self.loading_win.lift()
        self.loading_win.attributes("-topmost", True)
        self.loading_win.after(500, lambda: self.loading_win.attributes("-topmost", False))

        tk.Label(
            self.loading_win,
            text="Initializing System",
            bg="#111827",
            fg="#F9FAFB",
            font=("Segoe UI", 16, "bold")
        ).pack(pady=(28, 8))

        self.loading_text_var = tk.StringVar(value=message)

        tk.Label(
            self.loading_win,
            textvariable=self.loading_text_var,
            bg="#111827",
            fg="#CBD5E1",
            font=("Segoe UI", 10)
        ).pack(pady=(0, 14))

        progress = ttk.Progressbar(
            self.loading_win,
            mode="indeterminate",
            length=320
        )
        progress.pack(pady=(0, 18))
        progress.start(12)

        tk.Label(
            self.loading_win,
            text="Controls are disabled during loading.",
            bg="#111827",
            fg="#94A3B8",
            font=("Segoe UI", 9)
        ).pack()

        self.loading_win.protocol("WM_DELETE_WINDOW", lambda: None)
    
    
    def hide_loading_overlay(self):
        try:
            if hasattr(self, "loading_win") and self.loading_win.winfo_exists():
                self.loading_win.grab_release()
                self.loading_win.destroy()
        except Exception:
            pass
    
    
    def update_loading_text(self, text):
        try:
            if hasattr(self, "loading_text_var"):
                self.loading_text_var.set(text)
        except Exception:
            pass
    
    
    
    def _setup_style(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.style.configure("TFrame", background="#0B1120")
        self.style.configure("Card.TFrame", background="#111827")
        self.style.configure("Title.TLabel", background="#0B1120", foreground="#F9FAFB", font=("Segoe UI", 20, "bold"))
        self.style.configure("Sub.TLabel", background="#0B1120", foreground="#94A3B8", font=("Segoe UI", 10))

    def _button(self, parent, text, command, bg="#2563EB"):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg="#FFFFFF",
            activebackground=bg,
            activeforeground="#FFFFFF",
            relief="flat",
            bd=0,
            padx=14,
            pady=10,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2"
        )
        btn.pack(fill="x", pady=5)
        return btn

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=22, pady=(18, 8))
        ttk.Label(header, text="Smart Delivery Robot AI", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Record3D RGBD + LiDAR Safety + Open3D VO + Modular GUI", style="Sub.TLabel").pack(anchor="w", pady=(4, 0))

        body = tk.Frame(self, bg="#0B1120")
        body.pack(fill="both", expand=True, padx=22, pady=12)

        # Left column: only main actions. Mode-specific tools are on the right.
        sidebar = tk.Frame(body, bg="#111827", width=245)
        sidebar.pack(side="left", fill="y", padx=(0, 14))
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="MAIN CONTROL", bg="#111827", fg="#94A3B8", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(18, 8))
        self.start_camera_btn = self._button(sidebar, "Start Camera", self.on_start_camera_button, "#0EA5E9")
        self.drive_btn = self._button(sidebar, "Driving", self.on_drive, "#16A34A")
        self.map_btn = self._button(sidebar, "Map", self.on_map, "#2563EB")

        tk.Frame(sidebar, bg="#1F2937", height=1).pack(fill="x", padx=18, pady=14)
        self._button(sidebar, "Add Person", self.on_add_person, "#7C3AED")
        self._button(sidebar, "Exit", self.on_exit, "#991B1B")

        self.status_panel = tk.Frame(sidebar, bg="#0F172A", padx=12, pady=12)
        self.status_panel.pack(side="bottom", fill="x", padx=14, pady=14)
        self.status_text = tk.StringVar(value="Camera: OFF\nMode: DRIVE")
        tk.Label(self.status_panel, textvariable=self.status_text, bg="#0F172A", fg="#E5E7EB", justify="left", font=("Consolas", 9)).pack(anchor="w")

        # Center preview.
        main = tk.Frame(body, bg="#111827")
        main.pack(side="left", fill="both", expand=True)
        topbar = tk.Frame(main, bg="#111827", padx=16, pady=12)
        topbar.pack(fill="x")
        self.mode_badge = tk.Label(topbar, text="DRIVE", bg="#16A34A", fg="#FFFFFF", padx=14, pady=5, font=("Segoe UI", 10, "bold"))
        self.mode_badge.pack(side="left")
        self.info_label = tk.Label(topbar, text="Camera is off", bg="#111827", fg="#CBD5E1", font=("Segoe UI", 11))
        self.info_label.pack(side="left", padx=14)

        self.preview_frame = tk.Frame(main, bg="#020617", height=GUI_PREVIEW_MAX_H)
        self.preview_frame.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.preview_frame.pack_propagate(False)
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#020617", highlightthickness=0, bd=0)
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<Configure>", self.on_preview_resize)
        self.current_preview_bgr = None

        log_frame = tk.Frame(main, bg="#0F172A", height=105)
        log_frame.pack(fill="x", padx=16, pady=(0, 16))
        log_frame.pack_propagate(False)
        self.log_box = tk.Text(log_frame, bg="#020617", fg="#D1D5DB", insertbackground="#FFFFFF", relief="flat", font=("Consolas", 9), height=6)
        self.log_box.pack(fill="both", expand=True, padx=8, pady=8)

        # Right column: mode-specific actions only.
        self.right_panel = tk.Frame(body, bg="#111827", width=250)
        self.right_panel.pack(side="right", fill="y", padx=(14, 0))
        self.right_panel.pack_propagate(False)
        tk.Label(self.right_panel, text="MODE OPTIONS", bg="#111827", fg="#94A3B8", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(18, 8))
        self.mode_options_container = tk.Frame(self.right_panel, bg="#111827")
        self.mode_options_container.pack(fill="both", expand=True, padx=14, pady=4)

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        print(msg)
        
    def preload_models_async(self):
        self.show_loading_overlay("Loading YOLO model...")

        def worker():
            try:
                self.after(0, lambda: self.log("Preloading YOLO model..."))
                self.after(0, lambda: self.update_loading_text("Loading YOLO model..."))
                get_yolo_model()
                self.after(0, lambda: self.log("YOLO model loaded."))

                self.after(0, lambda: self.log("Preloading InsightFace model..."))
                self.after(0, lambda: self.update_loading_text("Loading InsightFace model..."))
                get_face_app()
                self.after(0, lambda: self.log("InsightFace model loaded."))

                self.after(0, lambda: self.update_loading_text("System ready."))
                self.after(600, self.hide_loading_overlay)

            except Exception as e:
                self.after(0, lambda: self.log(f"Model preload failed: {e}"))
                self.after(0, self.hide_loading_overlay)
                self.after(0, lambda: messagebox.showerror(
                    "Model Loading Error",
                    f"Failed to preload AI models:\n{e}",
                    parent=self
                ))

        threading.Thread(target=worker, daemon=True).start()
    

    def render_placeholder(self, text):
        img = np.zeros((600, 900, 3), dtype=np.uint8)
        img[:] = (23, 6, 2)

        lines = [line for line in text.split("\n") if line.strip()]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (230, 235, 245)

        line_height = 46
        total_text_height = len(lines) * line_height

        start_y = (img.shape[0] - total_text_height) // 2 + line_height

        for i, line in enumerate(lines):
            text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_w, text_h = text_size

            x = (img.shape[1] - text_w) // 2
            y = start_y + i * line_height

            cv2.putText(img, line, (x, y), font, font_scale, color, thickness)

        self.render_preview(img, force=True)
    
    def on_start_camera_button(self):
        if self.camera_started or self.camera_connecting:
            self.on_stop_camera()
        else:
            self.on_start_camera()

    def on_start_camera(self):
        if self.camera_started or self.camera_connecting:
            return
        self.camera_connecting = True
        self.start_camera_btn.configure(text="Connecting...", state="disabled", bg="#64748B")
        self.info_label.configure(text="Connecting to Record3D via USB...")
        self.status_text.set("Camera: connecting...\nMode: " + st.config.get("name", "DRIVE"))
        self.log("Connecting to Record3D via USB...")

        def worker():
            try:
                cam = Record3DCamera()
                cam.start()
                self.after(0, lambda cam=cam: self._camera_connected(cam))
            except Exception as e:
                error = str(e)
                self.after(0, lambda error=error: self._camera_failed(error))

        threading.Thread(target=worker, daemon=True).start()

    def _camera_connected(self, cam):
        self.camera = cam
        self.camera_started = True
        self.camera_connecting = False
        self.last_camera_frame_id = -1
        self.start_camera_btn.configure(text="Stop Camera", state="normal", bg="#DC2626")
        self.info_label.configure(text="Camera connected. Waiting for RGBD frame...")
        self.log("Connected to Record3D via USB.")

    def _camera_failed(self, error):
        self.camera_connecting = False
        self.start_camera_btn.configure(text="Start Camera", state="normal", bg="#0EA5E9")
        self.info_label.configure(text="Camera connection failed")
        self.status_text.set("Camera: OFF\nMode: " + st.config.get("name", "DRIVE"))
        self.log(f"Camera error: {error}")
        messagebox.showerror("Record3D connection failed", str(error), parent=self)


    def _safe_stop_camera(self, cam):
        try:
            cam.stop()
        except Exception as e:
            self.after(0, lambda e=str(e): self.log(f"Camera stop warning: {e}"))
            
            
    def on_stop_camera(self):
        if self.camera_connecting:
            messagebox.showinfo(
                "Camera",
                "Camera is still connecting. Wait a moment, then stop again if needed.",
                parent=self
            )
            return

        cam = self.camera
        self.camera = None
        self.camera_started = False
        self.last_camera_frame_id = -1

        if self.delivery_active:
            self.delivery_active = False
            self.delivery_code = None
            self.delivery_payload = None
            self.locker.unlock()
            self.log("Camera stopped during delivery. Locker UNLOCKED for safety.")

        self.last_frame = None
        self.last_depth = None
        self.vo_enabled = False
        self.mapping_enabled = False
        self.mapping_status = "Mapping paused"

        try:
            reset_visual_odometry()
        except Exception as e:
            self.log(f"VO reset warning: {e}")

        self.start_camera_btn.configure(text="Start Camera", state="normal", bg="#0EA5E9")
        self.info_label.configure(text="Camera stopped")
        self.status_text.set(f"Camera: OFF\nMode: {st.config.get('name')}")
        self.render_placeholder("Camera is OFF")
        self.set_mode_ui()
        self.log("Camera stopped.")

        if cam is not None:
            threading.Thread(target=self._safe_stop_camera, args=(cam,), daemon=True).start()

    def set_mode_ui(self):
        mode = st.config.get("name", "DRIVE")
        for child in self.mode_options_container.winfo_children():
            child.destroy()

        if mode == "DRIVE":
            self.mode_badge.configure(text="DRIVE", bg="#16A34A")
            self.info_label.configure(text="Driving: camera + compact LiDAR safety" if self.camera_started else "Camera is off")
            delivery_text = "Cancel Delivery" if self.delivery_active else "Delivery"
            delivery_color = "#DC2626" if self.delivery_active else "#F97316"
            self._button(self.mode_options_container, delivery_text, self.on_delivery_button, delivery_color)
            self._button(self.mode_options_container, "Hide LiDAR Zones" if st.show_lidar_zones else "Show LiDAR Zones", self.on_toggle_lidar_zones, "#334155")
            lock_line = "LOCKED" if self.locker.locked else "UNLOCKED"
            hint_text = f"Drive mode keeps lightweight LiDAR safety. Locker: {lock_line}. Press Delivery to lock locker and create a QR unlock code."
        else:
            self.mode_badge.configure(text="MAP", bg="#2563EB")
            self.info_label.configure(text="Map mode: start VO/mapping from the right panel")
            self._button(self.mode_options_container, "Stop VO" if self.vo_enabled else "Start VO", self.on_toggle_vo, "#D97706")
            self._button(self.mode_options_container, "Pause Mapping" if self.mapping_enabled else "Start Mapping", self.on_toggle_mapping, "#0F766E")
            self._button(self.mode_options_container, "Save Map", self.on_save_map, "#0EA5E9")
            self._button(self.mode_options_container, "Load Map", self.on_load_map, "#0891B2")
            self._button(self.mode_options_container, "Reset Map", self.on_reset, "#DC2626")
            self._button(self.mode_options_container, "Hide Camera Panel" if st.show_camera_side_panel else "Show Camera Panel", self.on_toggle_camera_side_panel, "#334155")
            self._button(self.mode_options_container, "Hide LiDAR Zones" if st.show_lidar_zones else "Show LiDAR Zones", self.on_toggle_lidar_zones, "#334155")
            hint_text = "When mapping is active, LiDAR boxes are hidden automatically to keep the preview clean."

        hint = tk.Label(
            self.mode_options_container,
            text=hint_text,
            bg="#111827", fg="#94A3B8", wraplength=210, justify="left",
            font=("Segoe UI", 9)
        )
        hint.pack(fill="x", pady=(14, 4))

    def on_delivery_button(self):
        if self.delivery_active:
            self.cancel_delivery()
        else:
            self.start_delivery()

    def start_delivery(self):
        if st.config.get("name") != "DRIVE":
            messagebox.showinfo("Delivery", "Switch to DRIVE mode before starting delivery.", parent=self)
            return
        if not self.camera_started:
            messagebox.showinfo("Camera", "Click Start Camera first. The camera is needed to scan the QR code.", parent=self)
            return

        self.delivery_code = make_delivery_code()
        self.delivery_payload = make_delivery_payload(self.delivery_code)
        self.delivery_active = True
        self.locker.lock()

        saved_path = save_qr_png(self.delivery_payload, self.delivery_code)
        qr_image = create_qr_image(self.delivery_payload)
        try:
            if self.delivery_qr_window is not None and self.delivery_qr_window.winfo_exists():
                self.delivery_qr_window.destroy()
        except Exception:
            pass
        self.delivery_qr_window = QRCodeWindow(self, qr_image, self.delivery_code, self.delivery_payload, saved_path)

        self.log(f"Delivery started. Locker LOCKED. QR code: {self.delivery_code}")
        self.info_label.configure(text="Delivery active: show the QR to the camera to unlock locker")
        self.set_mode_ui()

    def cancel_delivery(self):
        self.delivery_active = False
        self.delivery_code = None
        self.delivery_payload = None
        self.locker.unlock()
        self.log("Delivery cancelled. Locker UNLOCKED.")
        self.info_label.configure(text="Delivery cancelled")
        self.set_mode_ui()

    def complete_delivery_unlock(self):
        code = self.delivery_code
        self.delivery_active = False
        self.delivery_code = None
        self.delivery_payload = None
        self.locker.unlock()
        self.log(f"QR matched ({code}). Locker UNLOCKED.")
        self.info_label.configure(text="QR matched. Locker unlocked.")
        try:
            if self.delivery_qr_window is not None and self.delivery_qr_window.winfo_exists():
                self.delivery_qr_window.destroy()
        except Exception:
            pass
        #essagebox.showinfo("Delivery", "QR matched. Locker unlocked.", parent=self)
        self.set_mode_ui()

    def on_toggle_mapping(self):
        if st.config.get("name") != "MAP":
            messagebox.showinfo("Mapping", "Switch to MAP mode before starting mapping.", parent=self)
            return
        if not self.camera_started:
            messagebox.showinfo("Camera", "Click Start Camera before mapping.", parent=self)
            return
        self.mapping_enabled = not self.mapping_enabled
        if self.mapping_enabled:
            self.mapping_status = "Mapping active"
            self.log("Mapping started. LiDAR debug boxes will be hidden for a cleaner preview.")
        else:
            self.mapping_status = "Mapping paused"
            self.log("Mapping paused. Map will not update.")
        self.set_mode_ui()

    def on_toggle_lidar_zones(self):
        toggle_lidar_zones()
        self.set_mode_ui()
        self.log(f"LiDAR zones {'shown' if st.show_lidar_zones else 'hidden'}.")

    def on_toggle_camera_side_panel(self):
        st.show_camera_side_panel = not st.show_camera_side_panel
        self.set_mode_ui()
        self.log(f"Camera side panel {'shown' if st.show_camera_side_panel else 'hidden'}.")

    def on_drive(self):
        st.config = DRIVE_CONFIG.copy()
        self.vo_enabled = False
        self.mapping_enabled = False
        self.mapping_status = "Mapping paused"
        st.config["use_vo"] = False
        self.last_yolo_boxes = []
        self.last_recognized_faces = []
        self.set_mode_ui()
        self.log("Switched to DRIVE mode. Heavy AI/mapping disabled for lower latency.")

    def on_map(self):
        st.config = MAP_CONFIG.copy()
        st.config["use_vo"] = self.vo_enabled
        self.last_yolo_boxes = []
        self.last_recognized_faces = []
        self.mapping_enabled = False
        self.mapping_status = "Mapping paused. Press Start Mapping to update map."
        self.set_mode_ui()
        self.log("Switched to MAP mode. Map actions are on the right panel.")

    def on_toggle_vo(self):
        if st.config.get("name") != "MAP":
            messagebox.showinfo("VO", "Please switch to MAP mode before enabling Visual Odometry.", parent=self)
            return
        if not self.camera_started:
            messagebox.showinfo("Camera", "Click Start Camera before enabling VO.", parent=self)
            return
        self.vo_enabled = not self.vo_enabled
        st.config["use_vo"] = self.vo_enabled
        if not self.vo_enabled:
            reset_visual_odometry()
        self.set_mode_ui()
        self.log("Visual Odometry ON." if self.vo_enabled else "Visual Odometry OFF and reset.")

    def on_add_person(self):
        if not self.camera_started or self.last_frame is None:
            messagebox.showwarning("No camera frame", "Start the camera first, then stand in front of it.", parent=self)
            return
        dialog = AddPersonDialog(self)
        name = dialog.result
        if not name:
            return
        ok, msg = add_new_person_with_name(self.last_frame, name)
        self.log(msg)
        if ok:
            messagebox.showinfo("Face profile saved", msg, parent=self)
        else:
            messagebox.showwarning("Face profile not saved", msg, parent=self)

    def on_save_map(self):
        dialog = SaveMapDialog(self)
        info = dialog.result
        if not info:
            return
        self.autosave_minutes = float(info.get("autosave_minutes", 0) or 0)
        try:
            folder = save_map_bundle(info, self.last_local_map, self.last_global_map, self.last_frame, self.last_depth, self.last_intrinsics, self.pose_source, self.robot_pose)
            self.last_autosave_ts = time.time()
            self.log(f"Saved map bundle: {folder}")
            messagebox.showinfo("Map saved", f"Saved to:\n{folder}", parent=self)
        except Exception as e:
            self.log(f"Save map failed: {e}")
            messagebox.showerror("Save failed", str(e), parent=self)

    def on_load_map(self):
        folder = filedialog.askdirectory(title="Choose saved map folder", initialdir=SAVE_ROOT if os.path.exists(SAVE_ROOT) else os.getcwd(), parent=self)
        if not folder:
            return
        try:
            metadata = load_map_bundle(folder)
            reset_visual_odometry()
            self.mapping_enabled = False
            self.mapping_status = "Loaded map. Mapping paused."
            self.last_local_map = None
            self.last_global_map = None
            self.last_local_path = []
            self.log(f"Loaded map: {folder}")
            if metadata:
                self.log(f"Loaded metadata: {metadata.get('project_note', '')}")
            self.force_redraw_maps()
            self.render_preview(self.last_global_map if self.last_global_map is not None else self.last_local_map, force=True)
            messagebox.showinfo("Map loaded", "Map arrays loaded. Continue mapping near the original start position.", parent=self)
        except Exception as e:
            self.log(f"Load map failed: {e}")
            messagebox.showerror("Load failed", str(e), parent=self)

    

    def on_reset(self):
        if not messagebox.askyesno("Reset map", "Clear all map data and reset visual odometry?", parent=self):
            return
        reset_local_map()
        reset_global_map()
        reset_visual_odometry()
        self.mapping_enabled = False
        self.mapping_status = "Mapping paused"
        self.last_local_map = None
        self.last_global_map = None
        self.last_local_path = []
        self.log("Reset all maps and visual odometry.")
        self.set_mode_ui()

    def on_exit(self):
        self.running = False
        try:
            if self.camera is not None:
                self.camera.stop()
        except Exception:
            pass
        try:
            self.locker.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.destroy()

    def force_redraw_maps(self):
        local_free_grid, local_occupied_grid, local_inflated_grid = get_map_layers(st.local_occ, st.local_free, INFLATION_RADIUS_CELLS)
        self.last_local_path = astar(local_inflated_grid, LOCAL_ROBOT_CELL, LOCAL_GOAL_CELL)
        self.last_local_map = draw_grid_map(
            local_free_grid, local_occupied_grid, local_inflated_grid,
            self.last_local_path, LOCAL_ROBOT_CELL, LOCAL_GOAL_CELL,
            scale=3, title="LOCAL MAP: green=free red=obstacle black=unknown"
        )

        global_free_grid, global_occupied_grid, global_inflated_grid = get_map_layers(st.global_occ, st.global_free, INFLATION_RADIUS_CELLS)
        robot_cell = GLOBAL_ORIGIN_CELL
        if self.pose_ok and self.robot_pose is not None:
            gx, gy = global_world_to_grid(self.robot_pose[0], self.robot_pose[1])
            if in_grid(gx, gy, GLOBAL_GRID_W, GLOBAL_GRID_H):
                robot_cell = (gx, gy)
        self.last_global_map = draw_grid_map(
            global_free_grid, global_occupied_grid, global_inflated_grid,
            [], robot_cell, GLOBAL_ORIGIN_CELL,
            scale=2, title="GLOBAL MAP: saved/loadable occupancy grid"
        )

    def on_preview_resize(self, event=None):
        if self.current_preview_bgr is not None:
            self.render_preview(self.current_preview_bgr, force=True)

    def render_preview(self, img_bgr, force=False):
        if img_bgr is None:
            return
        self.current_preview_bgr = img_bgr
        if not force and self.frame_count % RENDER_EVERY_N_FRAMES != 0:
            return
        now = time.time()
        if not force and DISPLAY_MAX_FPS > 0 and (now - self.last_render_ts) < (1.0 / DISPLAY_MAX_FPS):
            return
        self.last_render_ts = now

        h, w = img_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return

        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        if canvas_w <= 2 or canvas_h <= 2:
            canvas_w = GUI_PREVIEW_W
            canvas_h = GUI_PREVIEW_H

        max_w = max(300, min(canvas_w, GUI_PREVIEW_MAX_W))
        max_h = max(220, min(canvas_h, GUI_PREVIEW_MAX_H))
        scale = min(max_w / w, max_h / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo_ref = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.preview_canvas.delete("all")
        x = max(0, (canvas_w - new_w) // 2)
        y = max(0, (canvas_h - new_h) // 2)
        self.preview_canvas.create_image(x, y, image=self.photo_ref, anchor="nw")

    def processing_tick(self):
        if not self.running:
            return
        try:
            self.process_one_frame()
        except Exception as e:
            self.log(f"Processing error: {e}")
        self.after(PROCESS_INTERVAL_MS, self.processing_tick)

    def process_one_frame(self):
        if self.camera is None:
            now = time.time()
            if now - self.last_status_update > 0.5:
                self.status_text.set(f"Camera: OFF\nMode: {st.config.get('name')}\nMapping: {self.mapping_status}")
                self.last_status_update = now
            return

        # v3 read_latest returns a frame_id. If frame_id did not change, skip all processing.
        if hasattr(self.camera, "read_latest"):
            ret, frame_raw, depth_raw, intrinsics, pose, frame_id = self.camera.read_latest()
        else:
            ret, frame_raw, depth_raw, intrinsics, pose = self.camera.read()
            frame_id = self.frame_count + 1

        if not ret:
            now = time.time()
            if now - self.last_status_update > 0.5:
                self.status_text.set(f"Camera: waiting frame...\nMode: {st.config.get('name')}\nMapping: {self.mapping_status}")
                self.last_status_update = now
            return

        if frame_id == self.last_camera_frame_id:
            return
        self.last_camera_frame_id = frame_id

        now = time.time()
        if MAX_PROCESS_FPS > 0 and (now - self.last_process_ts) < (1.0 / MAX_PROCESS_FPS):
            # Drop frames instead of building UI/processing backlog.
            return
        self.last_process_ts = now

        self.frame_count += 1
        dt = now - self.last_time
        if dt > 0:
            self.fps = 0.85 * self.fps + 0.15 * (1.0 / dt)
        self.last_time = now

        frame, depth_frame = resize_frame_and_depth(frame_raw, depth_raw, MAX_PROCESS_FRAME_W)
        self.last_frame = frame.copy()
        self.last_depth = depth_frame.copy()
        self.last_intrinsics = intrinsics

        frame_h, frame_w = frame.shape[:2]
        depth_ok, depth_valid_ratio, depth_median = get_depth_health(depth_frame)
        center_depth = get_center_lidar_depth(depth_frame)
        if self.frame_count % SAFETY_EVERY_N_FRAMES == 0 or self.last_safety_distance is None:
            raw_safety_state, raw_safety_distance = get_front_safety(depth_frame)
            safety_state, safety_distance = apply_safety_hold(raw_safety_state, raw_safety_distance)
            self.last_safety_state = safety_state
            self.last_safety_distance = safety_distance
        else:
            safety_state = self.last_safety_state
            safety_distance = self.last_safety_distance

        record3d_robot_pose = extract_pose_xytheta(pose)
        vo_pose_matrix = None
        if st.config.get("use_vo", False):
            if self.frame_count % st.config["vo_every"] == 0:
                vo_pose_matrix = update_visual_odometry(frame, depth_frame, intrinsics)
            elif st.vo_pose_ok:
                vo_pose_matrix = st.vo_global_pose.copy()
        vo_robot_pose = extract_pose_xytheta_from_matrix(vo_pose_matrix)

        if record3d_robot_pose is not None:
            self.robot_pose = record3d_robot_pose
            self.pose_source = "Record3D"
        elif vo_robot_pose is not None:
            self.robot_pose = vo_robot_pose
            self.pose_source = "VO"
        else:
            self.robot_pose = None
            self.pose_source = "OFF"
        self.pose_ok = self.robot_pose is not None

        human_count = 0
        obstacle_count = 0

        can_update_local_map = (
            st.config.get("name") == "MAP"
            and self.mapping_enabled
            and depth_ok
            and self.frame_count % st.config["local_map_every"] == 0
        )
        can_update_global_map = (
            can_update_local_map
            and self.pose_ok
            and self.frame_count % st.config["global_map_every"] == 0
        )

        if st.config.get("name") == "MAP" and self.mapping_enabled:
            if not depth_ok:
                self.mapping_status = f"Paused: bad/no depth ({depth_valid_ratio*100:.1f}% valid)"
            elif REQUIRE_POSE_FOR_GLOBAL_MAP and not self.pose_ok:
                self.mapping_status = "Local only: waiting for VO/Pose"
            else:
                self.mapping_status = "Mapping active"

        if can_update_local_map:
            update_map_from_depth(
                depth_frame, frame_w, frame_h, st.config["depth_step"],
                robot_pose=self.robot_pose, update_global=can_update_global_map
            )

        # Face recognition normally only runs when manually adding a person.
        if self.frame_count % st.config["face_every"] == 0:
            self.last_recognized_faces = []
            face_app = get_face_app()
            faces = face_app.get(frame)
            for face in faces:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                embedding = face.embedding
                embedding = embedding / np.linalg.norm(embedding)
                person_name, face_score = recognize_face(embedding)
                self.last_recognized_faces.append({"bbox": (fx1, fy1, fx2, fy2), "name": person_name, "score": face_score})

        # YOLO is disabled by default for realtime. Set DRIVE_CONFIG["yolo_every"] lower in config.py if needed.
        if self.frame_count % st.config["yolo_every"] == 0:
            self.last_yolo_boxes = []
            yolo_model = get_yolo_model()
            results = yolo_model(frame, device=YOLO_DEVICE, conf=YOLO_CONF, imgsz=st.config["yolo_imgsz"], verbose=False)
            for r in results:
                for box_obj in r.boxes:
                    cls_id = int(box_obj.cls[0])
                    obj_name = yolo_model.names[cls_id]
                    conf = float(box_obj.conf[0])
                    x1, y1, x2, y2 = map(int, box_obj.xyxy[0])
                    self.last_yolo_boxes.append((obj_name, conf, x1, y1, x2, y2))

        for obj_name, conf, x1, y1, x2, y2 in self.last_yolo_boxes:
            lidar_distance = get_lidar_distance_in_bbox(depth_frame, x1, y1, x2, y2)
            status, status_color = get_distance_status(lidar_distance)
            is_person = obj_name == "person"

            if lidar_distance is not None and st.config.get("name") == "MAP" and self.mapping_enabled and depth_ok:
                update_map_from_yolo_bbox(
                    x1, x2, frame_w, lidar_distance, is_person,
                    robot_pose=self.robot_pose,
                    update_global=(self.pose_ok and self.frame_count % st.config["global_map_every"] == 0)
                )
                zone_name = get_lidar_zone_name_for_pixel((x1 + x2) / 2, (y1 + y2) / 2, frame_w, frame_h)
                mark_lidar_zone_obstacle(zone_name, lidar_distance)

            if is_person:
                human_count += 1
                display_name = "Unknown"
                for f in self.last_recognized_faces:
                    fx1, fy1, fx2, fy2 = f["bbox"]
                    face_center_x = (fx1 + fx2) // 2
                    face_center_y = (fy1 + fy2) // 2
                    if x1 <= face_center_x <= x2 and y1 <= face_center_y <= y2:
                        display_name = f["name"]
                        break
                title = f"PERSON: {display_name}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 1)
                label = f"{title} | LiDAR --" if lidar_distance is None else f"{title} | {lidar_distance:.2f}m | {status}"
                draw_simple_label(frame, x1, y1, label, status_color)
            else:
                obstacle_count += 1

        if self.frame_count % st.config["draw_map_every"] == 0:
            self.force_redraw_maps()

        if (
            self.delivery_active
            and self.delivery_payload
            and st.config.get("name") == "DRIVE"
            and self.frame_count % QR_SCAN_EVERY_N_FRAMES == 0
        ):
            for qr_text in decode_qr_from_frame(frame):
                if qr_text.strip() == self.delivery_payload:
                    self.complete_delivery_unlock()
                    break

        path_found = len(self.last_local_path) > 0
        hide_debug_for_mapping = False
        # hide_debug_for_mapping = (
        #     HIDE_LIDAR_OVERLAY_WHILE_MAPPING
        #     and st.config.get("name") == "MAP"
        #     and self.mapping_enabled
        # )

        if st.show_lidar_zones:
            draw_front_zones(frame, compact=True, show_caption=False)

        draw_dashboard(
            frame, st.config["name"], human_count, obstacle_count, path_found,
            center_depth, safety_distance, safety_state, self.fps, self.pose_ok, self.pose_source
        )

        draw_safety_badge(frame, safety_state, safety_distance)

        if self.delivery_active:
            locker_text = f"DELIVERY LOCKED | scan QR {self.delivery_code}"
            cv2.putText(
                frame,
                locker_text,
                (8, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 220, 255),
                1
            )

        if st.config.get("name") == "MAP":
            if self.last_global_map is None or self.last_local_map is None:
                self.force_redraw_maps()
            if st.show_camera_side_panel:
                preview = build_camera_map_preview(frame, self.last_local_map, self.last_global_map)
            else:
                preview = self.last_global_map if self.last_global_map is not None else self.last_local_map
        else:
            preview = frame
        self.render_preview(preview)

        safety_text = "unknown" if safety_distance is None else f"{safety_distance:.2f}m"
        center_text = "unknown" if center_depth is None else f"{center_depth:.2f}m"
        if now - self.last_status_update > 0.18:
            self.status_text.set(
                f"Camera: ON\n"
                f"Mode: {st.config.get('name')}\n"
                f"FPS: {self.fps:.1f}\n"
                f"Front: {safety_text}\n"
                f"Center: {center_text}\n"
                f"Mapping: {self.mapping_status}\n"
                f"Locker: {'LOCKED' if self.locker.locked else 'UNLOCKED'}\n"
                f"Pose: {self.pose_source}"
            )
            self.last_status_update = now

        if self.autosave_minutes > 0 and (time.time() - self.last_autosave_ts) >= self.autosave_minutes * 60:
            info = {"user_name": "autosave", "project_note": "Automatic save", "autosave_minutes": self.autosave_minutes}
            folder = save_map_bundle(info, self.last_local_map, self.last_global_map, self.last_frame, self.last_depth, self.last_intrinsics, self.pose_source, self.robot_pose)
            self.last_autosave_ts = time.time()
            self.log(f"Autosaved map: {folder}")

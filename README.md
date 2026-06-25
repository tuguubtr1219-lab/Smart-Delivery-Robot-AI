# Smart Delivery Robot AI - Modular Version
```bash
#before run we need a little bit set up:you need to run :
pip install -r requirements.txt 
#or
python -m pip install -r requirements.txt
#or
python -m pip install opencv-python numpy open3d ultralytics insightface onnxruntime record3d Pillow "qrcode[pil]" pyserial
#before run main.py
#if you want to start camera you need to download record3d in you phone then connect with computer by usb
```

Run:

```bash
python main.py
```

idea:

- The app opens with camera OFF.
- Click **Start Camera** to connect to Record3D through USB.
- Left column only keeps main actions.
- Right column changes based on mode:
  - **Driving**: LiDAR zone show/hide.
  - **Map**: Start/Stop VO, Start/Pause Mapping, Camera Panel show/hide, LiDAR zone show/hide.
- YOLO and InsightFace now lazy-load only when needed.
- RGB + depth are downscaled early for lower latency.
- Map drawing, VO, YOLO, and mapping updates run less frequently to reduce lag.

note:

- `main.py`: app entry point
- `gui.py`: Tkinter GUI and processing loop
- `config.py`: all thresholds and performance settings
- `app_state.py`: shared runtime state and map arrays
- `camera_record3d.py`: Record3D USB camera wrapper
- `models.py`: lazy YOLO / InsightFace loading
- `face_recognition_utils.py`: face registration and recognition
- `depth_utils.py`: depth/LiDAR helpers
- `lidar_safety.py`: front-zone safety logic and overlay
- `visual_odometry.py`: Open3D RGBD VO
- `mapping.py`: occupancy mapping, A*, map drawing
- `pointcloud.py`: PLY export
- `save_load.py`: map bundle save/load
- `dialogs.py`: Tkinter dialogs
- `drawing.py`: overlay UI drawing helpers



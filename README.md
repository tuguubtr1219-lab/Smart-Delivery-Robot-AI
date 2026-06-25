# Smart Delivery Robot AI - Modular Version  
  ## Overview

This project is an AI-powered indoor delivery robot designed for autonomous delivery in indoor environments.

Features:

- RGBD Camera Input
- Face Recognition
- LiDAR Safety Detection
- Visual Odometry
- Occupancy Mapping
- Autonomous Navigation
- Graphical User Interface (GUI)

The robot can recognize users, monitor obstacles, generate maps, and navigate safely to a destination.

## Hardware Requirements

- Record3D compatible mobile device
- RGBD camera
- LiDAR sensor
- Windows 10/11
- Python 3.11

## Installation

```bash
pip install -r requirements.txt

```

## Run

```bash
python main.py
```

## Design Concept

* The application starts with the camera disabled by default.
* Users can connect a Record3D camera through USB and start real-time data acquisition.
* Face recognition is used for user identification and registration.
* LiDAR safety detection helps avoid obstacles and unsafe navigation paths.
* Visual odometry is used to estimate robot movement and position.
* Occupancy mapping generates a navigable map of the environment.
* The graphical user interface allows users to control navigation, mapping, and monitoring functions.

### Future Improvements

* Real-time autonomous path planning.
* Cloud-based map storage and synchronization.
* Voice command integration.
* Multi-user recognition support.
* Improved obstacle avoidance using AI-based object detection.


## Project Structure

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

## Authors

Providence University  
Department of Computer Science and Information Engineering

Team Project (2026)

Team Members:
- Tuguldur Batbayar
- 許俊成

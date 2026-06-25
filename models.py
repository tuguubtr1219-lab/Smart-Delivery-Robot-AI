"""Lazy model loading to keep GUI startup fast."""
import os
import pickle

from config import DB_FILE, YOLO_MODEL

_yolo_model = None
_face_app = None
_face_db = None


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        print("Loading YOLO...")
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
    return _yolo_model


def get_face_app():
    global _face_app
    if _face_app is None:
        print("Loading InsightFace...")
        from insightface.app import FaceAnalysis
        _face_app = FaceAnalysis(name="buffalo_l")
        _face_app.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app


def get_face_db():
    global _face_db
    if _face_db is None:
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "rb") as f:
                _face_db = pickle.load(f)
        else:
            _face_db = {}
    return _face_db


def save_face_db():
    with open(DB_FILE, "wb") as f:
        pickle.dump(get_face_db(), f)

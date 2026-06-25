import numpy as np

from config import SIMILARITY_THRESHOLD
from models import get_face_app, get_face_db, save_face_db


def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(a / na, b / nb))


def recognize_face(embedding):
    face_db = get_face_db()
    best_name = "Unknown"
    best_score = -1.0

    for name, data in face_db.items():
        if isinstance(data, np.ndarray):
            embeddings = [data]
        elif isinstance(data, list):
            embeddings = data
        elif isinstance(data, dict) and "embedding" in data:
            embeddings = [data["embedding"]]
        elif isinstance(data, dict) and "embeddings" in data:
            embeddings = data["embeddings"]
        else:
            continue

        for saved_embedding in embeddings:
            score = cosine_similarity(embedding, saved_embedding)
            if score > best_score:
                best_score = score
                best_name = name

    if best_score >= SIMILARITY_THRESHOLD:
        return best_name, best_score
    return "Unknown", best_score


def add_new_person_with_name(frame, name):
    """Register a face using a GUI-provided name."""
    if frame is None:
        return False, "No camera frame available."

    face_app = get_face_app()
    faces = face_app.get(frame)
    if len(faces) == 0:
        return False, "No face detected. Please face the camera clearly."

    face = max(
        faces,
        key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])
    )

    embedding = face.embedding
    embedding = embedding / np.linalg.norm(embedding)

    face_db = get_face_db()
    if name in face_db:
        old = face_db[name]
        if isinstance(old, list):
            old.append(embedding)
        else:
            face_db[name] = [old, embedding]
    else:
        face_db[name] = embedding

    save_face_db()
    return True, f"Saved face profile for: {name}"

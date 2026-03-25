import os
import numpy as np
import cv2
from deepface import DeepFace
from secure_storage import load_encrypted

# Cache embeddings for speed
database = {}

def load_all_faces():
    files = []

    # Load encrypted faces
    for f in os.listdir("faces"):
        files.append(os.path.join("faces", f))

    return files


def load_image(file_path):
    data = load_encrypted(file_path)
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def get_embedding(image):
    try:
        embedding = DeepFace.represent(
            image,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]
        return np.array(embedding)
    except:
        return None


def build_database():
    global database

    print("[INFO] Building face database...")

    for file_path in load_all_faces():
        name = os.path.basename(file_path).split(".")[0]

        img = load_image(file_path)
        emb = get_embedding(img)

        if emb is None:
            continue

        if name not in database:
            database[name] = []

        database[name].append(emb)

    print("✅ Database ready")


def recognize_face(face_img):
    if not database:
        build_database()

    query_emb = get_embedding(face_img)
    if query_emb is None:
        return "Unknown", False, 0

    best_match = "Unknown"
    best_distance = float("inf")

    for name, embeddings in database.items():
        for emb in embeddings:
            dist = np.linalg.norm(query_emb - emb)

            if dist < best_distance:
                best_distance = dist
                best_match = name

    threshold = 0.9 
    
    print(f"Match: {best_match}, Distance: {best_distance}")
    print(f"FINAL → {best_match}, distance: {best_distance}")
    if best_distance < threshold:
        confidence = max(0, (1 - best_distance) * 100)
        return best_match, True, confidence
    else:
        return "Unknown", False, 0
import os
import numpy as np
import cv2
from deepface import DeepFace
from secure_storage import load_encrypted

# Cache embeddings
database = {}

def load_all_faces():
    files = []

    for f in os.listdir("faces"):
        files.append(os.path.join("faces", f))

    return files


def load_image(file_path):
    data = load_encrypted(file_path)
    nparr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def get_embedding(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        embedding = DeepFace.represent(
            image,
            model_name="Facenet",
            enforce_detection=True
        )[0]["embedding"]
        emb = np.array(embedding)

        # 🔥 NORMALIZATION (CRITICAL FIX)
        emb = emb / np.linalg.norm(emb)

        return emb

    except:
        return None


def build_database():
    global database

    print("[INFO] Building face database...")

    for file_path in load_all_faces():
        name = os.path.basename(file_path).split(".")[0]

        img = load_image(file_path)
        if img is None:
            continue

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
    second_best = float("inf") 

    for name, embeddings in database.items():
        for emb in embeddings:
            dist = 1 - np.dot(query_emb, emb)

            if dist < best_distance:
                second_best = best_distance
                best_distance = dist
                best_match = name
            elif dist < second_best:
                second_best = dist

    threshold = 0.4  # can tune later
    print(f"{best_match} → best: {best_distance:.3f}, second: {second_best:.3f}")

    # 🔥 FINAL DECISION 
    if best_distance < threshold:
        confidence = max(0, 100 - best_distance * 100)
        return best_match, True, confidence
    else:
        return "Unknown", False, 0
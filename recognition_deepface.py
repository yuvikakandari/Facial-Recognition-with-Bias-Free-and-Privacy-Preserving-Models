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
        files.append(("custom", os.path.join("faces", f)))

    lfw_path = "lfw-deepfunneled"

    if os.path.exists(lfw_path):
        for person in os.listdir(lfw_path):
            person_path = os.path.join(lfw_path, person)

            if not os.path.isdir(person_path):
                continue

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                files.append(("lfw", img_path))

    return files

def load_image(source, file_path):
    if source == "custom":
        data = load_encrypted(file_path)
        nparr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    elif source == "lfw":
        return cv2.imread(file_path)

def get_embedding(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = DeepFace.represent(
            img_path=image,
            model_name="Facenet",
            enforce_detection=False  
        )

        embedding = result[0]["embedding"]
        emb = np.array(embedding)

        #  NORMALIZATION
        emb = emb / np.linalg.norm(emb)

        return emb

    except:
        return None


def build_database():
    global database

    print("[INFO] Building face database...")

    for source, file_path in load_all_faces():

        if source == "custom":
            name = os.path.basename(file_path).split(".")[0]

        elif source == "lfw":
            name = os.path.basename(os.path.dirname(file_path))

        img = load_image(source, file_path)
        if img is None:
            continue

        emb = get_embedding(img)
        if emb is None:
            continue

        if name not in database:
            database[name] = []

        # limit images per person 
        if len(database[name]) < 8:
            database[name].append(emb)

        database[name].append(emb)

    print(f"✅ Database ready with {len(database)} people")
    print("\n--- DATABASE CONTENT ---")
    for name in database:
        print(name, len(database[name]))
    print("------------------------\n")


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

    threshold = 0.6  # can tune later
    print(f"BEST MATCH: {best_match} → best: {best_distance:.3f}, second: {second_best:.3f}")

    # FINAL DECISION 
    if best_distance < threshold:
        confidence = max(0, 100 - best_distance * 100)
        return best_match, True, confidence
    else:
        return "Unknown", False, 0
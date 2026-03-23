from deepface import DeepFace
import os
import numpy as np
import cv2
from secure_storage import load_encrypted

FACES_DIR = "faces"

def get_embedding(image):
    embedding = DeepFace.represent(image, model_name="Facenet")[0]["embedding"]
    return np.array(embedding)

def recognize_face(face_img):
    try:
        input_embedding = get_embedding(face_img)

        for file in os.listdir(FACES_DIR):
            if file.endswith(".enc"):
                path = os.path.join(FACES_DIR, file)

                # 🔓 LOAD + DECRYPT IMAGE
                image_bytes = load_encrypted(path)

                # Convert bytes → image
                nparr = np.frombuffer(image_bytes, np.uint8)
                stored_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                stored_embedding = get_embedding(stored_img)

                # Compare embeddings
                distance = np.linalg.norm(input_embedding - stored_embedding)

                if distance < 10:  # threshold (adjust if needed)
                    name = file.split(".")[0]
                    return name, True

        return "Unknown", False

    except Exception as e:
        print("Error:", e)
        return "Unknown", False
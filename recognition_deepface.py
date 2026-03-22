from deepface import DeepFace
import cv2

def recognize_face(frame, db_path="dataset"):
    try:
        result = DeepFace.find(
            img_path=frame,
            db_path=db_path,
            enforce_detection=False
        )

        if len(result) > 0 and len(result[0]) > 0:
            identity = result[0].iloc[0]['identity']
            name = identity.split("\\")[-2]  # folder name
            return name, True
        else:
            return "Unknown", False

    except:
        return "Unknown", False
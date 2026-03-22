from deepface import DeepFace

def recognize_face(frame, db_path="dataset"):
    try:
        result = DeepFace.find(
            img_path=frame,
            db_path=db_path,
            enforce_detection=False
        )

        if len(result) > 0 and len(result[0]) > 0:
            identity = result[0].iloc[0]['identity']
            distance = result[0].iloc[0]['distance']

            name = identity.split("\\")[-2]

            # Convert distance → confidence
            confidence = 1 - distance

            return name, True, confidence
        else:
            return "Unknown", False, 0

    except:
        return "Unknown", False, 0
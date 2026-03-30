import cv2
import json

# Load trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.xml")

# Load label map
with open("labels.json", "r") as f:
    label_map = json.load(f)

label_map = {int(k): v for k, v in label_map.items()}

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

print("Press ESC to exit")

last_name = "Unknown"
stable_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        label, confidence = model.predict(face)

        if confidence < 85:
            name = label_map[label]
        else:
            name = "Unknown"

            # Blur unknown faces (privacy)
            face_region = frame[y:y+h, x:x+w]
            face_region = cv2.GaussianBlur(face_region, (99,99), 30)
            frame[y:y+h, x:x+w] = face_region

        text = f"{name} ({confidence:.2f})"

        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
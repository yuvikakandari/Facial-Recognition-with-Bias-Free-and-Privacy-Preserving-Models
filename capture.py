import cv2
import os

# Ask for person name
person_name = input("Enter person name: ")

# Create dataset folder
dataset_path = os.path.join("dataset", person_name)
os.makedirs(dataset_path, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam")
    exit()

count = 0
print("Press SPACE to capture images, ESC to exit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        face = gray[y:y+h, x:x+w]

        cv2.imshow("Capture Faces", frame)

        key = cv2.waitKey(1)

        if key == 32:  # SPACE
            count += 1
            file_path = os.path.join(dataset_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)
            print(f"Saved {file_path}")

        elif key == 27:  # ESC
            break

    if count >= 20:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import os
from secure_storage import save_encrypted

name = input("Enter name: ")

# Create folder if not exists
os.makedirs("faces", exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

count = 0  # number of images

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # Show detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        key = cv2.waitKey(1)

        if key == ord('s'):
            count += 1

            # Convert face to bytes
            _, buffer = cv2.imencode(".jpg", face_img)
            image_bytes = buffer.tobytes()

            # Save multiple images
            file_path = f"faces/{name}_{count}.enc"
            save_encrypted(file_path, image_bytes)

            print(f"Saved: {file_path}")

    cv2.imshow("Capture Face - Press S", frame)

    # Stop after 10 images
    if count >= 10:
        break

    # ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Dataset created successfully!")
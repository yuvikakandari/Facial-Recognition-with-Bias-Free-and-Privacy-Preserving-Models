import cv2
from recognition_deepface import recognize_face

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

# To reduce flickering
prev_label = ""
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        # 🔥 DeepFace recognition
        name, recognized, confidence = recognize_face(face_img)

        # 🎯 Stabilize label (reduces flicker)
        if frame_count % 5 == 0:
            prev_label = name if recognized else "Unknown"

        label = prev_label

        # 🔒 Privacy: blur unknown faces
        if not recognized:
            face_img = cv2.GaussianBlur(face_img, (99, 99), 30)

        # Put back face (blurred or original)
        frame[y:y+h, x:x+w] = face_img

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show label
        display_text = f"{label} ({confidence:.1f}%)" if recognized else "Unknown"

        cv2.putText(frame, display_text, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 255, 0), 2)

    frame_count += 1

    # Show frame
    cv2.imshow("Face Recognition", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
from recognition_deepface import recognize_face

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

y_true = []
y_scores = []

print("Press 'q' to stop recording data...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        name, recognized, confidence = recognize_face(face_img)

        # Ask user for ground truth
        actual = input("Is this a known person? (1=yes, 0=no): ")

        if actual == '1':
            y_true.append(1)
        else:
            y_true.append(0)

        y_scores.append(confidence)

        print(f"Captured → True: {actual}, Score: {confidence:.2f}")

    cv2.imshow("Collecting Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("y_true =", y_true)
print("y_scores =", y_scores)
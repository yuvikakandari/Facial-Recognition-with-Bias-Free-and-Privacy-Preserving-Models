import cv2
from secure_storage import save_encrypted

name = input("Enter name: ")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Capture Face - Press S", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        # Convert image to bytes
        _, buffer = cv2.imencode(".jpg", frame)
        image_bytes = buffer.tobytes()

        # 🔐 SAVE ENCRYPTED
        save_encrypted(f"faces/{name}.enc", image_bytes)
        print("Face saved securely!")
        break

cap.release()
cv2.destroyAllWindows()
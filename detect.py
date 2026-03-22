import cv2
import os

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Pick ANY image automatically from dataset
dataset_path = "dataset"

found = False

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        img = cv2.imread(img_path)

        if img is None:
            continue

        found = True
        break
    if found:
        break

if not found:
    print("❌ No images found in dataset!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

# Show image
cv2.imshow("Face Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []

label_map = {}
current_label = 0

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(current_label)

    current_label += 1

# Train model
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))

# Save model
model.save("face_model.xml")

print("✅ Model trained successfully!")
print("Label map:", label_map)
import json

with open("labels.json", "w") as f:
    json.dump(label_map, f)
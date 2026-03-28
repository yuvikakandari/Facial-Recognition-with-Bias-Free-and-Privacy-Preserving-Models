import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from recognition_deepface import get_embedding

lfw_path = "lfw-deepfunneled"

# ---------------------------
# LOAD IMAGES
# ---------------------------
def load_lfw_images():
    data = {}

    for person in os.listdir(lfw_path):
        person_path = os.path.join(lfw_path, person)

        if not os.path.isdir(person_path):
            continue

        images = []

        for img_name in os.listdir(person_path)[:5]:  # limit images
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            emb = get_embedding(img)
            if emb is not None:
                images.append(emb)

        if len(images) >= 2:
            data[person] = images

    return data


# ---------------------------
# CREATE PAIRS
# ---------------------------
def create_pairs(data):
    genuine = []
    impostor = []

    persons = list(data.keys())

    # Genuine pairs (same person)
    for person in persons:
        embeddings = data[person]

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = 1 - np.dot(embeddings[i], embeddings[j])
                genuine.append(dist)

    # Impostor pairs (different people)
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            emb1 = data[persons[i]][0]
            emb2 = data[persons[j]][0]

            dist = 1 - np.dot(emb1, emb2)
            impostor.append(dist)

    return genuine, impostor


# ---------------------------
# COMPUTE ROC
# ---------------------------
def compute_roc(genuine, impostor):
    thresholds = np.linspace(0, 1, 100)

    tpr = []
    fpr = []

    for t in thresholds:
        tp = sum(d < t for d in genuine)
        fn = sum(d >= t for d in genuine)

        fp = sum(d < t for d in impostor)
        tn = sum(d >= t for d in impostor)

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))

    return fpr, tpr


# ---------------------------
# PLOT ROC
# ---------------------------
def plot_roc(fpr, tpr):
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Face Recognition")
    plt.grid()
    plt.show()


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("[INFO] Loading LFW embeddings...")
    data = load_lfw_images()

    print("[INFO] Creating pairs...")
    genuine, impostor = create_pairs(data)

    print("[INFO] Computing ROC...")
    fpr, tpr = compute_roc(genuine, impostor)

    print("[INFO] Plotting ROC curve...")
    plot_roc(fpr, tpr)
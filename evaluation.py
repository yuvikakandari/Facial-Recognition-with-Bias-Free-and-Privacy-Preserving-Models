import os
import numpy as np
import cv2
from recognition_deepface import get_embedding

lfw_path = "lfw-deepfunneled"

# ---------------------------
# LOAD DATA
# ---------------------------
def load_data():
    data = {}

    for person in os.listdir(lfw_path):
        person_path = os.path.join(lfw_path, person)

        if not os.path.isdir(person_path):
            continue

        embeddings = []

        for img_name in os.listdir(person_path)[:5]:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            emb = get_embedding(img)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) >= 2:
            data[person] = embeddings

    return data


# ---------------------------
# CREATE PAIRS
# ---------------------------
def create_pairs(data):
    y_true = []
    y_pred = []

    threshold = 0.6

    persons = list(data.keys())

    # Genuine pairs
    for person in persons:
        embs = data[person]

        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                dist = 1 - np.dot(embs[i], embs[j])

                y_true.append(1)  # same person
                y_pred.append(1 if dist < threshold else 0)

    # Impostor pairs
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            emb1 = data[persons[i]][0]
            emb2 = data[persons[j]][0]

            dist = 1 - np.dot(emb1, emb2)

            y_true.append(0)  # different
            y_pred.append(1 if dist < threshold else 0)

    return np.array(y_true), np.array(y_pred)


# ---------------------------
# METRICS
# ---------------------------
def compute_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return accuracy, precision, recall, f1, TP, TN, FP, FN


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("[INFO] Loading data...")
    data = load_data()

    print("[INFO] Creating pairs...")
    y_true, y_pred = create_pairs(data)

    print("[INFO] Computing metrics...")
    acc, prec, rec, f1, TP, TN, FP, FN = compute_metrics(y_true, y_pred)

    print("\n===== MODEL EVALUATION =====")
    print(f"Accuracy  : {acc:.3f}")
    print(f"Precision : {prec:.3f}")
    print(f"Recall    : {rec:.3f}")
    print(f"F1 Score  : {f1:.3f}")

    print("\n===== CONFUSION MATRIX =====")
    print(f"TP: {TP}  FP: {FP}")
    print(f"FN: {FN}  TN: {TN}")
import os
import numpy as np
import cv2
from recognition_deepface import get_embedding

lfw_path = "lfw-deepfunneled"

models = ["Facenet", "VGG-Face", "ArcFace"]


def load_data(model_name):
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

            emb = get_embedding(img, model_name)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) >= 2:
            data[person] = embeddings

    return data


def evaluate(data):
    threshold = 0.6

    TP = TN = FP = FN = 0
    persons = list(data.keys())

    # Genuine
    for person in persons:
        embs = data[person]
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                dist = 1 - np.dot(embs[i], embs[j])

                if dist < threshold:
                    TP += 1
                else:
                    FN += 1

    # Impostor
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            dist = 1 - np.dot(data[persons[i]][0], data[persons[j]][0])

            if dist < threshold:
                FP += 1
            else:
                TN += 1

    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP + 1e-6)
    rec = TP / (TP + FN + 1e-6)
    f1 = 2 * prec * rec / (prec + rec + 1e-6)

    return acc, prec, rec, f1


if __name__ == "__main__":
    for model in models:
        print(f"\n===== {model} =====")

        data = load_data(model)
        acc, prec, rec, f1 = evaluate(data)

        print(f"Accuracy  : {acc:.3f}")
        print(f"Precision : {prec:.3f}")
        print(f"Recall    : {rec:.3f}")
        print(f"F1 Score  : {f1:.3f}")
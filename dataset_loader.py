import os
import cv2
from secure_storage import save_encrypted

os.makedirs("faces", exist_ok=True)

def load_dataset(dataset_path="dataset"):
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)

        if not os.path.isdir(person_folder):
            continue

        print(f"[INFO] Processing {person_name}")

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            image = cv2.imread(img_path)
            if image is None:
                continue

            _, buffer = cv2.imencode(".jpg", image)
            image_bytes = buffer.tobytes()

            save_encrypted(
                f"faces/{person_name}_{img_name}.enc",
                image_bytes
            )

    print("✅ Dataset loaded into encrypted storage")

if __name__ == "__main__":
    load_dataset()
import os

dataset_path = "dataset"

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    
    if os.path.isdir(person_path):
        images = os.listdir(person_path)
        print(f"{person} has {len(images)} images")

        for img in images:
            if not img.endswith(".jpg"):
                print(f"Warning: {img} is not a JPG file")
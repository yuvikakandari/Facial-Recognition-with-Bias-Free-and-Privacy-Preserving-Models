import matplotlib.pyplot as plt
from deepface import DeepFace

# Path to two images you want to compare
img1_path = "person1_image1.jpg"
img2_path = "person1_image2.jpg"

# Perform verification
result = DeepFace.verify(img1_path, img2_path, model_name = "Facenet")

# Plotting the result
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(plt.imread(img1_path))
ax[1].imshow(plt.imread(img2_path))

distance = round(result['distance'], 4)
verified = result['verified']

plt.suptitle(f"Verified: {verified} | Cosine Distance: {distance}")
plt.savefig("prediction_sample.png")
plt.show()
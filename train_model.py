import numpy as np
from PIL import Image
import os
import cv2

# Train the LBPH face recognizer using the collected dataset
def train_model(data_folder):
    image_paths = [os.path.join(data_folder, file) for file in os.listdir(data_folder)]
    faces, ids = [], []

    for img_path in image_paths:
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        face_np = np.array(img, 'uint8')
        user_id = int(os.path.split(img_path)[1].split('.')[1])
        faces.append(face_np)
        ids.append(user_id)

    ids = np.array(ids)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, ids)
    recognizer.save("trained_model_dataset.yml")
    print("Training completed. Model saved as trained_model_dataset.yml.")

train_model("data")

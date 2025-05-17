import cv2
import os
import time

# Prompt the user for name and assign a unique user ID
user_name = input("Please enter your full name: ")
user_id = str(abs(hash(user_name)) % 1000)
print(f"Hi {user_name}, your user ID is: {user_id}")

# Create directory if not exists
os.makedirs("data", exist_ok=True)

# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)
sample_count = 0
max_samples = 50

print("Look at the camera. Collecting your images...")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, 1.1, 4)

    for (x, y, w, h) in faces:
        sample_count += 1
        face_img = grayscale[y:y+h, x:x+w]
        cv2.imwrite(f"data/user.{user_id}.{sample_count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        time.sleep(0.2)  # Slow down the capture to avoid missing faces

    cv2.imshow("Capturing Face Data", frame)

    if sample_count >= max_samples or cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"Data collection complete. {sample_count} images saved for {user_name} (ID: {user_id})")

cam.release()
cv2.destroyAllWindows()

import cv2

# Draw a box around detected faces
def detect_faces(image, detector, scale_factor=1.1, min_neighbors=10, box_color=(255, 0, 0), label="Face"):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(grayscale, scale_factor, min_neighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    return image

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open webcam stream
cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()
    if not success:
        break
    frame = detect_faces(frame, face_cascade)
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
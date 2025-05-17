import cv2

def identify_and_label_faces(frame, face_detector, face_recognizer, scale_factor=1.1, min_neighbors=10, rectangle_color=(255, 255, 255)):
    """
    Identifies faces within the given video frame, attempts recognition using the LBPH model,
    and labels each detected face accordingly.
    
    Parameters:
        frame (ndarray): The current frame from the webcam.
        face_detector (CascadeClassifier): The Haar cascade classifier for detecting faces.
        face_recognizer (LBPHFaceRecognizer): Trained face recognizer model.
        scale_factor (float): How much the image size is reduced at each scale.
        min_neighbors (int): Number of neighbors to keep a rectangle.
        rectangle_color (tuple): Color of the rectangle and text for annotation.
        
    Returns:
        frame (ndarray): The annotated frame with labeled faces.
    """
    # Convert the original color frame to grayscale (LBPH works on grayscale images)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect all face-like areas in the frame
    detected_faces = face_detector.detectMultiScale(grayscale, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    # Label dictionary: map trained IDs to human-readable names
    label_map = {
        921: "Vinay",
        954: "Hariom"
    }

    # Confidence threshold: lower value means higher confidence in prediction
    CONFIDENCE_THRESHOLD = 100

    for (x, y, w, h) in detected_faces:
        # Draw a rectangle around each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

        # Extract face region of interest (ROI)
        face_roi = grayscale[y:y + h, x:x + w]

        # Predict the label (ID) and confidence using the recognizer
        predicted_id, confidence = face_recognizer.predict(face_roi)

        # Debug: Uncomment to print prediction details
        # print(f"Predicted ID: {predicted_id}, Confidence: {confidence:.2f}")

        # Check if confidence is acceptable; otherwise treat as unknown
        if confidence < CONFIDENCE_THRESHOLD:
            name_label = label_map.get(predicted_id, "Unknown")
        else:
            name_label = "Unknown"

        # Format the label to show both name and confidence
        display_text = f"{name_label} ({int(confidence)})"

        # Place the label text above the face box
        cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.85, rectangle_color, 2)

    return frame


if __name__ == "__main__":
    # Load Haar Cascade Classifier for detecting faces
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Create and load the pre-trained LBPH Face Recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("trained_model_dataset.yml")  # Ensure this file was created during training

    # Start capturing video from the default webcam (0)
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Face recognition started. Press 'q' or 'ctrl + c' to quit.\n")

    # Continuous loop to process video frames
    while True:
        ret, frame = video_capture.read()

        if not ret:
            print("Failed to grab frame from camera. Exiting...")
            break

        # Identify and label any faces in the current frame
        processed_frame = identify_and_label_faces(frame, face_detector, face_recognizer)

        # Display the output in a window
        cv2.imshow("Live Face Recognition", processed_frame)

        # Stop recognition on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExiting face recognition.")
            break

    # Clean up: release resources and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

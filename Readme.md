
#  Face Detection & Recognition with OpenCV in Python

This project is a simple yet effective face recognition system built using Python and OpenCV. It allows you to:
- Capture and store face images of different users
- Train a face recognizer based on those images
- Recognize faces in real-time through your webcam

---

##  Project Structure

```
face_recognition_project/
├── data/                             # Folder where captured face images are stored
├── collect_faces.py                  # Script to collect and save face images
├── train_model.py                    # Script to train the face recognition model
├── recognize.py                      # Script to recognize faces in real-time
├── haarcascade_frontalface_default.xml  # Pre-trained face detection model
```

---

##  Requirements

- Python 3.7 or higher
- OpenCV with extra modules (for LBPH recognizer)
- Numpy
- Pillow (for image handling)

###  Install dependencies

```bash
pip install opencv-python opencv-contrib-python numpy pillow
```

## How It Works

- **Face Detection** is handled using OpenCV’s Haar Cascade classifier.
- **Face Recognition** uses the **LBPH (Local Binary Patterns Histograms)** algorithm, which is good for smaller datasets and works well under varying lighting conditions.

---

##  Author
 - **Vinay Kumar**

---

"""Prove MediaPipe finds face landmarks and we can compute EAR."""
import mediapipe as mp
import numpy as np
import urllib.request

urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-tasks/face_landmarker/face_landmarker.task",
    "/tmp/face_landmarker.task",
)
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-assets/portrait.jpg", "/tmp/face.jpg"
)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

options = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path="/tmp/face_landmarker.task"))
with FaceLandmarker.create_from_options(options) as landmarker:
    img = mp.Image.create_from_file("/tmp/face.jpg")
    result = landmarker.detect(img)

print(f"Detected {len(result.face_landmarks)} faces.")

# Six landmarks for the right eye in MediaPipe's 478-point mesh
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
lms = result.face_landmarks[0]
pts = np.array([[lms[i].x, lms[i].y] for i in RIGHT_EYE])
ear = (np.linalg.norm(pts[1]-pts[5]) + np.linalg.norm(pts[2]-pts[4])) / (2 * np.linalg.norm(pts[0]-pts[3]))
print(f"Right-eye EAR: {ear:.3f}  (open eye ~0.25–0.35; closed ~0.10)")
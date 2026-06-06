"""Draw ALL Halpe-136 face keypoints (indices 26-93) with their numbers."""
import cv2
from services.worker.keypoints import keypoints_per_frame

SAMPLE = "0001"
FRAME_INDEX = 0
ANN = f"data/aide/AIDE_Dataset/annotation/{SAMPLE}.json"
IMG = f"data/aide/AIDE_Dataset/{SAMPLE}/incarframes/{FRAME_INDEX}.jpg"
OUT = "/tmp/all_face_kpts.jpg"

frames = keypoints_per_frame(ANN)
kp = frames[FRAME_INDEX]
img = cv2.imread(IMG)

# Draw face keypoints (Halpe indices 26-93 = 68 face points)
for i in range(26, 94):
    x, y, conf = float(kp[i, 0]), float(kp[i, 1]), float(kp[i, 2])
    if conf < 0.1:
        continue
    color = (0, 255, 255) if conf > 0.5 else (128, 128, 128)
    cv2.circle(img, (int(x), int(y)), 2, color, -1)
    cv2.putText(img, str(i), (int(x) + 3, int(y) - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

cv2.imwrite(OUT, img)
print(f"Saved {OUT}")
print("Yellow = high-confidence keypoints, grey = low-confidence.")
print("Find the cluster around an eye, note those 6 indices, send me the image.")
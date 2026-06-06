"""Draw the assumed eye-landmark indices on one frame so we can verify by eye."""
import cv2
from services.worker.keypoints import keypoints_per_frame, RIGHT_EYE_IDXS, LEFT_EYE_IDXS

SAMPLE = "0001"
FRAME_INDEX = 22   # mid-clip
ANN = f"data/aide/AIDE_Dataset/annotation/{SAMPLE}.json"
IMG = f"data/aide/AIDE_Dataset/{SAMPLE}/incarframes/{FRAME_INDEX}.jpg"
OUT = "/tmp/eye_check.jpg"

frames = keypoints_per_frame(ANN)
assert frames is not None, "No pose_list in annotation"
kp = frames[FRAME_INDEX]

img = cv2.imread(IMG)
for i in RIGHT_EYE_IDXS:
    x, y = int(kp[i, 0]), int(kp[i, 1])
    cv2.circle(img, (x, y), 4, (0, 0, 255), -1)   # RED on right eye
    cv2.putText(img, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
for i in LEFT_EYE_IDXS:
    x, y = int(kp[i, 0]), int(kp[i, 1])
    cv2.circle(img, (x, y), 4, (0, 255, 0), -1)   # GREEN on left eye
    cv2.putText(img, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imwrite(OUT, img)
print(f"Saved: {OUT}")
print("Open it. Red dots should be around the RIGHT eye, green around the LEFT eye.")
print("If they're not, send me the image — we'll fix the indices.")
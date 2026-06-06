"""See the actual confidence values for eye keypoints across frames + samples."""
from pathlib import Path
import numpy as np
from services.worker.keypoints import (
    keypoints_per_frame, RIGHT_EYE_IDXS, LEFT_EYE_IDXS,
)

SAMPLES = ["0001", "0002", "0003", "0011", "0012", "0013"]

for sid in SAMPLES:
    ann = f"data/aide/AIDE_Dataset/annotation/{sid}.json"
    if not Path(ann).is_file():
        continue
    frames = keypoints_per_frame(ann)
    if not frames:
        print(f"{sid}: no pose_list")
        continue

    rights, lefts = [], []
    for f in frames.values():
        rights.append(f[RIGHT_EYE_IDXS, 2].mean())
        lefts.append(f[LEFT_EYE_IDXS, 2].mean())

    print(f"{sid}:  right_eye_conf  mean={np.mean(rights):.3f}  max={np.max(rights):.3f}  "
          f"|  left_eye_conf  mean={np.mean(lefts):.3f}  max={np.max(lefts):.3f}")
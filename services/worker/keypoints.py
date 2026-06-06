# import json
# import numpy as np
# from pathlib import Path

# RIGHT_EYE_IDXS = [62, 63, 64, 65, 66, 67]
# LEFT_EYE_IDXS  = [68, 69, 70, 71, 72, 73]
# NOSE_IDX, LEYE_IDX, REYE_IDX = 0, 1, 2

# MIN_EYE_CONFIDENCE = 0.3   # below this, we don't trust the keypoint
# MIN_GAZE_CONFIDENCE = 0.3


# def keypoints_per_frame(annotation_path: str) -> dict[int, np.ndarray] | None:
#     with open(annotation_path) as f:
#         ann = json.load(f)
#     pose_list = ann.get("pose_list")
#     if not pose_list:
#         return None

#     out = {}
#     for entry in pose_list:
#         try:
#             frame_idx = int(Path(entry.get("imgname", "")).stem)
#         except (ValueError, AttributeError):
#             continue
#         result = entry.get("result") or []
#         if not result:
#             continue
#         kpts_flat = result[0].get("keypoints")
#         if not kpts_flat or len(kpts_flat) != 408:
#             continue
#         out[frame_idx] = np.asarray(kpts_flat, dtype=float).reshape(136, 3)
#     return out or None


# def _ear(eye_pts: np.ndarray) -> float:
#     p1, p2, p3, p4, p5, p6 = eye_pts
#     return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4))


# def _eye_confident(eye_kpts: np.ndarray, min_conf: float = MIN_EYE_CONFIDENCE) -> bool:
#     """All 6 eye points must be above the confidence threshold."""
#     return bool((eye_kpts[:, 2] >= min_conf).all())


# def derive_blinks(frames: dict[int, np.ndarray], fps: float = 15.0,
#                   ear_threshold_factor: float = 0.7) -> list[dict]:
#     """Only computes EAR on frames where at least one eye is high-confidence."""
#     if not frames:
#         return []
#     ordered_idx = sorted(frames.keys())
#     ordered = [frames[i] for i in ordered_idx]

#     # Compute EAR for each frame; None if no reliable eye data
#     ears: list[float | None] = []
#     for f in ordered:
#         right = f[RIGHT_EYE_IDXS]
#         left  = f[LEFT_EYE_IDXS]
#         right_ok = _eye_confident(right)
#         left_ok  = _eye_confident(left)
#         if not right_ok and not left_ok:
#             ears.append(None)
#         elif right_ok and left_ok:
#             ears.append((_ear(right[:, :2]) + _ear(left[:, :2])) / 2)
#         elif right_ok:
#             ears.append(_ear(right[:, :2]))
#         else:
#             ears.append(_ear(left[:, :2]))

#     # We need at least a handful of valid EARs to calibrate a threshold
#     valid_ears = [e for e in ears if e is not None]
#     if len(valid_ears) < 5:
#         return []   # not enough reliable frames in this clip — skip blink detection

#     threshold = float(np.median(valid_ears)) * ear_threshold_factor

#     blinks, in_blink, start = [], False, 0
#     for i, e in enumerate(ears):
#         if e is None:
#             # Reliability gap. If we were in a blink, close it conservatively.
#             if in_blink:
#                 in_blink = False
#             continue
#         if e < threshold and not in_blink:
#             in_blink, start = True, i
#         elif e >= threshold and in_blink:
#             in_blink = False
#             window = [x for x in ears[start:i+1] if x is not None]
#             blinks.append({
#                 "start_ts": ordered_idx[start] / fps,
#                 "end_ts":   ordered_idx[i] / fps,
#                 "magnitude": float(np.median(valid_ears) - min(window)),
#             })
#     return blinks


# def derive_gaze_off_road(frames: dict[int, np.ndarray], fps: float = 15.0,
#                          yaw_threshold: float = 0.15,
#                          min_duration_s: float = 0.3) -> list[dict]:
#     """Yaw proxy from body keypoints (nose + eyes). Filtered by confidence."""
#     if not frames:
#         return []
#     ordered_idx = sorted(frames.keys())
#     ordered = [frames[i] for i in ordered_idx]

#     yaws: list[float | None] = []
#     for f in ordered:
#         nose, leye, reye = f[NOSE_IDX], f[LEYE_IDX], f[REYE_IDX]
#         # Require all three body keypoints to be confident
#         if (nose[2] < MIN_GAZE_CONFIDENCE or
#             leye[2] < MIN_GAZE_CONFIDENCE or
#             reye[2] < MIN_GAZE_CONFIDENCE):
#             yaws.append(None)
#             continue
#         mid = (leye[:2] + reye[:2]) / 2
#         inter = np.linalg.norm(leye[:2] - reye[:2]) + 1e-6
#         yaws.append(float((nose[0] - mid[0]) / inter))

#     min_frames = max(1, int(min_duration_s * fps))
#     events, in_off, start = [], False, 0
#     for i, y in enumerate(yaws):
#         if y is None:
#             if in_off:
#                 in_off = False
#             continue
#         if abs(y) > yaw_threshold and not in_off:
#             in_off, start = True, i
#         elif abs(y) <= yaw_threshold and in_off:
#             in_off = False
#             if i - start >= min_frames:
#                 window = [yaws[j] for j in range(start, i) if yaws[j] is not None]
#                 events.append({
#                     "start_ts": ordered_idx[start] / fps,
#                     "end_ts":   ordered_idx[i] / fps,
#                     "magnitude": float(max(abs(y_) for y_ in window)),
#                 })
#     return events
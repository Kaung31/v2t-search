"""Run YOLO26 on a folder of frames; return per-frame detections."""
from pathlib import Path
from ultralytics import YOLO

_model = None


def _load() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolo26n.pt")
    return _model


def detect_on_frames(frames_dir: str, fps: float = 15.0,
                     sample_fps: float = 5.0) -> list[dict]:
    """Return detections sampled at sample_fps (default 5fps → ~15 per 3s clip)."""
    model = _load()
    files = sorted(Path(frames_dir).glob("*.jpg"), key=lambda p: int(p.stem))
    step = max(1, int(round(fps / sample_fps)))
    out = []
    for i, p in enumerate(files):
        if i % step != 0:
            continue
        ts = i / fps
        results = model.predict(str(p), verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                out.append({
                    "frame_index": i,
                    "timestamp_s": ts,
                    "class": model.names[int(b.cls[0])],
                    "conf": float(b.conf[0]),
                    "bbox_xyxy": [float(x) for x in b.xyxy[0].tolist()],
                })
    return out

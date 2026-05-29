"""Prove YOLO26 loads and detects objects."""
from ultralytics import YOLO

model = YOLO("yolo26n.pt")          # downloads ~6MB on first run
results = model.predict("/tmp/dog.jpg", verbose=False)  # <--- Changed to dog.jpg

for r in results:
    for b in r.boxes:
        cls = model.names[int(b.cls[0])]
        conf = float(b.conf[0])
        print(f"  {cls}: {conf:.2f}")
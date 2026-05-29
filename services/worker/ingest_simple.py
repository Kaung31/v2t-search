"""
The DUMBEST possible end-to-end ingestion.
No YOLO, no events. Just frames → SigLIP → Qdrant.
We add layers on top of this in later weeks.
"""
import cv2
import hashlib
import sys
import uuid
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/test.mp4"
SAMPLE_FPS = 2.0
COLLECTION = "frames"

# Use MPS on Apple Silicon if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Processing: {VIDEO_PATH} on device: {device}")

content_hash = hashlib.md5(Path(VIDEO_PATH).read_bytes()).hexdigest()[:12]

# --- Qdrant setup ---
qdrant = QdrantClient(host="localhost", port=6333)
if not qdrant.collection_exists(COLLECTION):
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

# --- Load SigLIP ---
print("Loading SigLIP 2...")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-256")
model = AutoModel.from_pretrained("google/siglip2-base-patch16-256").eval().to(device)

# --- Walk video, sample frames, embed, upsert ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
step = max(1, int(fps / SAMPLE_FPS))
frame_idx, sampled = 0, 0
points = []

print(f"Video fps={fps:.1f}, sampling every {step} frames")
while True:
    ok, frame_bgr = cap.read()
    if not ok:
        break
    if frame_idx % step == 0:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        ts = frame_idx / fps

        with torch.no_grad():
            inputs = processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            emb = model.get_image_features(**inputs)
            
            # --- TRANSFORMERS 5.x FIX: Extract the actual tensor from the container ---
            if not isinstance(emb, torch.Tensor):
                emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb[0]
            # --------------------------------------------------------------------------
                
            emb = (emb / emb.norm(dim=-1, keepdim=True))[0].cpu().tolist()

        thumb_path = f"data/thumbnails/{content_hash}_{sampled:05d}.jpg"
        Path(thumb_path).parent.mkdir(parents=True, exist_ok=True)
        pil_thumb = pil_img.copy()
        pil_thumb.thumbnail((320, 320))
        pil_thumb.save(thumb_path)

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"asset_hash": content_hash, "frame_index": frame_idx,
                     "timestamp": ts, "thumbnail": thumb_path},
        ))
        sampled += 1
    frame_idx += 1

cap.release()
print(f"Sampled {sampled} frames. Upserting to Qdrant...")
qdrant.upsert(collection_name=COLLECTION, points=points)
print(f"Done. Indexed {sampled} frames.")
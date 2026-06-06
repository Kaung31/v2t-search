"""For each AIDE sample, run YOLO on the 3 exterior views, embed crops,
upsert to Qdrant. Resumable: skips samples already in the objects collection."""
import csv
import uuid
from pathlib import Path
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from services.worker.detect import detect_on_frames
from services.worker.embed import embed_image, ensure_collections

MANIFEST = "data/aide/manifest.csv"
EXTERIOR_VIEWS = ("front", "left", "right")
COLLECTION = "objects"
QDRANT = QdrantClient(host="localhost", port=6333)


def process_sample(sample_dir: str, sample_id: str) -> int:
    points = []
    for view in EXTERIOR_VIEWS:
        frames_dir = f"{sample_dir}/{view}frames"
        if not Path(frames_dir).is_dir():
            continue
        for det in detect_on_frames(frames_dir):
            frame_path = Path(frames_dir) / f"{det['frame_index']}.jpg"
            if not frame_path.exists():
                continue
            img = Image.open(frame_path)
            x1, y1, x2, y2 = det["bbox_xyxy"]
            crop = img.crop((max(0, x1), max(0, y1),
                             min(img.width, x2), min(img.height, y2)))
            if crop.width < 16 or crop.height < 16:
                continue
            try:
                emb = embed_image(crop)
            except Exception as e:
                print(f"  embed fail {view}/{det['frame_index']}: {e}", flush=True)
                continue
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "sample_id":   sample_id,
                    "view":        view,
                    "frame_index": det["frame_index"],
                    "timestamp_s": det["timestamp_s"],
                    "class":       det["class"],
                    "conf":        det["conf"],
                    "bbox":        det["bbox_xyxy"],
                },
            ))
    if points:
        QDRANT.upsert(collection_name=COLLECTION, points=points)
    return len(points)


def main(limit: int | None = None):
    ensure_collections(QDRANT)
    skipped = 0
    processed = 0
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and processed >= limit:
                break
            existing = QDRANT.count(
                collection_name=COLLECTION,
                count_filter=Filter(must=[
                    FieldCondition(key="sample_id",
                                   match=MatchValue(value=row["sample_id"]))
                ]),
            ).count
            if existing > 0:
                skipped += 1
                continue
            n = process_sample(row["sample_dir"], row["sample_id"])
            processed += 1
            if processed % 10 == 0:
                print(f"  [{processed} done, {skipped} skipped] "
                      f"{row['sample_id']} → {n} crops", flush=True)
    print(f"✅ Done. Processed {processed}, skipped {skipped}.", flush=True)


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)

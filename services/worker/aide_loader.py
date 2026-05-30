"""Walk AIDE and produce data/aide/manifest.csv with all 2898 samples + labels."""
import csv
import json
from pathlib import Path
from typing import Iterator

DATASET_ROOT  = Path("data/aide/AIDE_Dataset")
ANNOTATION_DIR = DATASET_ROOT / "annotation"
SPLITS = {
    "train": Path("data/aide/training.csv"),
    "val":   Path("data/aide/validation.csv"),
    "test":  Path("data/aide/testing.csv"),
}

# The four real label fields in AIDE's JSON files
LABEL_KEYS = {
    "driver_behavior":   "driver_behavior_label",
    "driver_emotion":    "emotion_label",
    "traffic_context":   "scene_centric_context_label",
    "vehicle_condition": "vehicle_based_context_label",
}


def _ids_in_csv(csv_path: Path) -> set[str]:
    """Read a split CSV and return the set of sample ids in it."""
    ids = set()
    if not csv_path.is_file():
        return ids
    with csv_path.open() as f:
        for row in csv.reader(f):
            for cell in row:
                for part in Path(cell.strip()).parts:
                    if part.isdigit():
                        ids.add(part)
                        break
                else:
                    continue
                break
    return ids


def iter_samples() -> Iterator[dict]:
    train_ids = _ids_in_csv(SPLITS["train"])
    val_ids   = _ids_in_csv(SPLITS["val"])
    test_ids  = _ids_in_csv(SPLITS["test"])

    for sample_dir in sorted(DATASET_ROOT.iterdir()):
        if not sample_dir.is_dir() or not sample_dir.name.isdigit():
            continue
        sid = sample_dir.name
        ann_file = ANNOTATION_DIR / f"{sid}.json"
        if not ann_file.is_file():
            continue

        split = ("train" if sid in train_ids else
                 "val"   if sid in val_ids   else
                 "test"  if sid in test_ids  else "unknown")

        yield {
            "sample_id": sid,
            "split": split,
            "sample_dir": str(sample_dir),
            "annotation_path": str(ann_file),
        }


def build_manifest(output_path: str = "data/aide/manifest.csv",
                   limit: int | None = None) -> None:
    rows = []
    for i, sample in enumerate(iter_samples()):
        if limit and i >= limit:
            break
        with open(sample["annotation_path"]) as f:
            ann = json.load(f)
        labels = {ours: ann.get(theirs) for ours, theirs in LABEL_KEYS.items()}
        rows.append({
            "sample_id":       sample["sample_id"],
            "split":           sample["split"],
            "sample_dir":      sample["sample_dir"],
            "annotation_path": sample["annotation_path"],
            **{f"label_{k}": v for k, v in labels.items()},
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"✅ Wrote {len(rows)} samples to {output_path}")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    build_manifest(limit=limit)
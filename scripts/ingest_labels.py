"""Read manifest.csv, register assets in Postgres, insert task-label events."""
import csv
import hashlib
from pathlib import Path
import psycopg

from packages.core.config import settings
from services.worker.events import events_from_task_labels, replace_events

MANIFEST = "data/aide/manifest.csv"


def _conn():
    return psycopg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db,
    )


def get_or_create_asset(conn, sample_id: str, split: str, ann_path: str) -> int:
    h = hashlib.md5(Path(ann_path).read_bytes()).hexdigest()[:12]
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO assets (content_hash, sample_id, split)
            VALUES (%s, %s, %s)
            ON CONFLICT (content_hash) DO UPDATE SET sample_id = EXCLUDED.sample_id
            RETURNING id
        """, (h, sample_id, split))
        return cur.fetchone()[0]


def ensure_clips(conn, asset_id: int, sample_dir: str) -> None:
    with conn.cursor() as cur:
        for view in ("front", "left", "right", "incar"):
            cur.execute("""
                INSERT INTO clips (asset_id, view, video_path, frames_dir)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (asset_id, view) DO NOTHING
            """, (asset_id, view,
                  f"{sample_dir}/{view}.mp4",
                  f"{sample_dir}/{view}frames"))


def main(limit: int | None = None):
    with _conn() as conn, open(MANIFEST) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            asset_id = get_or_create_asset(
                conn, row["sample_id"], row["split"], row["annotation_path"]
            )
            ensure_clips(conn, asset_id, row["sample_dir"])

            task_labels = {
                k.replace("label_", ""): v
                for k, v in row.items()
                if k.startswith("label_") and v
            }
            n = replace_events(asset_id, "task_label",
                               events_from_task_labels(asset_id, task_labels),
                               conn=conn)
            conn.commit()
            if i % 100 == 0:
                print(f"  {i}: {row['sample_id']} → {n} events")
        conn.commit()
    print("✅ Done.")


if __name__ == "__main__":
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit)
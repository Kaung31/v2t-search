"""Process one sample: register asset + clips, write task-label events, mark indexed.

Object embeddings were done by the overnight ingest_objects run, so this pipeline
handles the relational/event side only. Keypoints are parked (low confidence)."""
import json
import psycopg

from packages.core.config import settings
from services.worker.events import events_from_task_labels, replace_events

LABEL_KEYS = {
    "driver_behavior":   "driver_behavior_label",
    "driver_emotion":    "emotion_label",
    "traffic_context":   "scene_centric_context_label",
    "vehicle_condition": "vehicle_based_context_label",
}


def _conn():
    return psycopg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db,
    )


def _get_or_create_asset(conn, fields):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO assets (content_hash, sample_id, split)
            VALUES (%s, %s, %s)
            ON CONFLICT (content_hash) DO UPDATE SET sample_id = EXCLUDED.sample_id
            RETURNING id
        """, (fields["content_hash"], fields["sample_id"], fields["split"]))
        return cur.fetchone()[0]


def _ensure_clips(conn, asset_id, sample_dir):
    with conn.cursor() as cur:
        for view in ("front", "left", "right", "incar"):
            cur.execute("""
                INSERT INTO clips (asset_id, view, video_path, frames_dir, status)
                VALUES (%s, %s, %s, %s, 'pending')
                ON CONFLICT (asset_id, view) DO NOTHING
            """, (asset_id, view,
                  f"{sample_dir}/{view}.mp4",
                  f"{sample_dir}/{view}frames"))


def _mark_indexed(conn, asset_id):
    with conn.cursor() as cur:
        cur.execute("UPDATE clips SET status='indexed' WHERE asset_id=%s", (asset_id,))


def process_sample(fields: dict) -> None:
    with _conn() as conn:
        asset_id = _get_or_create_asset(conn, fields)
        _ensure_clips(conn, asset_id, fields["sample_dir"])
        conn.commit()

        with open(fields["annotation_path"]) as f:
            ann = json.load(f)
        labels = {ours: ann.get(theirs) for ours, theirs in LABEL_KEYS.items()}
        replace_events(asset_id, "task_label",
                       events_from_task_labels(asset_id, labels))

        _mark_indexed(conn, asset_id)
        conn.commit()

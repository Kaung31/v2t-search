"""Convert AIDE task labels + keypoint-derived signals into events table rows."""
from dataclasses import dataclass
from typing import Iterable
import psycopg
from psycopg.types.json import Jsonb

from packages.core.config import settings


# Which view each task label belongs to
TASK_VIEW_MAP = {
    "driver_behavior":   "incar",
    "driver_emotion":    "incar",
    "traffic_context":   "front",
    "vehicle_condition": "front",
}


@dataclass
class Event:
    asset_id: int
    view: str
    event_type: str
    source: str        # 'task_label' | 'keypoint' | 'yolo'
    start_ts: float
    end_ts: float
    magnitude: float | None = None
    payload: dict | None = None


def _conn():
    return psycopg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db,
    )


def replace_events(asset_id: int, source: str, events: Iterable[Event],
                   conn=None) -> int:
    """Idempotent. If a conn is passed, uses it (no commit); otherwise opens its own."""
    sql_del = "DELETE FROM events WHERE asset_id = %s AND source = %s"
    sql_ins = """
        INSERT INTO events (asset_id, view, event_type, source, start_ts, end_ts, magnitude, payload)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    own = conn is None
    if own:
        conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql_del, (asset_id, source))
            count = 0
            for e in events:
                cur.execute(sql_ins, (
                    e.asset_id, e.view, e.event_type, e.source,
                    e.start_ts, e.end_ts, e.magnitude,
                    Jsonb(e.payload) if e.payload else None,
                ))
                count += 1
        if own:
            conn.commit()
        return count
    finally:
        if own:
            conn.close()


def events_from_task_labels(asset_id: int, task_labels: dict,
                            duration_s: float = 3.0) -> list[Event]:
    """Each non-null AIDE task label becomes one event spanning the whole clip."""
    out = []
    for task, label in task_labels.items():
        if not label:
            continue
        event_type = str(label).strip().lower().replace(" ", "_").replace("/", "_")
        out.append(Event(
            asset_id=asset_id,
            view=TASK_VIEW_MAP.get(task, "incar"),
            event_type=event_type,
            source="task_label",
            start_ts=0.0,
            end_ts=duration_s,
            magnitude=1.0,
            payload={"task": task, "raw_label": label},
        ))
    return out
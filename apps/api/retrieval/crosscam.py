"""Cross-camera correspondence retrieval.

Finds clips where a driver-state event (incar) and a road-state event (front) co-occur
in the same 3-second clip. The join is an indexed SQL self-join on the events table.
"""
from dataclasses import dataclass
import psycopg
from psycopg.rows import dict_row

from packages.core.config import settings


@dataclass
class CorrespondenceHit:
    sample_id: str
    driver_state: str
    road_state: str
    score: float
    matched_events: dict


def _conn():
    return psycopg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db, row_factory=dict_row,
    )


def find_correspondences(
    driver_states: list[str] | None = None,
    road_states: list[str] | None = None,
    limit: int = 50,
) -> list[CorrespondenceHit]:
    conditions = [
        "i.view = 'incar'", "i.source = 'task_label'",
        "f.view = 'front'", "f.source = 'task_label'",
    ]
    params: list = []
    if driver_states:
        conditions.append("i.event_type = ANY(%s)")
        params.append(driver_states)
    if road_states:
        conditions.append("f.event_type = ANY(%s)")
        params.append(road_states)

    where = " AND ".join(conditions)
    sql = f"""
        SELECT
            a.sample_id  AS sample_id,
            i.event_type AS driver_state,
            f.event_type AS road_state,
            i.magnitude  AS driver_conf,
            f.magnitude  AS road_conf
        FROM events i
        JOIN events f ON i.asset_id = f.asset_id
        JOIN assets a ON a.id = i.asset_id
        WHERE {where}
        GROUP BY a.sample_id, i.event_type, f.event_type, i.magnitude, f.magnitude
        ORDER BY (COALESCE(i.magnitude,1) + COALESCE(f.magnitude,1)) DESC
        LIMIT %s
    """
    params.append(limit)

    out = []
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        for row in cur.fetchall():
            out.append(CorrespondenceHit(
                sample_id=row["sample_id"],
                driver_state=row["driver_state"],
                road_state=row["road_state"],
                score=float((row["driver_conf"] or 1.0) + (row["road_conf"] or 1.0)) / 2.0,
                matched_events={
                    "incar": {"event_type": row["driver_state"], "view": "incar"},
                    "front": {"event_type": row["road_state"], "view": "front"},
                },
            ))
    return out

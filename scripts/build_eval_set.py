"""Auto-derive ~60 evaluation queries + ground-truth sample sets from the indexed data."""
import json
import psycopg
from psycopg.rows import dict_row
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from packages.core.config import settings

MIN_RELEVANT = 4          # drop queries with fewer relevant clips than this
OUT = "eval/eval_queries.json"


def _pg():
    return psycopg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db, row_factory=dict_row,
    )

QD = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


# ---- query specs ----
# Appearance: (natural language, object_class)
APPEARANCE = [
    ("a truck on the road", "truck"),
    ("a large truck", "truck"),
    ("a pedestrian crossing the street", "person"),
    ("people walking near the road", "person"),
    ("a motorcycle", "motorcycle"),
    ("a motorcyclist on the road", "motorcycle"),
    ("a bus", "bus"),
    ("a bicycle", "bicycle"),
    ("a cyclist", "bicycle"),
    ("a traffic light", "traffic light"),
    ("cars on the road", "car"),
    ("a car ahead", "car"),
]

# Single-state: (natural language, driver_states or None, road_states or None)
SINGLE = [
    ("an anxious driver", ["anxiety"], None),
    ("a driver looking around", ["looking_around"], None),
    ("a driver on the phone", ["making_phone"], None),
    ("a drowsy driver", ["dozing_off", "weariness"], None),
    ("a driver smoking", ["smoking"], None),
    ("an angry driver", ["anger"], None),
    ("a happy driver", ["happiness"], None),
    ("a driver talking", ["talking"], None),
    ("a calm driver", ["peace"], None),
    ("heavy traffic jam", None, ["traffic_jam"]),
    ("a vehicle changing lanes", None, ["changing_lane"]),
    ("a turning vehicle", None, ["turning"]),
    ("smooth flowing traffic", None, ["smooth_traffic"]),
    ("waiting at a light", None, ["waiting"]),
    ("a parked vehicle", None, ["parking"]),
    ("reversing vehicle", None, ["backward_moving"]),
]

# Compound: (natural language, driver_states, road_states)
COMPOUND = [
    ("distracted driver during a lane change", ["looking_around"], ["changing_lane"]),
    ("driver looking around while turning", ["looking_around"], ["turning"]),
    ("driver looking around in a traffic jam", ["looking_around"], ["traffic_jam"]),
    ("driver on the phone while turning", ["making_phone"], ["turning"]),
    ("driver on the phone in a traffic jam", ["making_phone"], ["traffic_jam"]),
    ("driver on the phone while waiting", ["making_phone"], ["waiting"]),
    ("anxious driver changing lanes", ["anxiety"], ["changing_lane"]),
    ("anxious driver in a traffic jam", ["anxiety"], ["traffic_jam"]),
    ("anxious driver while turning", ["anxiety"], ["turning"]),
    ("anxious driver waiting at a light", ["anxiety"], ["waiting"]),
    ("drowsy driver in a traffic jam", ["weariness", "dozing_off"], ["traffic_jam"]),
    ("drowsy driver waiting", ["weariness", "dozing_off"], ["waiting"]),
    ("drowsy driver in smooth traffic", ["weariness", "dozing_off"], ["smooth_traffic"]),
    ("smoking driver in smooth traffic", ["smoking"], ["smooth_traffic"]),
    ("smoking driver while waiting", ["smoking"], ["waiting"]),
    ("smoking driver while turning", ["smoking"], ["turning"]),
    ("looking around while waiting", ["looking_around"], ["waiting"]),
    ("driver on the phone in smooth traffic", ["making_phone"], ["smooth_traffic"]),
    ("weary driver while turning", ["weariness"], ["turning"]),
    ("anxious driver in smooth traffic", ["anxiety"], ["smooth_traffic"]),
]


def relevant_for_structured(driver_states, road_states):
    """sample_ids where the driver and/or road conditions hold (co-occurring in the same clip)."""
    clauses = []
    params = []
    if driver_states:
        clauses.append("""EXISTS (SELECT 1 FROM events di WHERE di.asset_id=a.id
                           AND di.view='incar' AND di.source='task_label'
                           AND di.event_type = ANY(%s))""")
        params.append(driver_states)
    if road_states:
        clauses.append("""EXISTS (SELECT 1 FROM events fr WHERE fr.asset_id=a.id
                           AND fr.view='front' AND fr.source='task_label'
                           AND fr.event_type = ANY(%s))""")
        params.append(road_states)
    where = " AND ".join(clauses)
    sql = f"SELECT a.sample_id FROM assets a WHERE {where}"
    with _pg() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return sorted({r["sample_id"] for r in cur.fetchall()})


def relevant_for_object(object_class, conf=0.4):
    """sample_ids where the object class is detected (from the objects collection)."""
    seen = set()
    offset = None
    while True:
        pts, offset = QD.scroll(
            collection_name="objects",
            scroll_filter=Filter(must=[
                FieldCondition(key="class", match=MatchValue(value=object_class)),
            ]),
            limit=1000, offset=offset, with_payload=True, with_vectors=False,
        )
        for p in pts:
            if p.payload.get("conf", 0) >= conf:
                seen.add(p.payload["sample_id"])
        if offset is None:
            break
    return sorted(seen)


def main():
    queries = []
    qid = 0

    for text, obj in APPEARANCE:
        rel = relevant_for_object(obj)
        if len(rel) >= MIN_RELEVANT:
            qid += 1
            queries.append({"id": qid, "query": text, "type": "appearance",
                            "object_class": obj, "relevant": rel})

    for text, d, r in SINGLE:
        rel = relevant_for_structured(d, r)
        if len(rel) >= MIN_RELEVANT:
            qid += 1
            queries.append({"id": qid, "query": text, "type": "single_state",
                            "driver_states": d, "road_states": r, "relevant": rel})

    for text, d, r in COMPOUND:
        rel = relevant_for_structured(d, r)
        if len(rel) >= MIN_RELEVANT:
            qid += 1
            queries.append({"id": qid, "query": text, "type": "compound",
                            "driver_states": d, "road_states": r, "relevant": rel})

    with open(OUT, "w") as f:
        json.dump(queries, f, indent=2)

    by_type = {}
    for q in queries:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    print(f"✅ Wrote {len(queries)} queries to {OUT}")
    for t, n in by_type.items():
        print(f"   {t}: {n}")
    print("\nRelevant-set sizes (sample):")
    for q in queries[:10]:
        print(f"   [{q['type']:12s}] {q['query'][:40]:40s} → {len(q['relevant'])} clips")


if __name__ == "__main__":
    main()

"""Long-running Redis Streams consumer; idempotent + checkpointed."""
import os, socket, time, redis, psycopg
from packages.core.config import settings
from services.worker.pipeline import process_sample

STREAM, GROUP = "ingest:jobs", "workers"
CONSUMER = os.environ.get("HOSTNAME", socket.gethostname())


def _redis():
    return redis.Redis(
        host=settings.redis_host, port=settings.redis_port,
        decode_responses=True, socket_timeout=30,
    )


def _ensure_group(r):
    try:
        r.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise


def _conn():
    return psycopg.connect(
        host=settings.postgres_host, port=settings.postgres_port,
        user=settings.postgres_user, password=settings.postgres_password,
        dbname=settings.postgres_db,
    )


def _already_indexed(content_hash):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM assets a JOIN clips c ON c.asset_id = a.id "
            "WHERE a.content_hash = %s AND c.status = 'indexed' LIMIT 1",
            (content_hash,),
        )
        return cur.fetchone() is not None


def _handle(r, msg_id, fields):
    if _already_indexed(fields["content_hash"]):
        print(f"  {msg_id} skip ({fields['sample_id']})", flush=True)
        r.xack(STREAM, GROUP, msg_id)
        return
    try:
        process_sample(fields)
        r.xack(STREAM, GROUP, msg_id)
        print(f"  {msg_id} OK {fields['sample_id']}", flush=True)
    except Exception as e:
        print(f"  {msg_id} FAIL {fields['sample_id']}: {e}", flush=True)


def main():
    r = _redis()
    _ensure_group(r)
    print(f"Worker {CONSUMER} consuming from {STREAM}...", flush=True)
    while True:
        try:
            # Reclaim jobs abandoned by crashed workers (idle > 60s)
            claim = r.xautoclaim(STREAM, GROUP, CONSUMER, min_idle_time=60_000, count=10)
            if claim and claim[1]:
                for mid, fld in claim[1]:
                    _handle(r, mid, fld)

            msgs = r.xreadgroup(GROUP, CONSUMER, {STREAM: ">"}, count=1, block=5000)
            if not msgs:
                continue
            for _, entries in msgs:
                for mid, fld in entries:
                    _handle(r, mid, fld)
        except redis.exceptions.TimeoutError:
            # No jobs arrived within the block window — normal, just loop
            continue
        except redis.exceptions.ConnectionError as e:
            print(f"  Redis connection lost: {e}; retrying in 3s", flush=True)
            time.sleep(3)
            r = _redis()
            continue


if __name__ == "__main__":
    main()

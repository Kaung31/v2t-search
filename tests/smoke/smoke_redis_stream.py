"""Prove Redis Streams: produce one message, consume it."""
import redis

r = redis.Redis(host="localhost", port=6379, decode_responses=True)
STREAM = "smoke:jobs"

r.xadd(STREAM, {"asset_id": "test-001", "action": "process"})
print("Produced.")

msgs = r.xread({STREAM: "0"}, count=1)
for stream_name, entries in msgs:
    for msg_id, fields in entries:
        print(f"  Got {msg_id}: {fields}")
        r.xdel(STREAM, msg_id)
print("Redis Streams OK")
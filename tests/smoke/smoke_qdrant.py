"""Prove we can talk to Qdrant: create a collection, upsert, search."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import random

client = QdrantClient(host="localhost", port=6333)
NAME = "smoke_test"

client.recreate_collection(
    collection_name=NAME,
    vectors_config=VectorParams(size=8, distance=Distance.COSINE),
)

points = [
    PointStruct(id=i, vector=[random.random() for _ in range(8)], payload={"i": i})
    for i in range(10)
]
client.upsert(collection_name=NAME, points=points)

query = [random.random() for _ in range(8)]
hits = client.query_points(collection_name=NAME, query=query, limit=3).points
for h in hits:
    print(f"  id={h.id} score={h.score:.3f}")

client.delete_collection(NAME)
print("Qdrant OK")
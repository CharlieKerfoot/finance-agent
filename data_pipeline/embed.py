# /// script
# dependencies = [
#  "qdrant-client",
#  "sentence-transformers",
#  "tqdm",
#  "python-dotenv"
# ]
# ///

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

client = QdrantClient(url="http://localhost:6333")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

COLLECTION = "finance_chunks"

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)

with open("dataset/finance_chunks.jsonl") as f:
    chunks = [json.loads(line) for line in f]

points = []
for idx, chunk in enumerate(tqdm(chunks)):
    embedding = model.encode(chunk['text']).tolist()
    points.append(PointStruct(
        id=idx,
        vector=embedding,
        payload={
            "text": chunk['text'],
            "ticker": chunk['meta']['ticker'],
            "year": chunk['meta']['year']
        }
    ))

    if len(points) >= 100:
        client.upsert(collection_name=COLLECTION, points=points)
        points = []

if points:
    client.upsert(collection_name=COLLECTION, points=points)

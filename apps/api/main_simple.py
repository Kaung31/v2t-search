"""Dumb search API: text query → SigLIP text embed → Qdrant search → return thumbnails."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"API loading on device: {device}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/thumbs", StaticFiles(directory="data/thumbnails"), name="thumbs")

print("Loading SigLIP 2 for queries...")
processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-256")
model = AutoModel.from_pretrained("google/siglip2-base-patch16-256").eval().to(device)
qdrant = QdrantClient(host="localhost", port=6333)

class SearchRequest(BaseModel):
    query: str
    limit: int = 12

@app.post("/search")
def search(req: SearchRequest):
    with torch.no_grad():
        inputs = processor(text=[req.query], return_tensors="pt", padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        emb = model.get_text_features(**inputs)
        
        # --- TRANSFORMERS 5.x FIX: Extract the actual tensor from the container ---
        if not isinstance(emb, torch.Tensor):
            emb = emb.pooler_output if hasattr(emb, "pooler_output") else emb[0]
        # --------------------------------------------------------------------------
            
        emb = (emb / emb.norm(dim=-1, keepdim=True))[0].cpu().tolist()

    hits = qdrant.query_points(collection_name="frames", query=emb, limit=req.limit).points
    return {
        "query": req.query,
        "results": [
            {
                "score": h.score,
                "timestamp": h.payload["timestamp"],
                "thumbnail": f"/thumbs/{h.payload['thumbnail'].split('/')[-1]}",
            }
            for h in hits
        ],
    }
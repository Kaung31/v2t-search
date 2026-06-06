"""SigLIP 2 embedding (image + text) and Qdrant collection setup."""
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from packages.core.config import settings

_processor, _model, _device = None, None, None


def _load():
    global _processor, _model, _device
    if _model is None:
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading SigLIP 2 on {_device}…")
        _processor = AutoProcessor.from_pretrained(settings.siglip_model)
        _model = AutoModel.from_pretrained(settings.siglip_model).eval().to(_device)
    return _processor, _model, _device


def _extract_tensor(result) -> torch.Tensor:
    """Pull the embedding tensor out of whatever object the model returned.
    Handles both raw-tensor and wrapped-output return shapes across transformers versions."""
    if torch.is_tensor(result):
        return result
    # Wrapped output object — try the most common attribute names in priority order
    for attr in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        if hasattr(result, attr):
            t = getattr(result, attr)
            if torch.is_tensor(t):
                return t
    raise TypeError(
        f"Could not extract tensor from {type(result).__name__}. "
        f"Available attributes: {[a for a in dir(result) if not a.startswith('_')][:10]}"
    )


def _normalize(t: torch.Tensor) -> torch.Tensor:
    # If last_hidden_state-shaped (B, seq, dim), pool over sequence first
    if t.dim() == 3:
        t = t.mean(dim=1)
    return t / t.norm(dim=-1, keepdim=True)


def embed_image(pil_img: Image.Image) -> list[float]:
    proc, model, device = _load()
    with torch.no_grad():
        inputs = proc(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        raw = model.get_image_features(**inputs)
        tensor = _extract_tensor(raw)
        normalized = _normalize(tensor)
    return normalized[0].cpu().tolist()


def embed_text(text: str, use_template: bool = True) -> list[float]:
    proc, model, device = _load()
    # SigLIP was trained on "This is a photo of X" style prompts — gives better retrieval
    prompt = f"This is a photo of {text}." if use_template else text
    with torch.no_grad():
        inputs = proc(text=[prompt], return_tensors="pt", padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        raw = model.get_text_features(**inputs)
        tensor = _extract_tensor(raw)
        normalized = _normalize(tensor)
    return normalized[0].cpu().tolist()


def ensure_collections(client: QdrantClient) -> None:
    for name in ("frames", "objects", "events"):
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=settings.siglip_dim,
                                            distance=Distance.COSINE),
            )

"""Prove SigLIP 2 loads and computes a similarity score."""
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import urllib.request

# Use PyTorch's official developer test image (never blocks Python scripts)
url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
urllib.request.urlretrieve(url, "/tmp/dog.jpg")

# Use MPS on Apple Silicon if available, otherwise CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model_id = "google/siglip2-base-patch16-256"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).eval().to(device)

image = Image.open("/tmp/dog.jpg")
texts = ["a photo of a dog", "a photo of a truck", "a person riding a bicycle"]

with torch.no_grad():
    inputs = processor(text=texts, images=image, return_tensors="pt", padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # SigLIP uses sigmoid (independent per-pair scores), not softmax
    probs = torch.sigmoid(outputs.logits_per_image)

for t, p in zip(texts, probs[0].tolist()):
    print(f"  {t}: {p:.3f}")
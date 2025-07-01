import os
import sys
import json
import torch
import random
from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.models as models
import torchvision.transforms as T
#from submit import submit
# Percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..")))

from metrics import build_filename_to_class_mapping, top_k_accuracy, precision_at_k

GALLERY_DIR = os.path.join(BASE_DIR, "..", "..", "data_preEval", "training")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "..", "results", "ResNet", "RN50", "submission.json")
TOP_K = 10
N_QUERIES = 20

# Trasformazioni
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Caricamento modello ResNet50
print("🔍 Caricamento modello ResNet50...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()
model.eval().to(device)

def load_images_from_flat_folder(folder):
    images, filenames = [], []
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if os.path.isfile(path) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = transform(Image.open(path).convert("RGB"))
                images.append(img)
                filenames.append(fname)
            except Exception as e:
                print(f"Errore con {fname}: {e}")
    return torch.stack(images).to(device), filenames

def extract_features(model, images, batch_size=16):
    features_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            feats = model(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.cpu())
    return torch.cat(features_list)

# Carica immagini dalla gallery
print("📥 Caricamento immagini gallery...")
gallery_images, gallery_filenames = load_images_from_flat_folder(GALLERY_DIR)
print(f"✅ Gallery: {len(gallery_images)} immagini")

# Seleziona 20 immagini a caso come query
indices = random.sample(range(len(gallery_images)), min(N_QUERIES, len(gallery_images)))
query_images = gallery_images[indices]
query_filenames = [gallery_filenames[i] for i in indices]

# Estrazione feature
print("📈 Estrazione feature gallery e query...")
gallery_features = extract_features(model, gallery_images)
query_features = extract_features(model, query_images)

# Calcola similarità
print("🔎 Retrieval...")
similarity = query_features @ gallery_features.T
_, topk_indices = similarity.topk(TOP_K, dim=1)

# Prepara risultati
results = []
for i, query_name in enumerate(query_filenames):
    top_gallery = [gallery_filenames[idx] for idx in topk_indices[i]]
    results.append({"filename": query_name, "samples": top_gallery})

# Salva JSON
print("💾 Salvataggio file JSON...")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Fatto! Output salvato in {OUTPUT_FILE}")
#submit(results, "Py.tatine")

dataset_dir = GALLERY_DIR

results_dict = {entry["filename"]: entry["samples"] for entry in results}

filename_mapping = build_filename_to_class_mapping(dataset_dir)

acc = top_k_accuracy(results_dict, filename_mapping)
prec = precision_at_k(results_dict, filename_mapping)

print("Accuracy=", acc)
print("Precision=", prec)
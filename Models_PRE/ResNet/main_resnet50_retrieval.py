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

dataset_dir = os.path.join(BASE_DIR, "..", "data_preEval", "training")

def build_filename_to_class_mapping(dataset_dir):
    """
    Costruisce una mappa: nome_file.jpg → nome_classe (cartella)
    Scorre tutte le sottocartelle e associa il nome del file alla sua classe.
    """
    mapping = {}
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                class_name = os.path.basename(root)
                mapping[f] = class_name
    return mapping

def top_k_accuracy(res, filename_to_class, k=10):
    """
    Top-k accuracy: almeno 1 immagine rilevata ha la stessa classe della query.
    """
    correct = 0
    total = 0
    for qfile, retrieved_files in res.items():
        q_class = filename_to_class.get(qfile)
        if q_class is None:
            continue  # file non trovato nella mappa
        retrieved_classes = [filename_to_class.get(f) for f in retrieved_files[:k]]
        if q_class in retrieved_classes:
            correct += 1
        total += 1
    acc = correct / total if total > 0 else 0.0
    print(f"[METRIC] Top-{k} Accuracy: {acc:.4f}")
    return acc

def precision_at_k(res, filename_to_class, k=10):
    """
    Precision@k: media delle proporzioni di immagini rilevate che hanno la stessa classe della query.
    """
    total_precision = 0
    total_queries = 0
    for qfile, retrieved_files in res.items():
        q_class = filename_to_class.get(qfile)
        if q_class is None:
            continue
        retrieved_classes = [filename_to_class.get(f) for f in retrieved_files[:k]]
        correct = sum(1 for c in retrieved_classes if c == q_class)
        total_precision += correct / k
        total_queries += 1
    avg_precision = total_precision / total_queries if total_queries > 0 else 0.0
    print(f"[METRIC] Precision@{k}: {avg_precision:.4f}")
    return avg_precision


filename_mapping = build_filename_to_class_mapping(dataset_dir)

acc= top_k_accuracy(results, filename_mapping)
prec= precision_at_k(results, filename_mapping)

print("Accuracy=", acc)
print("Precision=", prec)
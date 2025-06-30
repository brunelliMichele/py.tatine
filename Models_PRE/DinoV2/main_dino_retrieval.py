import os
import sys
import random
import json
import torch
from PIL import Image
import timm
import torchvision.transforms as T
#from submit import submit

# Percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))

GALLERY_DIR = os.path.join(BASE_DIR, "..", "..", "data_preEval", "training")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "..", "results", "DINO", "submission.json")
TOP_K = 10
NUM_QUERY = 20

# Trasformazioni
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Caricamento modello DINOv2
print("🔍 Caricamento modello DINOv2...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("vit_base_patch16_224.dino", pretrained=True)
model.eval().to(device)

# Funzioni
def load_images_from_folder(folder):
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

def extract_cls(model, images):
    with torch.no_grad():
        out = model.forward_features(images)
        cls_tokens = out[:, 0]
        return cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)

# --- Caricamento immagini gallery ---
print("📥 Caricamento immagini gallery...")
gallery_images_all, gallery_filenames_all = load_images_from_folder(GALLERY_DIR)
print(f"✅ Gallery: {len(gallery_filenames_all)} immagini totali")

# --- Estrazione feature per tutte ---
print("📈 Estrazione feature...")
gallery_features_all = extract_cls(model, gallery_images_all)

# --- Selezione 20 immagini random come query ---
indices = random.sample(range(len(gallery_filenames_all)), NUM_QUERY)
query_filenames = [gallery_filenames_all[i] for i in indices]
query_features = gallery_features_all[indices]
print(f"🔍 Scelte {NUM_QUERY} immagini random dalla gallery come query")

# --- Retrieval ---
results = []
print("🔎 Retrieval...")
similarity = query_features @ gallery_features_all.T
topk_values, topk_indices = similarity.topk(TOP_K, dim=1)

for i, qname in enumerate(query_filenames):
    top_gallery_files = [gallery_filenames_all[idx] for idx in topk_indices[i]]
    results.append({
        "filename": qname,
        "samples": top_gallery_files
    })

# --- Salvataggio JSON ---
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
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
sys.path.insert(0, os.path.abspath(os.path.join(BASE_DIR, "..")))
from metrics import build_filename_to_class_mapping, precision_at_k, top_k_accuracy

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
    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, fname)  # ✅ FIX HERE
                try:
                    img = transform(Image.open(path).convert("RGB"))
                    images.append(img)
                    filenames.append(os.path.basename(fname))  # keep basename for metric match
                except Exception as e:
                    print(f"❌ Error loading {fname}: {e}")
    print(f"✅ Loaded {len(images)} images.")
    if not images:
        raise RuntimeError("❌ No images were loaded — check folder path or file types.")
    return torch.stack(images).to(device), filenames

def extract_cls(model, images, batch_size=64):
    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            out = model.forward_features(batch)
            cls_tokens = out[:, 0]
            cls_tokens = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)
            all_features.append(cls_tokens)
    return torch.cat(all_features, dim=0)

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


dataset_dir = GALLERY_DIR

results_dict = {entry["filename"]: entry["samples"] for entry in results}

filename_mapping = build_filename_to_class_mapping(dataset_dir)
print("🔎 Classes found:", set(filename_mapping.values()))
print("🗂 Mapping keys:", list(filename_mapping.keys())[:5])
print("📄 Query filenames:", [r['filename'] for r in results[:5]])
q = results[0]
print("Query:", q["filename"], "| Class:", filename_mapping.get(q["filename"]))
print("Retrieved:", q["samples"])
print("Retrieved classes:", [filename_mapping.get(f) for f in q["samples"]])


acc = top_k_accuracy(results_dict, filename_mapping)
prec = precision_at_k(results_dict, filename_mapping)

print("Accuracy=", acc)
print("Precision=", prec)
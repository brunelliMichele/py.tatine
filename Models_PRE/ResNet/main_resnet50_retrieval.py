import os
import sys
# In alto, sotto import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from submit import submit
import subprocess
import json
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as T

# Percorsi
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUERY_DIR = os.path.join(BASE_DIR, "..", "..", "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "..", "..", "data", "test", "gallery")
TRAIN_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "training"))
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "..", "results", "ResNet", "RN50", "submission.json")
TOP_K = 10
MODEL_PATH = os.path.join(BASE_DIR, "..", "resnet50_finetuned.pth")
TRAINING_SCRIPT = os.path.join(BASE_DIR, "..", "ResNet-Fine-Tuning", "resNet_fine_tuning.py")

# Controlla se il modello esiste
if not os.path.exists(MODEL_PATH):
    print("⚠️ Modello non trovato. Eseguo fine-tuning...")
    subprocess.run(["python", TRAINING_SCRIPT], check=True)

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
torch.cuda.empty_cache()
model = models.resnet50(pretrained=True)
model.fc = torch.nn.Identity()  # Rimuove il classificatore
model.eval().to(device)


def load_images(folder, use_subfolders=False):
    images, filenames = [], []
    if use_subfolders:
        for cls in sorted(os.listdir(folder)):
            cls_folder = os.path.join(folder, cls)
            if not os.path.isdir(cls_folder):
                continue
            for fname in sorted(os.listdir(cls_folder)):
                path = os.path.join(cls_folder, fname)
                if os.path.isfile(path) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        img = transform(Image.open(path).convert("RGB"))
                        images.append(img)
                        filenames.append((cls, fname))
                    except Exception as e:
                        print(f"Errore con {fname}: {e}")
    else:
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
    model.eval()
    features_list = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch = batch.to(device)
            feats = model(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.cpu())

    return torch.cat(features_list)


def show_retrieval_results(query_dir, gallery_dir, results):
    for item in results:
        if '/' in item['filename']:
            query_cls, query_fname = item['filename'].split('/', 1)
            query_path = os.path.join(query_dir, query_cls, query_fname)
        else:
            query_path = os.path.join(query_dir, item['filename'])

        query_img = Image.open(query_path).convert("RGB")

        gallery_imgs = []
        for path in item['samples']:
            if '/' in path:
                cls, fname = path.split('/', 1)
                img_path = os.path.join(gallery_dir, cls, fname)
            else:
                img_path = os.path.join(gallery_dir, path)

            gallery_imgs.append(Image.open(img_path).convert("RGB"))

        fig, axes = plt.subplots(1, len(gallery_imgs) + 1, figsize=(15, 5))
        axes[0].imshow(query_img)
        axes[0].set_title("Query")
        axes[0].axis('off')

        for i, img in enumerate(gallery_imgs):
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f"Top {i+1}")
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.show()


print("📥 Caricamento immagini gallery...")
gallery_images, gallery_filenames = load_images(GALLERY_DIR, use_subfolders=False)
print(f"✅ Gallery: {len(gallery_images)} immagini")

print("📈 Estrazione feature gallery...")
gallery_features = extract_features(model, gallery_images, batch_size=16)

print("📥 Caricamento immagini query...")
query_images, query_filenames = load_images(QUERY_DIR, use_subfolders=False)
print(f"✅ Query: {len(query_images)} immagini")

print("🔎 Retrieval...")
query_features = extract_features(model, query_images, batch_size=16)
similarity = query_features @ gallery_features.T
topk_values, topk_indices = similarity.topk(TOP_K, dim=1)

results = {}
for i, query_fname in enumerate(query_filenames):
    if isinstance(gallery_filenames[0], tuple):
        top_gallery_files = [f"{gallery_filenames[idx][0]}/{gallery_filenames[idx][1]}" for idx in topk_indices[i]]
    else:
        top_gallery_files = [gallery_filenames[idx] for idx in topk_indices[i]]    
    results[query_fname] = top_gallery_files

print("💾 Salvataggio file JSON...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Fatto! Output salvato in {OUTPUT_FILE}")
submit(results, "Py.tatine")


# Mostra risultati
# show_retrieval_results(QUERY_DIR, GALLERY_DIR, results)
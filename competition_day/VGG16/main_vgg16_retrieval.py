import os
import json
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as T
from tensorflow.keras.models import load_model

# Controlla se esiste il modello fine-tuned
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNED_MODEL_PATH = os.path.join(BASE_DIR, "..", "vgg16_finetuned_from_script.keras")
if not os.path.exists(FINETUNED_MODEL_PATH):
    print("⚠️  Modello fine-tuned non trovato, lo creo ora...")
    import subprocess
    subprocess.run(["python", os.path.join(BASE_DIR, "..", "VGG16_fine_tuned", "vgg16_fine_tuning.py")], check=True)


# Percorsi
QUERY_DIR = os.path.join(BASE_DIR, "..", "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "..", "data", "test", "gallery")
OUTPUT_FILE = os.path.join(BASE_DIR, "submission.json")
TOP_K = 3

# Trasformazioni
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("🔍 Caricamento modello VGG16 fine-tuned...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(FINETUNED_MODEL_PATH)
model = torch.nn.Sequential(*list(models.vgg16().features)).eval().to(device)  # Dummy per compatibilità



def extract_features(model, images):
    with torch.no_grad():
        features = model(images)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        return features / features.norm(dim=-1, keepdim=True)

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

print("📥 Caricamento immagini gallery...")
gallery_images, gallery_filenames = load_images_from_folder(GALLERY_DIR)
print(f"✅ Gallery: {len(gallery_images)} immagini")

print("📈 Estrazione feature gallery...")
gallery_features = extract_features(model, gallery_images)

print("📥 Caricamento immagini query...")
query_images, query_filenames = load_images_from_folder(QUERY_DIR)
print(f"✅ Query: {len(query_images)} immagini")

print("🔎 Retrieval...")
query_features = extract_features(model, query_images)
similarity = query_features @ gallery_features.T
topk_values, topk_indices = similarity.topk(TOP_K, dim=1)

results = []
for i, query_fname in enumerate(query_filenames):
    top_gallery_files = [gallery_filenames[idx] for idx in topk_indices[i]]
    results.append({
        "filename": query_fname,
        "samples": top_gallery_files
    })

print("💾 Salvataggio file JSON...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Fatto! Output salvato in {OUTPUT_FILE}")

# --- Visualizzazione risultati ---
def show_retrieval_results(query_dir, gallery_dir, results):
    for item in results:
        query_img = Image.open(os.path.join(query_dir, item['filename'])).convert("RGB")
        gallery_imgs = [Image.open(os.path.join(gallery_dir, fname)).convert("RGB") for fname in item['samples']]

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

show_retrieval_results(QUERY_DIR, GALLERY_DIR, results)
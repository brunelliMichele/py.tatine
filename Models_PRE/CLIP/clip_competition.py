import os
import sys
import random
import glob
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from submit import submit
from metrics import build_filename_to_class_mapping, precision_at_k

# Paths
QUERY_DIR = os.path.join(BASE_DIR, "..", "..", "data_preEval", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "..", "..", "data_preEval", "test", "gallery")
TRAIN_DIR = os.path.join(BASE_DIR, "..", "..", "data_preEval", "training")

print(f"Resolved QUERY_DIR: {QUERY_DIR}")
print(f"Resolved GALLERY_DIR: {GALLERY_DIR}")
print(f"📁 Looking for training data in: {TRAIN_DIR}")
if not os.path.exists(GALLERY_DIR):
    raise FileNotFoundError(f"❌ Directory not found: {GALLERY_DIR}")
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "..", "results", "CLIP", "RN50x64", "submission.json")

TOP_K = 10

# Utility to sample random queries from training
def sample_random_queries_from_training(training_dir, num_queries=20, preprocess_fn=None):
    all_images = []
    for root, _, files in os.walk(training_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(root, file))

    if len(all_images) < num_queries:
        raise ValueError(f"Not enough images in {training_dir} to sample {num_queries} queries. Found {len(all_images)}.")

    selected_paths = random.sample(all_images, num_queries)
    selected_filenames = [os.path.basename(p) for p in selected_paths]
    images = [preprocess_fn(Image.open(p).convert("RGB")) for p in selected_paths] if preprocess_fn else selected_paths
    return images, selected_filenames, selected_paths

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x64", device=device)


def load_images_from_folder(folder):
    print(f"🔍 Trying to load images from: {folder}")
    if not os.path.exists(folder):
        raise FileNotFoundError(f"❌ Directory not found: {folder}")
    images = []
    filenames = []
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if os.path.isfile(path) and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                img = preprocess(Image.open(path).convert("RGB"))
                images.append(img)
                filenames.append(fname)
            except Exception as e:
                print(f"Errore con {fname}: {e}")
    return images, filenames

def show_retrieval_results(query_dir, gallery_dir, results):
    for query_fname, gallery_list in results.items():
        query_img = Image.open(os.path.join(query_dir, query_fname)).convert("RGB")
        gallery_imgs = [Image.open(os.path.join(gallery_dir, fname)).convert("RGB") for fname in gallery_list]

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


def extract_clip_features(model, images, batch_size=16):
    model.eval()
    all_features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch = torch.stack(batch).to(device)
            feats = model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_features.append(feats.cpu())
    return torch.cat(all_features).to(device)

print("🔄 Caricamento immagini gallery...")
gallery_images, gallery_filenames = load_images_from_folder(GALLERY_DIR)
print(f"✅ {len(gallery_images)} immagini caricate nella gallery")

print("🔄 Estrazione feature gallery...")
gallery_features = extract_clip_features(model, gallery_images)


print("🔄 Selezione casuale di 20 immagini query dal training set...")
query_images, query_filenames, query_paths = sample_random_queries_from_training(TRAIN_DIR, 20, preprocess)
print(f"✅ {len(query_images)} immagini query selezionate")

print("🔄 Estrazione feature query e retrieval...")
results = {}

query_features = extract_clip_features(model, query_images)

similarity = query_features @ gallery_features.T  # (num_query, num_gallery)
topk_values, topk_indices = similarity.topk(TOP_K, dim=1)

for i, query_fname in enumerate(query_filenames):
    top_gallery_files = [gallery_filenames[idx] for idx in topk_indices[i]]
    results[query_fname] = top_gallery_files

print("💾 Salvataggio file JSON...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Fatto! Output salvato in {OUTPUT_FILE}")

file_name_mapping = build_filename_to_class_mapping(TRAIN_DIR)
accuracy = precision_at_k(results, file_name_mapping, 10)

print(accuracy)

# submit(results, "Py.tatine")

# chiama la funzione per visualizzare le immagini
# show_retrieval_results(QUERY_DIR, GALLERY_DIR, results)
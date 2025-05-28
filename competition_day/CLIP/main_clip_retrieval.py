import os
import sys
# In alto, sotto import os
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

# Paths

QUERY_DIR = os.path.join(BASE_DIR, "..", "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "..", "data", "test", "gallery")
TRAIN_DIR = os.path.join(BASE_DIR, "..", "data", "training")
print(f"Resolved QUERY_DIR: {QUERY_DIR}")
print(f"Resolved GALLERY_DIR: {GALLERY_DIR}")
print(f"Resolved TRAIN_DIR: {TRAIN_DIR}")
if not os.path.exists(GALLERY_DIR):
    raise FileNotFoundError(f"❌ Directory not found: {GALLERY_DIR}")
OUTPUT_FILE = os.path.join(BASE_DIR, "submission.json")
TOP_K = 10

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

class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, folder, preprocess):
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self.preprocess = preprocess
        for idx, label in enumerate(sorted(os.listdir(folder))):
            class_dir = os.path.join(folder, label)
            if os.path.isdir(class_dir):
                self.label_map[label] = idx
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, fname))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = self.preprocess(Image.open(self.image_paths[idx]).convert("RGB"))
        label = self.labels[idx]
        return img, label

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

print("🧪 Preparazione dataloader per training...")
train_dataset = SimpleImageDataset(TRAIN_DIR, preprocess)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Add a linear classification head for fine-tuning
projection_head = torch.nn.Linear(model.visual.output_dim, len(train_dataset.label_map)).to(device)

# 🚀 Inizio fine-tuning CLIP con classificazione supervisionata...
print("🚀 Inizio fine-tuning CLIP con classificazione supervisionata...")
model.train()
params = list(model.parameters()) + list(projection_head.parameters())
optimizer = torch.optim.AdamW(params, lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(1):  # Cambia il numero di epoche se serve
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features = model.encode_image(images).float()
        logits = projection_head(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📉 Fine epoca: loss media = {total_loss / len(train_loader):.4f}")
model.eval()

print("🔄 Caricamento immagini query...")
query_images, query_filenames = load_images_from_folder(QUERY_DIR)
print(f"✅ {len(query_images)} immagini query caricate")

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

submit(results, "Py.tatine")

# chiama la funzione per visualizzare le immagini
# show_retrieval_results(QUERY_DIR, GALLERY_DIR, results)
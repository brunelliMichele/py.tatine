import numpy as np
import tensorflow as tf
import cv2
import os
import sys
import random
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
#from submit import submit
import json
from annoy import AnnoyIndex

# --- Load EfficientNetB0 (pretrained, no top) ---
efficientnet_model = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3)
)
model = tf.keras.Sequential([
    efficientnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
])

def extract_features(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img, verbose=0)
    return features.flatten()

# --- New: Provide the single folder with all images ---
IMAGE_DIR = os.path.join(BASE_DIR, "..", "..", "data_preEval", "training")

print(f"📁 Looking for images in: {IMAGE_DIR}")

all_images = []
for root, _, files in os.walk(IMAGE_DIR):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append(os.path.join(root, filename))

if len(all_images) < 20:
    raise ValueError("❌ Not enough images found in the folder. Need at least 20 images.")

print(f"🔍 Found {len(all_images)} images in total.")

# --- Randomly select 20 images as queries ---
random.seed(42)  # For reproducibility
query_images = random.sample(all_images, 20)

# --- Extract features for all images ---
print("⏳ Extracting features for all images...")
features = []
for img_path in all_images:
    features.append(extract_features(img_path))
print("✅ Feature extraction complete.")

# --- Build Annoy index ---
feature_dim = len(features[0])
annoy_index = AnnoyIndex(feature_dim, 'euclidean')
for i, vec in enumerate(features):
    annoy_index.add_item(i, vec)
annoy_index.build(n_trees=10)

top_n = 10
results = {}

# --- For each query, find top 10 similar excluding itself ---
print("🔎 Performing retrieval for queries...")
for query_path in query_images:
    query_idx = all_images.index(query_path)
    query_features = features[query_idx]
    
    # Get top_n+1 neighbors to exclude the query image itself
    neighbors = annoy_index.get_nns_by_vector(query_features, top_n + 1)
    
    # Remove the query image index from neighbors
    neighbors = [idx for idx in neighbors if idx != query_idx][:top_n]
    
    query_fname = os.path.basename(query_path).replace("\\", "/")
    gallery_fnames = [os.path.basename(all_images[idx]).replace("\\", "/") for idx in neighbors]
    results[query_fname] = gallery_fnames

# --- Save results ---
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "..", "results", "EfficientNet", "B0", "submission.json")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Retrieval complete. Results saved to '{OUTPUT_FILE}'.")
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
print("🔎 Classes found in mapping:", set(filename_mapping.values()))
print("🗂 Example mapping keys:", list(filename_mapping.keys())[:5])
print("📄 Example result keys:", list(results.keys())[:5])
for query, retrieved in list(results.items())[:1]:
    print("Query:", query, "| Class:", filename_mapping.get(query))
    print("Retrieved:", retrieved)
    print("Retrieved classes:", [filename_mapping.get(r) for r in retrieved])
acc= top_k_accuracy(results, filename_mapping)
prec= precision_at_k(results, filename_mapping)
print("Accuracy=", acc)
print("Precision=", prec)

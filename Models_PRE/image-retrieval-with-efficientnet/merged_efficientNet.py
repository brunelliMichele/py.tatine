import os
import sys
import random
import json
import cv2
import numpy as np
import tensorflow as tf
from annoy import AnnoyIndex

# --- Paths and Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
#from submit import submit

# --- EfficientNetB4 Model ---
efficientnet_model = tf.keras.applications.EfficientNetB4(
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
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return features.flatten()

# --- Parameters ---
INPUT_FOLDER = os.path.join(BASE_DIR, "..", "..", "data_preEval", "training")  # Replace with your folder path
TOP_N = 10
NUM_QUERIES = 20

# --- Collect all images ---
print(f"📁 Collecting images from: {INPUT_FOLDER}")
all_images = [
    os.path.join(root, file)
    for root, _, files in os.walk(INPUT_FOLDER)
    for file in files if file.lower().endswith(('.jpg', '.jpeg', '.png'))
]
if len(all_images) < NUM_QUERIES:
    raise ValueError(f"❌ Not enough images in the folder to sample {NUM_QUERIES} queries.")

# --- Randomly sample 20 queries ---
query_images = random.sample(all_images, NUM_QUERIES)
query_set = set(query_images)

# --- Extract features for all images ---
print("🔍 Extracting features for all images...")
features = [extract_features(p) for p in all_images]
feature_dim = len(features[0])

# --- Build Annoy Index ---
annoy_index = AnnoyIndex(feature_dim, 'angular')
for i, vec in enumerate(features):
    annoy_index.add_item(i, vec)
annoy_index.build(200)
print("✅ Annoy index built.")

# --- Perform retrieval ---
results = {}
for i, query_path in enumerate(query_images):
    print(f"\n🔎 Processing query {i+1}/{len(query_images)}: {os.path.basename(query_path)}")
    query_feature = extract_features(query_path)
    idxs = annoy_index.get_nns_by_vector(query_feature, TOP_N + 1)  # +1 to exclude self
    query_idx = all_images.index(query_path)
    
    # Filter out the query image from the results
    filtered_idxs = [idx for idx in idxs if idx != query_idx][:TOP_N]
    similar_paths = [all_images[idx] for idx in filtered_idxs]

    query_fname = os.path.basename(query_path).replace("\\", "/")
    gallery_fnames = [os.path.basename(p).replace("\\", "/") for p in similar_paths]
    results[query_fname] = gallery_fnames

# --- Save Results ---
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "..", "results", "EfficientNet", "B4", "custom_submission.json")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Retrieval complete. Results saved to '{OUTPUT_FILE}'.")
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
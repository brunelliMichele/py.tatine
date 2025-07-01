import numpy as np
import tensorflow as tf
import cv2
import os
import sys
# In alto, sotto import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from submit import submit
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

# --- Feature extraction function ---
def extract_features(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img, verbose=0)
    return features.flatten()

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
QUERY_DIR = os.path.join(BASE_DIR, "..", "..", "data", "test", "query")
GALLERY_DIR = os.path.join(BASE_DIR, "..", "..", "data", "test", "gallery")
TRAIN_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "training"))

print(f"📁 Looking for gallery images in: {GALLERY_DIR}")
print(f"📁 Looking for query images in: {QUERY_DIR}")
query_image_count = sum(1 for _, _, files in os.walk(QUERY_DIR) for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
if query_image_count == 0:
    raise ValueError(f"❌ No images found in query directory: {QUERY_DIR}")

top_n = 10

# --- Extract gallery features ---
gallery_features = []
gallery_filenames = []

for root, _, files in os.walk(GALLERY_DIR):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, filename)
            gallery_filenames.append(path)
            gallery_features.append(extract_features(path))
if not gallery_filenames:
    raise ValueError(f"❌ No images found in gallery: {GALLERY_DIR}")
print(f"🔍 Found {len(gallery_filenames)} gallery images.")

# --- Build Annoy index ---
feature_dim = len(gallery_features[0])
annoy_index = AnnoyIndex(feature_dim, 'euclidean')
for i, vec in enumerate(gallery_features):
    annoy_index.add_item(i, vec)
annoy_index.build(n_trees=10)

# --- Process query images and prepare results ---
results = {}

for root, _, files in os.walk(QUERY_DIR):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            query_path = os.path.join(root, filename)
            query_features = extract_features(query_path)

            similar_indices = annoy_index.get_nns_by_vector(query_features, top_n)

            query_fname = os.path.basename(query_path).replace("\\", "/")
            gallery_fnames = [os.path.basename(gallery_filenames[idx]).replace("\\", "/") for idx in similar_indices]

            results[query_fname] = gallery_fnames

# --- Write to JSON file in the same folder as the script ---
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "..", "..", "results", "EfficientNet", "B0", "submission.json")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Retrieval complete. Results saved to '{OUTPUT_FILE}'.")
# submit(results, "Py.tatine")

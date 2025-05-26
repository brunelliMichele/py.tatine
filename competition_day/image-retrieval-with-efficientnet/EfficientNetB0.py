import numpy as np
import tensorflow as tf
import cv2
import os
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
gallery_dir = "data/test/gallery"
query_dir = "data/test/query"
top_n = 10

# --- Extract gallery features ---
gallery_features = []
gallery_filenames = []

for root, _, files in os.walk(gallery_dir):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(root, filename)
            gallery_filenames.append(path)
            gallery_features.append(extract_features(path))

# --- Build Annoy index ---
feature_dim = len(gallery_features[0])
annoy_index = AnnoyIndex(feature_dim, 'euclidean')
for i, vec in enumerate(gallery_features):
    annoy_index.add_item(i, vec)
annoy_index.build(n_trees=10)

# --- Process query images and prepare results ---
results = []

for root, _, files in os.walk(query_dir):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            query_path = os.path.join(root, filename)
            query_features = extract_features(query_path)

            similar_indices = annoy_index.get_nns_by_vector(query_features, top_n)

            result = {
                "filename": query_path.replace("\\", "/"),  # Normalize for JSON
                "gallery_images": [gallery_filenames[idx].replace("\\", "/") for idx in similar_indices]
            }
            results.append(result)

# --- Write to JSON file in the same folder as the script ---
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, "retrieval_resultsB0.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Retrieval complete. Results saved to '{output_file}'.")



